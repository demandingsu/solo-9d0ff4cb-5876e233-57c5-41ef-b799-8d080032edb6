"""SQLite persistence.

Two tables carry the whole state:

* ``input_versions`` – append-only versions of each input.  Exactly one row
  per ``kind`` is flagged ``is_current``.  A report names the version ids it
  was computed from, so "is this report stale?" is always a comparison
  against the current versions rather than a flag that could drift.
* ``reports`` – only fully computed reports are ever inserted.  The insert
  re-checks that the snapshot versions are still current *inside* the same
  ``BEGIN IMMEDIATE`` transaction that writes the report, which is what
  makes publishing safe when an upload interleaves with a calculation.
* ``retest_batches`` / ``retest_revisions`` – append-only late retest
  corrections, each bound to the *samples input version* current at upload
  time.  Applying a batch re-checks that binding inside the write
  transaction; a batch submitted while the samples input is being replaced
  therefore fails as a whole.  ``(samples_version, sample_id, revision)``
  is unique, so one revision can never be recorded twice for one version.

All writes from this process are serialised by a re-entrant lock; SQLite's
own locking covers other processes/workers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .inputs import fraction_to_text
from .retests import (
    RetestBatchResult,
    RetestConflictError,
    RetestRevision,
    RetestStaleVersionError,
    plan_retests,
)

KINDS = ("baseline", "samples")


class StoreError(RuntimeError):
    """A persistence problem surfaced to the UI."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class VersionedInput:
    version_id: int
    kind: str
    filename: str | None
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    baseline: VersionedInput
    samples: VersionedInput


@dataclass(frozen=True)
class ReportRecord:
    report_id: int
    baseline_version: int
    samples_version: int
    created_at: str
    payload: dict[str, Any]
    baseline_is_current: bool
    samples_is_current: bool

    @property
    def is_stale(self) -> bool:
        return not (self.baseline_is_current and self.samples_is_current)


@dataclass(frozen=True)
class ReviewState:
    """Consistent snapshot used by the review page.

    Everything comes from one read transaction: the current inputs, the
    retest history bound to the current samples version, and the latest
    report.  The page therefore cannot combine a just-committed retest
    batch with a newer input or an older report.
    """

    current: dict[str, "VersionedInput | None"]
    revisions: list[RetestRevision]
    report: "ReportRecord | None"
    # Revisions recorded against a *different* samples version.  They are
    # deliberately excluded from the review; this count lets the page say
    # so instead of silently pretending they do not exist.
    other_version_revisions: int = 0

    @property
    def samples(self) -> "VersionedInput | None":
        return self.current["samples"]


class Store:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self._write_lock = threading.RLock()
        self._initialized = False

    # -- connections -----------------------------------------------------

    def _ensure_ready(self) -> None:
        """Create the directory and schema on first use.

        Deferring this keeps merely importing the app (or constructing a
        store) free of filesystem side effects.
        """

        if self._initialized:
            return
        with self._write_lock:
            if self._initialized:
                return
            if self.path.parent and str(self.path.parent) not in ("", "."):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS input_versions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind        TEXT    NOT NULL
                                    CHECK (kind IN ('baseline', 'samples')),
                        filename    TEXT,
                        payload     TEXT    NOT NULL,
                        created_at  TEXT    NOT NULL,
                        is_current  INTEGER NOT NULL DEFAULT 0
                                    CHECK (is_current IN (0, 1))
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS
                        input_versions_one_current
                    ON input_versions (kind)
                    WHERE is_current = 1;

                    CREATE TABLE IF NOT EXISTS reports (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        baseline_version  INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        samples_version   INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        payload           TEXT    NOT NULL,
                        created_at        TEXT    NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS retest_batches (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        samples_version INTEGER NOT NULL
                                        REFERENCES input_versions (id),
                        filename        TEXT,
                        rows_total      INTEGER NOT NULL,
                        rows_applied    INTEGER NOT NULL,
                        rows_replayed   INTEGER NOT NULL,
                        created_at      TEXT    NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS retest_revisions (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id        INTEGER NOT NULL
                                        REFERENCES retest_batches (id),
                        samples_version INTEGER NOT NULL
                                        REFERENCES input_versions (id),
                        sample_id       TEXT    NOT NULL,
                        revision        INTEGER NOT NULL CHECK (revision >= 1),
                        status          TEXT    NOT NULL
                                        CHECK (status IN ('生效', '撤回')),
                        values_json     TEXT,
                        created_at      TEXT    NOT NULL,
                        UNIQUE (samples_version, sample_id, revision)
                    );

                    CREATE INDEX IF NOT EXISTS retest_revisions_by_version
                    ON retest_revisions (samples_version);
                    """
                )
            except sqlite3.Error as exc:
                raise StoreError(f"初始化数据库失败：{exc}") from exc
            finally:
                conn.close()
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        self._ensure_ready()
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            self._ensure_ready()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # -- inputs ----------------------------------------------------------

    def save_inputs(
        self, updates: dict[str, tuple[dict[str, Any], str | None]]
    ) -> dict[str, int]:
        """Replace one or both inputs in a single transaction.

        Whole-file replacement plus atomicity: either every supplied input
        gets a new current version, or none of them change.
        """

        for kind in updates:
            if kind not in KINDS:
                raise ValueError(f"unknown input kind: {kind!r}")
        encoded = {
            kind: json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            for kind, (payload, _filename) in updates.items()
        }
        now = _utcnow()
        new_ids: dict[str, int] = {}
        try:
            with self._write() as conn:
                for kind in sorted(encoded):
                    _payload, filename = updates[kind]
                    conn.execute(
                        "UPDATE input_versions SET is_current = 0 "
                        "WHERE kind = ? AND is_current = 1",
                        (kind,),
                    )
                    cursor = conn.execute(
                        "INSERT INTO input_versions "
                        "(kind, filename, payload, created_at, is_current) "
                        "VALUES (?, ?, ?, ?, 1)",
                        (kind, filename, encoded[kind], now),
                    )
                    new_ids[kind] = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存输入失败：{exc}") from exc
        return new_ids

    def save_input(
        self, kind: str, payload: dict[str, Any], filename: str | None
    ) -> int:
        return self.save_inputs({kind: (payload, filename)})[kind]

    def _current_row(self, conn: sqlite3.Connection, kind: str):
        return conn.execute(
            "SELECT id, kind, filename, payload, created_at "
            "FROM input_versions WHERE kind = ? AND is_current = 1",
            (kind,),
        ).fetchone()

    @staticmethod
    def _row_to_input(row: sqlite3.Row) -> VersionedInput:
        return VersionedInput(
            version_id=row["id"],
            kind=row["kind"],
            filename=row["filename"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
        )

    def current_inputs(self) -> dict[str, VersionedInput | None]:
        with self._read() as conn:
            result: dict[str, VersionedInput | None] = {}
            for kind in KINDS:
                row = self._current_row(conn, kind)
                result[kind] = self._row_to_input(row) if row else None
            return result

    def snapshot(self) -> Snapshot | None:
        """Consistent read of both current inputs, or ``None`` if incomplete."""

        with self._read() as conn:
            rows = {kind: self._current_row(conn, kind) for kind in KINDS}
            if rows["baseline"] is None or rows["samples"] is None:
                return None
            return Snapshot(
                baseline=self._row_to_input(rows["baseline"]),
                samples=self._row_to_input(rows["samples"]),
            )

    # -- reports ---------------------------------------------------------

    @staticmethod
    def _versions_are_current(
        conn: sqlite3.Connection, baseline_version: int, samples_version: int
    ) -> bool:
        rows = {
            "baseline": conn.execute(
                "SELECT id FROM input_versions "
                "WHERE kind = 'baseline' AND is_current = 1"
            ).fetchone(),
            "samples": conn.execute(
                "SELECT id FROM input_versions "
                "WHERE kind = 'samples' AND is_current = 1"
            ).fetchone(),
        }
        return (
            rows["baseline"] is not None
            and rows["samples"] is not None
            and rows["baseline"]["id"] == baseline_version
            and rows["samples"]["id"] == samples_version
        )

    def publish_report(
        self,
        snapshot: Snapshot,
        payload: dict[str, Any],
    ) -> int | None:
        """Insert a complete report if, and only if, its inputs are current.

        Returns the new report id, or ``None`` when an upload won the race.
        The check and the insert share one ``BEGIN IMMEDIATE`` transaction,
        so an upload cannot slip in between them.
        """

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                if not self._versions_are_current(
                    conn,
                    snapshot.baseline.version_id,
                    snapshot.samples.version_id,
                ):
                    return None
                cursor = conn.execute(
                    "INSERT INTO reports "
                    "(baseline_version, samples_version, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        snapshot.baseline.version_id,
                        snapshot.samples.version_id,
                        encoded,
                        _utcnow(),
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存报告失败：{exc}") from exc

    def latest_report(self) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT r.id, r.baseline_version, r.samples_version, "
                "       r.payload, r.created_at, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples "
                "FROM reports r ORDER BY r.id DESC LIMIT 1"
            ).fetchone()
            return self._row_to_report(row) if row else None

    def get_report(self, report_id: int) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT r.id, r.baseline_version, r.samples_version, "
                "       r.payload, r.created_at, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples "
                "FROM reports r WHERE r.id = ?",
                (report_id,),
            ).fetchone()
            return self._row_to_report(row) if row else None

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> ReportRecord:
        return ReportRecord(
            report_id=row["id"],
            baseline_version=row["baseline_version"],
            samples_version=row["samples_version"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            baseline_is_current=row["current_baseline"]
            == row["baseline_version"],
            samples_is_current=row["current_samples"] == row["samples_version"],
        )

    # -- retest corrections ---------------------------------------------

    @staticmethod
    def _retest_rows(
        conn: sqlite3.Connection, samples_version: int
    ) -> list[RetestRevision]:
        rows = conn.execute(
            "SELECT batch_id, samples_version, sample_id, revision, status, "
            "       values_json, created_at "
            "FROM retest_revisions WHERE samples_version = ? "
            "ORDER BY sample_id, revision",
            (samples_version,),
        ).fetchall()
        return [RetestRevision.from_row(row) for row in rows]

    def retest_history(
        self, samples_version: int
    ) -> list[RetestRevision]:
        """All recorded revisions for one samples version, oldest first.

        Retests are keyed by samples version, so a different (e.g. newer)
        samples input can never see or inherit them.
        """

        with self._read() as conn:
            return self._retest_rows(conn, samples_version)

    def review_state(self) -> "ReviewState":
        """One consistent read for the retest review page.

        Returns the current samples input, the retest history bound to it,
        and the latest report record, all from the same read transaction so
        the page cannot mix a committed retest batch with a newer input.
        """

        with self._read() as conn:
            current: dict[str, VersionedInput | None] = {}
            for kind in KINDS:
                row = self._current_row(conn, kind)
                current[kind] = self._row_to_input(row) if row else None
            samples = current["samples"]
            revisions = (
                self._retest_rows(conn, samples.version_id) if samples else []
            )
            report_row = conn.execute(
                "SELECT r.id, r.baseline_version, r.samples_version, "
                "       r.payload, r.created_at, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples "
                "FROM reports r ORDER BY r.id DESC LIMIT 1"
            ).fetchone()
            report = self._row_to_report(report_row) if report_row else None
            other = (
                conn.execute(
                    "SELECT COUNT(*) FROM retest_revisions "
                    "WHERE samples_version != ?",
                    (samples.version_id,),
                ).fetchone()[0]
                if samples
                else conn.execute(
                    "SELECT COUNT(*) FROM retest_revisions"
                ).fetchone()[0]
            )
            return ReviewState(
                current=current,
                revisions=revisions,
                report=report,
                other_version_revisions=int(other),
            )

    def apply_retests(
        self,
        samples_version: int,
        rows: Sequence[Any],
        filename: str | None,
    ) -> RetestBatchResult:
        """Apply one retest batch atomically, or apply nothing at all.

        The check that ``samples_version`` is still the current samples
        input happens inside the same ``BEGIN IMMEDIATE`` transaction that
        writes the batch, so a samples upload racing this call makes the
        whole batch fail with :class:`RetestStaleVersionError`.
        """

        def encode(row: Any) -> str | None:
            if row.values is None:
                return None
            return json.dumps(
                [fraction_to_text(value) for value in row.values],
                ensure_ascii=False,
                separators=(",", ":"),
            )

        now = _utcnow()
        try:
            with self._write() as conn:
                current = conn.execute(
                    "SELECT id FROM input_versions "
                    "WHERE kind = 'samples' AND is_current = 1"
                ).fetchone()
                if current is None or current["id"] != samples_version:
                    raise RetestStaleVersionError(
                        "提交期间成品输入已被替换；本次复测订正已整批拒绝，"
                        "此前有效的复测记录保持不变"
                    )

                history = self._retest_rows(conn, samples_version)
                grouped: dict[str, list[RetestRevision]] = {}
                for revision in history:
                    grouped.setdefault(revision.sample_id, []).append(revision)
                plan = plan_retests(grouped, rows)

                if not plan.new_revisions:
                    # An all-replay batch is a success but writes nothing.
                    return RetestBatchResult(
                        batch_id=0,
                        samples_version=samples_version,
                        applied=0,
                        replayed=plan.replayed,
                    )

                cursor = conn.execute(
                    "INSERT INTO retest_batches "
                    "(samples_version, filename, rows_total, rows_applied, "
                    " rows_replayed, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        samples_version,
                        filename,
                        len(rows),
                        len(plan.new_revisions),
                        plan.replayed,
                        now,
                    ),
                )
                batch_id = int(cursor.lastrowid)
                for row in plan.new_revisions:
                    conn.execute(
                        "INSERT INTO retest_revisions "
                        "(batch_id, samples_version, sample_id, revision, "
                        " status, values_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            samples_version,
                            row.sample_id,
                            row.revision,
                            row.status,
                            encode(row),
                            now,
                        ),
                    )
                return RetestBatchResult(
                    batch_id=batch_id,
                    samples_version=samples_version,
                    applied=len(plan.new_revisions),
                    replayed=plan.replayed,
                )
        except (RetestConflictError, RetestStaleVersionError):
            raise
        except sqlite3.Error as exc:
            raise StoreError(f"保存复测失败：{exc}") from exc
