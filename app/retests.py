"""Late retest corrections and withdrawals.

A retest batch is bound to the *samples input version* that was current when
the upload was submitted.  Corrections never modify the original samples
CSV nor any published report; they are kept as an append-only revision
history per (samples version, sample).

Revision rules
--------------

* Revisions for one sample only move forward.
* Submitting an already recorded revision with byte-for-byte equivalent
  content (status and exact values) is an idempotent replay: it is accepted
  but does **not** create another effective revision.
* A revision number that already exists with different content (including a
  changed status) rejects the whole batch.
* A late, never-delivered revision older than the sample's latest revision
  rejects the whole batch: it cannot be ordered against what is already
  known.
* A higher revision may restore a previously withdrawn retest by using
  status ``生效`` again.
* Any conflict rejects the entire batch; previously applied revisions stay
  untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .inputs import (
    RETEST_STATUS_EFFECTIVE,
    RETEST_STATUS_WITHDRAWN,
    RetestInput,
    RetestRow,
    SamplesInput,
    fraction_to_text,
)


class RetestError(ValueError):
    """A retest batch cannot be applied as submitted."""


class RetestConflictError(RetestError):
    """The batch conflicts with the existing revision history."""


class RetestStaleVersionError(RetestError):
    """The samples input was replaced while the batch was being uploaded."""


@dataclass(frozen=True)
class RetestRevision:
    batch_id: int
    samples_version: int
    sample_id: str
    revision: int
    status: str
    values: tuple[Fraction, ...] | None
    created_at: str

    @classmethod
    def from_row(cls, row) -> "RetestRevision":
        encoded = row["values_json"]
        values = None
        if encoded is not None:
            values = tuple(Fraction(value) for value in json.loads(encoded))
        return cls(
            batch_id=row["batch_id"],
            samples_version=row["samples_version"],
            sample_id=row["sample_id"],
            revision=row["revision"],
            status=row["status"],
            values=values,
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class RetestPlan:
    new_revisions: tuple[RetestRow, ...]
    replayed: int


@dataclass(frozen=True)
class RetestBatchResult:
    batch_id: int
    samples_version: int
    applied: int
    replayed: int


@dataclass(frozen=True)
class RetestSampleView:
    """One current-version sample together with its effective retest state."""

    sample_id: str
    original: tuple[Fraction, ...]
    effective_values: tuple[Fraction, ...] | None
    differences: tuple[Fraction, ...] | None
    latest_revision: int | None
    latest_status: str | None
    changed: bool  # effective retest differs from the original in some element

    @property
    def is_withdrawn(self) -> bool:
        return self.latest_status == RETEST_STATUS_WITHDRAWN


def align_retests(
    parsed: RetestInput, samples: SamplesInput, samples_version: int
) -> tuple[RetestRow, ...]:
    """Match retest element columns to the submitted samples input by name.

    Returns rows whose ``values`` follow ``samples.elements`` order.  An
    element mismatch or an unknown sample id rejects the whole batch.
    """

    missing = [name for name in samples.elements if name not in parsed.elements]
    extra = [name for name in parsed.elements if name not in samples.elements]
    if missing or extra:
        details = []
        if missing:
            details.append("复测缺少元素列：" + "、".join(missing))
        if extra:
            details.append("复测多出元素列：" + "、".join(extra))
        raise RetestError(
            "复测文件的元素列必须与当前成品输入一一对应；"
            + "；".join(details)
            + "。成品元素为：" + "、".join(samples.elements)
        )

    index = [parsed.elements.index(name) for name in samples.elements]
    known = set(samples.sample_ids)
    aligned: list[RetestRow] = []
    for row in parsed.rows:
        if row.sample_id not in known:
            raise RetestError(
                f"第 {row.line_no} 行：样本编号“{row.sample_id}”不在当前成品输入"
                f"（v{samples_version}）中"
            )
        values = None
        if row.values is not None:
            values = tuple(row.values[i] for i in index)
        aligned.append(replace(row, values=values))
    return tuple(aligned)


def _same_content(left: RetestRow, right: RetestRevision) -> bool:
    return left.status == right.status and left.values == right.values


def plan_retests(
    existing: Mapping[str, Sequence[RetestRevision]],
    rows: Sequence[RetestRow],
) -> RetestPlan:
    """Validate a batch against the revision history, without writing."""

    history: dict[str, dict[int, RetestRevision]] = {
        sample_id: {revision.revision: revision for revision in revisions}
        for sample_id, revisions in existing.items()
    }
    seen_in_batch: set[str] = set()
    new_revisions: list[RetestRow] = []
    replayed = 0

    for row in rows:
        if row.sample_id in seen_in_batch:
            raise RetestConflictError(
                f"第 {row.line_no} 行：样本编号“{row.sample_id}”在同一批中重复；"
                "批内冲突会拒绝整批"
            )
        seen_in_batch.add(row.sample_id)

        revisions = history.get(row.sample_id, {})
        recorded = revisions.get(row.revision)
        if recorded is not None:
            if _same_content(row, recorded):
                replayed += 1
                continue
            raise RetestConflictError(
                f"第 {row.line_no} 行：样本“{row.sample_id}”的修订号 "
                f"{row.revision} 已存在且内容不同；同一修订号不能改内容"
            )

        latest = max(revisions) if revisions else None
        if latest is not None and row.revision < latest:
            raise RetestConflictError(
                f"第 {row.line_no} 行：样本“{row.sample_id}”的修订号 "
                f"{row.revision} 低于当前最新修订 {latest}，且该修订号从未送达；"
                "迟到的旧修订不能作为首次提交"
            )
        new_revisions.append(row)

    return RetestPlan(new_revisions=tuple(new_revisions), replayed=replayed)


def build_sample_views(
    samples: SamplesInput,
    revisions: Iterable[RetestRevision],
    *,
    samples_version: int,
) -> list[RetestSampleView]:
    """One view per sample of the current samples input.

    Only revisions recorded against ``samples_version`` are considered;
    retests belonging to an older samples input never leak into the view.
    """

    latest: dict[str, RetestRevision] = {}
    for revision in revisions:
        if revision.samples_version != samples_version:
            continue
        previous = latest.get(revision.sample_id)
        if previous is None or revision.revision > previous.revision:
            latest[revision.sample_id] = revision

    views: list[RetestSampleView] = []
    for sample_id, original in zip(samples.sample_ids, samples.rows):
        revision = latest.get(sample_id)
        if revision is None:
            views.append(
                RetestSampleView(
                    sample_id=sample_id,
                    original=tuple(original),
                    effective_values=None,
                    differences=None,
                    latest_revision=None,
                    latest_status=None,
                    changed=False,
                )
            )
            continue

        if revision.status == RETEST_STATUS_EFFECTIVE:
            assert revision.values is not None
            effective = tuple(revision.values)
            differences = tuple(
                new - old for new, old in zip(effective, original)
            )
            views.append(
                RetestSampleView(
                    sample_id=sample_id,
                    original=tuple(original),
                    effective_values=effective,
                    differences=differences,
                    latest_revision=revision.revision,
                    latest_status=revision.status,
                    changed=any(value != 0 for value in differences),
                )
            )
        else:
            views.append(
                RetestSampleView(
                    sample_id=sample_id,
                    original=tuple(original),
                    effective_values=None,
                    differences=None,
                    latest_revision=revision.revision,
                    latest_status=revision.status,
                    changed=False,
                )
            )
    return views


def view_to_payload(
    view: RetestSampleView,
    samples_version: int,
    *,
    report_on_same_version: bool,
) -> dict[str, object]:
    """JSON-ready representation used by the API.

    ``changed`` is a property of the retest itself.  ``needs_review`` also
    requires the latest report to reference this same samples version;
    otherwise the difference cannot be attributed to this correction.
    """

    def encode(values: tuple[Fraction, ...] | None) -> list[str] | None:
        return None if values is None else [fraction_to_text(v) for v in values]

    return {
        "samples_version": samples_version,
        "sample_id": view.sample_id,
        "original": encode(view.original),
        "effective_values": encode(view.effective_values),
        "differences": encode(view.differences),
        "latest_revision": view.latest_revision,
        "latest_status": view.latest_status,
        "changed": view.changed,
        "needs_review": view.changed and report_on_same_version,
    }
