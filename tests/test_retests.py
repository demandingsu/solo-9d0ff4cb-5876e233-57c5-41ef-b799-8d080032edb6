"""Tests for late retest corrections and withdrawals."""

from __future__ import annotations

import sqlite3
from fractions import Fraction
from urllib.parse import unquote
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import Store, StoreError
from app.inputs import (
    InputValidationError,
    SamplesInput,
    parse_retests,
    parse_samples,
)
from app.main import create_app
from app.retests import (
    RetestConflictError,
    RetestRevision,
    RetestStaleVersionError,
    align_retests,
    build_sample_views,
    plan_retests,
)

BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

SAMPLES_CSV = """sample_id,Fe,SiO2,Al2O3,CaO
S1,62.74,3.71,2.02,1.13
S2,60,4,2,1
S3,63.5,3.55,1.93,1.1
"""


def retest_csv(*rows: str) -> bytes:
    return (
        "sample_id,revision,status,Fe,SiO2,Al2O3,CaO\n" + "\n".join(rows) + "\n"
    ).encode()


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        test_client.app_state = app.state  # type: ignore[attr-defined]
        yield test_client


def upload_inputs(client: TestClient):
    response = client.post(
        "/inputs/upload",
        files={
            "baseline_file": ("baseline.csv", BASE_CSV.encode(), "text/csv"),
            "samples_file": ("samples.csv", SAMPLES_CSV.encode(), "text/csv"),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303


def compute(client: TestClient):
    return client.post("/compute", follow_redirects=False)


def post_retests(client: TestClient, content: bytes, version=None):
    data = {}
    if version is not None:
        data["samples_version"] = str(version)
    return client.post(
        "/retests/upload",
        files={"retest_file": ("retests.csv", content, "text/csv")},
        data=data,
        follow_redirects=False,
    )


def retest_api(client: TestClient) -> dict:
    return client.get("/api/latest").json()["retests"]


def fracs(values):
    return None if values is None else [Fraction(value) for value in values]


# -- parsing -------------------------------------------------------------


def test_parse_retest_ok():
    parsed = parse_retests(
        retest_csv("S1,1,生效,62.7,3.7,2.0,1.1", "S2,2,撤回,,,,")
    )
    assert parsed.elements == ("Fe", "SiO2", "Al2O3", "CaO")
    assert parsed.rows[0].status == "生效"
    assert parsed.rows[0].values == (
        Fraction("62.7"),
        Fraction("3.7"),
        Fraction("2.0"),
        Fraction("1.1"),
    )
    assert parsed.rows[1].is_withdrawal
    assert parsed.rows[1].values is None


@pytest.mark.parametrize(
    "rows,match",
    [
        (["S1,1,无效,1,2,3,4"], "状态只能是"),
        (["S1,0,生效,1,2,3,4"], "修订号必须从 1 开始"),
        (["S1,abc,生效,1,2,3,4"], "修订号必须是正整数"),
        (["S1,99999999999,生效,1,2,3,4"], "修订号过大"),
        (["S1,1,生效,1,2,3"], "应有 7 列"),
        (["S1,1,撤回,1,,,"], "撤回行不填含量"),
        (["S1,1,生效,1,2,3,"], "含量不能为空"),
        (["S1,1,生效,-1,2,3,4"], "不能为负数"),
        (["S1,1,生效,1,2,3,4", "S1,2,生效,1,2,3,4"], "重复"),
        ([",1,生效,1,2,3,4"], "样本编号不能为空"),
    ],
)
def test_parse_retest_rejects_bad_rows(rows, match):
    with pytest.raises(InputValidationError, match=match):
        parse_retests(retest_csv(*rows))


def test_parse_retest_rejects_bad_header():
    with pytest.raises(InputValidationError, match="7 列"):
        parse_retests(b"sample_id,revision,status,Fe\nS1,1,\xe7\x94\x9f\xe6\x95\x88,1\n")


def test_parse_retest_rejects_header_only_and_bom():
    with pytest.raises(InputValidationError, match="没有数据行"):
        parse_retests(b"a,b,c,d,e,f,g\n")
    raw = b"\xef\xbb\xbf" + retest_csv("S1,1,撤回,,,,")
    assert parse_retests(raw).rows[0].is_withdrawal


def test_parse_retest_requires_utf8():
    with pytest.raises(InputValidationError, match="UTF-8"):
        parse_retests("a,b,c,d,e,f,g\nS1,1,生效,1,2,3,4\n".encode("gb18030"))


# -- revision plan -------------------------------------------------------


def _revision(sample_id: str, revision: int, status: str, values=None):
    return RetestRevision(
        batch_id=1,
        samples_version=1,
        sample_id=sample_id,
        revision=revision,
        status=status,
        values=None if values is None else tuple(Fraction(v) for v in values),
        created_at="2026-01-01T00:00:00+00:00",
    )


def _rows(*specs):
    from app.inputs import RetestRow

    result = []
    for index, (sample_id, revision, status, values) in enumerate(specs, start=2):
        result.append(
            RetestRow(
                sample_id=sample_id,
                revision=revision,
                status=status,
                values=None if values is None else tuple(Fraction(v) for v in values),
                line_no=index,
            )
        )
    return result


def test_plan_accepts_monotonic_revisions():
    plan = plan_retests(
        {"S1": [_revision("S1", 1, "生效", ["1", "2", "3", "4"])]},
        _rows(("S1", 2, "生效", ["1", "2", "3", "5"])),
    )
    assert [row.revision for row in plan.new_revisions] == [2]
    assert plan.replayed == 0


def test_plan_accepts_identical_replay_without_new_revision():
    plan = plan_retests(
        {"S1": [_revision("S1", 1, "生效", ["1", "2", "3", "4"])]},
        _rows(("S1", 1, "生效", ["1", "2", "3", "4"])),
    )
    assert plan.new_revisions == ()
    assert plan.replayed == 1


def test_plan_rejects_same_revision_different_content():
    with pytest.raises(RetestConflictError, match="内容不同"):
        plan_retests(
            {"S1": [_revision("S1", 1, "生效", ["1", "2", "3", "4"])]},
            _rows(("S1", 1, "生效", ["1", "2", "3", "5"])),
        )


def test_plan_rejects_same_revision_different_status():
    with pytest.raises(RetestConflictError, match="内容不同"):
        plan_retests(
            {"S1": [_revision("S1", 2, "撤回", None)]},
            _rows(("S1", 2, "生效", ["1", "2", "3", "4"])),
        )


def test_plan_rejects_late_never_delivered_old_revision():
    with pytest.raises(RetestConflictError, match="低于当前最新修订 2"):
        plan_retests(
            {
                "S1": [
                    _revision("S1", 2, "生效", ["1", "2", "3", "5"]),
                ]
            },
            _rows(("S1", 1, "生效", ["1", "2", "3", "5"])),
        )


def test_plan_rejects_intra_batch_duplicate_sample():
    with pytest.raises(RetestConflictError, match="批中重复"):
        plan_retests(
            {},
            _rows(
                ("S1", 1, "生效", ["1", "2", "3", "4"]),
                ("S1", 2, "生效", ["1", "2", "3", "5"]),
            ),
        )


def test_plan_higher_revision_restores_withdrawn():
    plan = plan_retests(
        {
            "S1": [
                _revision("S1", 1, "生效", ["1", "2", "3", "4"]),
                _revision("S1", 2, "撤回", None),
            ]
        },
        _rows(("S1", 3, "生效", ["1", "2", "3", "4"])),
    )
    assert [row.revision for row in plan.new_revisions] == [3]


def test_plan_conflict_in_one_row_rejects_whole_batch():
    with pytest.raises(RetestConflictError):
        plan_retests(
            {"S1": [_revision("S1", 3, "生效", ["1", "2", "3", "4"])]},
            _rows(
                ("S2", 1, "生效", ["1", "2", "3", "4"]),
                ("S1", 1, "生效", ["1", "2", "3", "4"]),
            ),
        )


# -- alignment / views ---------------------------------------------------


def test_align_retests_reorders_elements_and_rejects_unknown_sample():
    samples = parse_samples(SAMPLES_CSV.encode())
    parsed = parse_retests(
        b"id,rev,status,CaO,Al2O3,SiO2,Fe\nS1,1,\xe7\x94\x9f\xe6\x95\x88,1,2,3,4\n"
    )
    aligned = align_retests(parsed, samples, 7)
    assert aligned[0].values == (
        Fraction(4),
        Fraction(3),
        Fraction(2),
        Fraction(1),
    )
    with pytest.raises(Exception, match="不在当前成品输入"):
        align_retests(
            parse_retests(retest_csv("NOPE,1,撤回,,,,")), samples, 7
        )


def test_align_retests_rejects_element_mismatch():
    samples = parse_samples(SAMPLES_CSV.encode())
    parsed = parse_retests(
        b"id,rev,status,MgO,Al2O3,SiO2,Fe\nS1,1,\xe7\x94\x9f\xe6\x95\x88,1,2,3,4\n"
    )
    with pytest.raises(Exception, match="元素列"):
        align_retests(parsed, samples, 7)


def test_views_ignore_revisions_bound_to_another_version():
    samples = parse_samples(SAMPLES_CSV.encode())
    revisions = [
        _revision("S1", 5, "生效", ["9", "9", "9", "9"]),
    ]
    views = build_sample_views(samples, revisions, samples_version=2)
    assert views[0].effective_values is None
    assert views[0].latest_revision is None


def test_views_only_use_latest_revision_and_mark_withdrawal():
    samples = parse_samples(SAMPLES_CSV.encode())
    revisions = [
        _revision("S1", 1, "生效", ["1", "1", "1", "1"]),
        _revision("S1", 2, "撤回", None),
        _revision("S2", 1, "生效", ["60", "4", "2", "1"]),
    ]
    views = build_sample_views(samples, revisions, samples_version=1)
    by_id = {view.sample_id: view for view in views}
    assert by_id["S1"].is_withdrawn
    assert by_id["S1"].effective_values is None
    assert by_id["S1"].latest_revision == 2
    assert not by_id["S1"].changed
    # Same as original: no review needed.
    assert by_id["S2"].differences == (0, 0, 0, 0)
    assert not by_id["S2"].changed
    # Untouched sample.
    assert by_id["S3"].latest_revision is None


# -- persistence ---------------------------------------------------------


def test_store_apply_retests_and_history(tmp_path: Path):
    store = Store(str(tmp_path / "r.db"))
    payload = SamplesInput.from_payload(
        parse_samples(SAMPLES_CSV.encode()).to_payload()
    )
    version = store.save_input("samples", payload.to_payload(), "s.csv")
    rows = _rows(("S1", 1, "生效", ["1", "2", "3", "4"]))
    result = store.apply_retests(version, rows, "r.csv")
    assert result.applied == 1 and result.batch_id > 0
    history = store.retest_history(version)
    assert len(history) == 1
    assert history[0].values == (Fraction(1), Fraction(2), Fraction(3), Fraction(4))

    # Replay writes nothing but succeeds.
    replay = store.apply_retests(
        version, _rows(("S1", 1, "生效", ["1", "2", "3", "4"])), "r.csv"
    )
    assert replay.applied == 0 and replay.replayed == 1
    assert len(store.retest_history(version)) == 1


def test_store_rejects_batch_when_samples_version_replaced(tmp_path: Path):
    store = Store(str(tmp_path / "r.db"))
    samples = parse_samples(SAMPLES_CSV.encode())
    old_version = store.save_input("samples", samples.to_payload(), "s.csv")
    store.save_input("samples", samples.to_payload(), "new.csv")

    with pytest.raises(RetestStaleVersionError):
        store.apply_retests(
            old_version, _rows(("S1", 1, "生效", ["1", "2", "3", "4"])), "r.csv"
        )
    with sqlite3.connect(tmp_path / "r.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM retest_revisions").fetchone()[0]
    assert count == 0


def test_store_conflict_keeps_previous_revisions(tmp_path: Path):
    store = Store(str(tmp_path / "r.db"))
    samples = parse_samples(SAMPLES_CSV.encode())
    version = store.save_input("samples", samples.to_payload(), "s.csv")
    store.apply_retests(
        version, _rows(("S1", 1, "生效", ["1", "2", "3", "4"])), "ok.csv"
    )
    with pytest.raises(RetestConflictError):
        store.apply_retests(
            version,
            _rows(
                ("S2", 1, "生效", ["1", "2", "3", "4"]),
                ("S1", 1, "生效", ["9", "9", "9", "9"]),
            ),
            "bad.csv",
        )
    history = store.retest_history(version)
    assert [rev.sample_id for rev in history] == ["S1"]
    assert history[0].values[0] == 1


# -- web flow ------------------------------------------------------------


def test_retest_upload_requires_samples_input(client: TestClient):
    response = post_retests(client, retest_csv("S1,1,撤回,,,,"))
    assert response.status_code == 400
    assert "尚无成品输入" in response.text


def test_retest_upload_applies_and_shows_differences(client: TestClient):
    upload_inputs(client)
    compute(client)
    response = post_retests(
        client,
        retest_csv(
            "S1,1,生效,62.74,3.71,2.02,1.13",
            "S2,1,生效,60.10,4.00,2.00,1.00",
        ),
    )
    assert response.status_code == 303
    api = retest_api(client)
    assert api["samples_version"] == samples_version(client)
    assert api["report_on_same_version"] is True
    by_id = {row["sample_id"]: row for row in api["rows"]}
    assert fracs(by_id["S2"]["original"]) == [60, 4, 2, 1]
    assert fracs(by_id["S2"]["effective_values"]) == [Fraction("60.1"), 4, 2, 1]
    assert fracs(by_id["S2"]["differences"]) == [Fraction("0.1"), 0, 0, 0]
    assert by_id["S2"]["needs_review"] is True
    assert fracs(by_id["S1"]["differences"]) == [0, 0, 0, 0]
    assert by_id["S1"]["needs_review"] is False
    assert by_id["S3"]["effective_values"] is None

    html = client.get("/").text
    assert "需复核" in html
    assert "无有效复测值" not in html
    # The form carries the samples version it was rendered against.
    assert f'name="samples_version" value="{samples_version(client)}"' in html


def samples_version(client: TestClient) -> int:
    return client.app_state.store.current_inputs()["samples"].version_id


def test_retest_upload_replay_does_not_duplicate(client: TestClient):
    upload_inputs(client)
    content = retest_csv("S2,1,生效,60.1,4,2,1")
    post_retests(client, content)
    response = post_retests(client, content)
    assert response.status_code == 303
    assert "未重复生效" in unquote(response.headers["location"])
    store: Store = client.app_state.store
    assert len(store.retest_history(samples_version(client))) == 1


def test_retest_upload_withdrawal_marks_status(client: TestClient):
    upload_inputs(client)
    post_retests(client, retest_csv("S2,1,生效,60.1,4,2,1"))
    response = post_retests(client, retest_csv("S2,2,撤回,,,,"))
    assert response.status_code == 303
    row = {r["sample_id"]: r for r in retest_api(client)["rows"]}["S2"]
    assert row["latest_status"] == "撤回"
    assert row["effective_values"] is None
    assert row["needs_review"] is False
    html = client.get("/").text
    assert "已撤回" in html
    assert "无有效复测值" in html


def test_retest_conflict_rejects_batch_and_keeps_previous(client: TestClient):
    upload_inputs(client)
    post_retests(client, retest_csv("S1,1,生效,1,2,3,4"))
    response = post_retests(
        client,
        retest_csv(
            "S2,1,生效,1,2,3,4",
            "S1,1,生效,9,9,9,9",
        ),
    )
    assert response.status_code == 400
    assert "整批拒收" in response.text
    store: Store = client.app_state.store
    history = store.retest_history(samples_version(client))
    assert [rev.sample_id for rev in history] == ["S1"]
    assert history[0].values[0] == 1
    # S2's row from the rejected batch was not applied either.
    assert "S2" not in {rev.sample_id for rev in history}


def test_retest_late_old_revision_rejected(client: TestClient):
    upload_inputs(client)
    post_retests(client, retest_csv("S1,1,生效,1,2,3,4"))
    post_retests(client, retest_csv("S1,2,生效,1,2,3,5"))
    # A never-delivered revision 1 is impossible now that 1 exists, so use
    # revision 1 with new content: same number, different content.
    response = post_retests(client, retest_csv("S1,1,生效,1,2,3,9"))
    assert response.status_code == 400
    assert "内容不同" in response.text
    # Identical replay is accepted but creates nothing new.
    response = post_retests(client, retest_csv("S1,1,生效,1,2,3,4"))
    assert response.status_code == 303
    assert "未重复生效" in unquote(response.headers["location"])
    # A different sample's late revision 1 (never delivered) is accepted
    # because nothing higher exists for it.
    response = post_retests(client, retest_csv("S3,1,撤回,,,,"))
    assert response.status_code == 303


def test_retest_higher_revision_restores_withdrawal(client: TestClient):
    upload_inputs(client)
    post_retests(client, retest_csv("S2,1,生效,60.1,4,2,1"))
    post_retests(client, retest_csv("S2,2,撤回,,,,"))
    response = post_retests(client, retest_csv("S2,3,生效,60.2,4,2,1"))
    assert response.status_code == 303
    row = {r["sample_id"]: r for r in retest_api(client)["rows"]}["S2"]
    assert row["latest_revision"] == 3
    assert fracs(row["effective_values"]) == [Fraction("60.2"), 4, 2, 1]
    assert row["latest_status"] == "生效"


def test_retest_unknown_sample_rejects_whole_batch(client: TestClient):
    upload_inputs(client)
    response = post_retests(
        client, retest_csv("S1,1,生效,1,2,3,4", "GHOST,1,撤回,,,,")
    )
    assert response.status_code == 400
    assert "不在当前成品输入" in response.text
    assert client.app_state.store.retest_history(samples_version(client)) == []


def test_new_samples_version_does_not_inherit_retests(client: TestClient):
    upload_inputs(client)
    post_retests(client, retest_csv("S2,1,生效,61,4,2,1"))
    compute(client)
    assert retest_api(client)["rows"][1]["needs_review"] is True

    changed = SAMPLES_CSV.replace("S2,60,4,2,1", "S2,61,4.2,2.1,1.05")
    first_version = samples_version(client)
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s2.csv", changed.encode(), "text/csv")},
        follow_redirects=False,
    )
    api = retest_api(client)
    assert api["samples_version"] == first_version + 1
    assert api["report_version"] == first_version
    assert api["report_on_same_version"] is False
    assert all(row["effective_values"] is None for row in api["rows"])
    assert all(row["needs_review"] is False for row in api["rows"])
    html = client.get("/").text
    assert "不能" in html and "归因" in html

    # Retests may be applied against the new version independently.
    response = post_retests(client, retest_csv("S2,1,生效,61.5,4.2,2.1,1.05"))
    assert response.status_code == 303
    api = retest_api(client)
    assert fracs(api["rows"][1]["effective_values"]) == [Fraction("61.5"), Fraction("4.2"), Fraction("2.1"), Fraction("1.05")]


def test_retest_does_not_modify_original_samples_or_report(client: TestClient):
    upload_inputs(client)
    compute(client)
    report_before = client.get("/api/latest").json()["report"]
    sample_export_before = client.get("/inputs/samples/export.csv").content
    post_retests(client, retest_csv("S2,1,生效,60.5,4,2,1"))
    report_after = client.get("/api/latest").json()["report"]
    assert report_after == report_before
    assert client.get("/inputs/samples/export.csv").content == sample_export_before


def test_review_page_is_single_committed_state(client: TestClient):
    upload_inputs(client)
    compute(client)
    post_retests(client, retest_csv("S2,1,生效,60.5,4,2,1"))
    html = client.get("/").text
    # Original and effective values appear side by side in the same table.
    assert "60.5" in html
    assert "0.5" in html
    assert "需复核" in html
    assert "导出复测核对 CSV" in html


def test_retest_export(client: TestClient):
    upload_inputs(client)
    post_retests(client, retest_csv("S2,1,生效,60.5,4,2,1", "S1,1,撤回,,,,"))
    response = client.get("/retests/export.csv")
    assert response.status_code == 200
    lines = response.content.decode("utf-8-sig").strip().splitlines()
    assert lines[0].startswith("sample_id,samples_version,report_version,report_on_same_version,latest_revision")
    assert len(lines) == 4
    import csv as csv_module
    import io

    rows = {
        row["sample_id"]: row
        for row in csv_module.DictReader(io.StringIO("\n".join(lines)))
    }
    version = str(samples_version(client))
    assert rows["S2"]["samples_version"] == version
    assert rows["S2"]["latest_status"] == "生效"
    assert rows["S2"]["latest_revision"] == "1"
    # No report was published, so nothing may be attributed to the retest.
    assert rows["S2"]["changed"] == "yes"
    assert rows["S2"]["needs_review"] == "no"
    assert rows["S1"]["latest_status"] == "撤回"
    assert rows["S1"]["effective_Fe"] == ""


def test_retest_upload_rejects_during_samples_replacement(client: TestClient):
    """A samples upload racing the retest submit fails the whole batch."""

    upload_inputs(client)
    store: Store = client.app_state.store
    version_before = store.current_inputs()["samples"].version_id

    original_apply = store.apply_retests

    def apply_after_replacement(samples_version, rows, filename):
        # Replace samples just before the batch write, simulating the
        # upload landing during the request.
        samples = parse_samples(SAMPLES_CSV.encode())
        store.save_input("samples", samples.to_payload(), "raced.csv")
        return original_apply(samples_version, rows, filename)

    store.apply_retests = apply_after_replacement  # type: ignore[method-assign]
    response = post_retests(
        client,
        retest_csv("S2,1,生效,60.5,4,2,1"),
        version=version_before,
    )
    assert response.status_code == 400
    assert "整批拒绝" in response.text
    assert store.retest_history(version_before) == []
    assert store.current_inputs()["samples"].version_id == version_before + 1


def test_retest_upload_rejects_stale_page_version(client: TestClient):
    """The version shown on the form is part of the upload's binding.

    If the samples input was replaced after the page rendered, submitting
    the old form must reject the whole batch, even though the server-side
    "current version" at submit time would otherwise look fine.
    """

    upload_inputs(client)
    stale_version = samples_version(client)
    changed = SAMPLES_CSV.replace("S2,60,4,2,1", "S2,60.5,4,2,1")
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s2.csv", changed.encode(), "text/csv")},
        follow_redirects=False,
    )
    post_retests(client, retest_csv("S2,1,生效,61,4,2,1"))  # new version
    current_version = samples_version(client)
    response = post_retests(
        client,
        retest_csv("S2,1,生效,61,4,2,1"),
        version=stale_version,
    )
    assert response.status_code == 400
    assert "整批拒绝" in response.text
    assert "已被替换" in response.text
    # Nothing extra was recorded on either version.
    assert len(client.app_state.store.retest_history(stale_version)) == 0
    assert len(client.app_state.store.retest_history(current_version)) == 1


def test_retest_upload_accepts_matching_page_version(client: TestClient):
    upload_inputs(client)
    version = samples_version(client)
    response = post_retests(
        client, retest_csv("S2,1,生效,61,4,2,1"), version=version
    )
    assert response.status_code == 303
    assert len(client.app_state.store.retest_history(version)) == 1


def test_retest_persistence_error_keeps_previous(client: TestClient, monkeypatch):
    upload_inputs(client)
    post_retests(client, retest_csv("S1,1,生效,1,2,3,4"))
    store: Store = client.app_state.store

    def broken(*args, **kwargs):
        raise StoreError("磁盘已满（测试注入）")

    monkeypatch.setattr(store, "apply_retests", broken)
    response = post_retests(client, retest_csv("S2,1,生效,1,2,3,4"))
    assert response.status_code == 400
    assert "保持不变" in response.text
    # Previous revision is intact after the injected failure.
    assert len(store.retest_history(samples_version(client))) == 1
