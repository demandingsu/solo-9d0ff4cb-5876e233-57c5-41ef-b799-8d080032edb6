"""FastAPI application for ore-blend proportion reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import formatting
from .compute import ComputeError, compute_report
from .db import ReportRecord, ReviewState, Store, StoreError, VersionedInput
from .inputs import (
    MAX_UPLOAD_BYTES,
    BaselineInput,
    InputValidationError,
    SamplesInput,
    parse_baseline,
    parse_retests,
    parse_samples,
)
from .retests import (
    RetestConflictError,
    RetestError,
    RetestStaleVersionError,
    align_retests,
    build_sample_views,
    view_to_payload,
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_ROW_LIMIT = 500


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(title="矿石配料反推与异常定位")
    store = Store(database_path or os.environ.get("ORE_DB_PATH", "data/app.db"))
    app.state.store = store

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["decimal"] = formatting.decimal
    templates.env.filters["percent"] = formatting.percent
    templates.env.filters["signed"] = formatting.signed
    templates.env.filters["ratio"] = formatting.ratio_text
    templates.env.filters["exact"] = formatting.exact

    app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    # -- helpers ---------------------------------------------------------

    def redirect_notice(
        message: str, kind: str = "ok", anchor: str = ""
    ) -> RedirectResponse:
        query = urlencode({"notice": message, "notice_kind": kind})
        url = f"/?{query}"
        if anchor:
            url += f"#{anchor}"
        return RedirectResponse(url, status_code=303)

    def payload_to_inputs(
        snapshot,
    ) -> tuple[BaselineInput, SamplesInput]:
        return (
            BaselineInput.from_payload(snapshot.baseline.payload),
            SamplesInput.from_payload(snapshot.samples.payload),
        )

    def input_summary(value: VersionedInput | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if value.kind == "baseline":
            parsed = BaselineInput.from_payload(value.payload)
            return {
                "version": value.version_id,
                "filename": value.filename,
                "created_at": value.created_at,
                "elements": list(parsed.elements),
                "names": list(parsed.ores),
                "rows": [
                    [formatting.decimal(v, 8) for v in row] for row in parsed.rows
                ],
                "count": len(parsed.rows),
            }
        parsed_samples = SamplesInput.from_payload(value.payload)
        return {
            "version": value.version_id,
            "filename": value.filename,
            "created_at": value.created_at,
            "elements": list(parsed_samples.elements),
            "names": list(parsed_samples.sample_ids[:5]),
            "rows": [],
            "count": len(parsed_samples.sample_ids),
        }

    def report_view(record: ReportRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        payload = dict(record.payload)
        rows = payload.get("results", [])
        visible = rows[:REPORT_ROW_LIMIT]
        return {
            "id": record.report_id,
            "created_at": record.created_at,
            "baseline_version": record.baseline_version,
            "samples_version": record.samples_version,
            "stale": record.is_stale,
            "baseline_current": record.baseline_is_current,
            "samples_current": record.samples_is_current,
            "payload": payload,
            "rows": visible,
            "hidden_rows": max(0, len(rows) - len(visible)),
        }

    def retest_review(state: ReviewState) -> dict[str, Any] | None:
        """Per-sample original / effective retest / difference view.

        The review is always built from the *current* samples version, so
        retests bound to an older samples input can never appear here.  A
        published report can only be blamed on a retest when it refers to
        that same samples version; otherwise the page says the difference
        cannot be attributed to this correction.
        """

        samples_input = state.samples
        if samples_input is None:
            return None
        parsed = SamplesInput.from_payload(samples_input.payload)
        views = build_sample_views(
            parsed,
            state.revisions,
            samples_version=samples_input.version_id,
        )
        report = state.report
        report_on_version = (
            report is not None
            and report.samples_version == samples_input.version_id
        )
        rows = []
        for view in views:
            entry = {
                "sample_id": view.sample_id,
                "original": [
                    formatting.decimal(value, 8) for value in view.original
                ],
                "effective": (
                    None
                    if view.effective_values is None
                    else [
                        formatting.decimal(value, 8)
                        for value in view.effective_values
                    ]
                ),
                "differences": (
                    None
                    if view.differences is None
                    else [
                        formatting.signed(value, 8) for value in view.differences
                    ]
                ),
                "latest_revision": view.latest_revision,
                "latest_status": view.latest_status,
                "withdrawn": view.is_withdrawn,
                "correction": view.effective_values is not None,
                "changed": view.changed,
                # Only a report computed from this exact samples version
                # may have the change attributed to the retest.
                "needs_review": view.changed and report_on_version,
            }
            rows.append(entry)
        changed = [row for row in rows if row["changed"]]
        return {
            "samples_version": samples_input.version_id,
            "filename": samples_input.filename,
            "elements": list(parsed.elements),
            "rows": rows,
            "corrected_count": sum(1 for row in rows if row["correction"]),
            "withdrawn_count": sum(1 for row in rows if row["withdrawn"]),
            "changed_count": len(changed),
            "needs_review_count": sum(
                1 for row in rows if row["needs_review"]
            ),
            "report_version": report.samples_version if report else None,
            "report_on_same_version": report_on_version,
            "report_available": report is not None,
            "other_version_revisions": state.other_version_revisions,
        }

    def context(request: Request) -> dict[str, Any]:
        state = store.review_state()
        current = state.current
        return {
            "request": request,
            "current": {
                kind: input_summary(value) for kind, value in current.items()
            },
            "report": report_view(state.report),
            "retests": retest_review(state),
            "notice": request.query_params.get("notice"),
            "notice_kind": request.query_params.get("notice_kind", "ok"),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", context(request))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inputs/upload")
    async def upload_inputs(
        request: Request,
        baseline_file: UploadFile | None = File(None),
        samples_file: UploadFile | None = File(None),
    ) -> Response:
        provided = {
            "baseline": baseline_file,
            "samples": samples_file,
        }
        if not any(file is not None and file.filename for file in provided.values()):
            return redirect_notice("请至少选择一个 CSV 文件再上传。", "error", "upload")

        parsed: dict[str, Any] = {}
        filenames: dict[str, str] = {}
        labels = {"baseline": "基准", "samples": "成品"}
        try:
            for kind, file in provided.items():
                if file is None or not file.filename:
                    continue
                raw = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise _LabeledError(
                        labels[kind],
                        f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                parser = parse_baseline if kind == "baseline" else parse_samples
                try:
                    parsed[kind] = await run_in_threadpool(parser, raw)
                except InputValidationError as exc:
                    raise _LabeledError(labels[kind], str(exc)) from exc
                filenames[kind] = (
                    file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
                    or "未命名.csv"
                )
        except _LabeledError as exc:
            return _render_with_error(request, f"{exc.label}文件：{exc.message}")

        updates: dict[str, tuple[dict[str, Any], str]] = {}
        for kind, value in parsed.items():
            updates[kind] = (value.to_payload(), filenames[kind])

        had_report = await run_in_threadpool(
            lambda: store.latest_report() is not None
        )
        try:
            versions = await run_in_threadpool(store.save_inputs, updates)
        except StoreError as exc:
            return _render_with_error(
                request, f"输入写入失败，原有输入未被替换：{exc}"
            )

        replaced = []
        if "baseline" in versions:
            replaced.append(f"基准 v{versions['baseline']}")
        if "samples" in versions:
            replaced.append(f"成品 v{versions['samples']}")
        suffix = "。旧报告已标记为过期。" if had_report else "。"
        return redirect_notice(
            "已整版替换：" + "，".join(replaced) + suffix,
            "ok",
            "report",
        )

    def _render_with_error(request: Request, message: str) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    @app.post("/retests/upload")
    async def upload_retests(
        request: Request,
        retest_file: UploadFile | None = File(None),
        samples_version: str | None = Form(None),
    ) -> Response:
        if retest_file is None or not retest_file.filename:
            return redirect_notice(
                "请选择复测订正 CSV 再上传。", "error", "retests"
            )

        raw = await retest_file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return _render_retest_error(
                request,
                f"复测文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限，"
                "整批未接收。",
            )
        try:
            parsed = await run_in_threadpool(parse_retests, raw)
        except InputValidationError as exc:
            return _render_retest_error(
                request, f"复测文件：{exc}。整批未接收。"
            )

        # Bind the batch to the samples version shown when the form was
        # rendered (submitted with the file).  A version mismatch means the
        # samples input was replaced while the clerk was preparing the
        # upload, so the correction must fail as a whole rather than being
        # silently applied to a different set of samples.  apply_retests
        # re-checks this binding inside its write transaction too.
        state = await run_in_threadpool(store.review_state)
        samples_value = state.samples
        if samples_value is None:
            return _render_retest_error(
                request, "尚无成品输入，复测无法绑定版本。整批未接收。"
            )
        if samples_version is not None and samples_version.strip():
            if not samples_version.strip().isdigit():
                return _render_retest_error(
                    request, "复测提交的成品版本号无效。整批未接收。"
                )
            if int(samples_version.strip()) != samples_value.version_id:
                return _render_retest_error(
                    request,
                    f"复测提交绑定成品 v{samples_version.strip()}，"
                    f"但当前成品已是 v{samples_value.version_id}"
                    "（上传期间成品输入已被替换）。本次订正整批拒绝，"
                    "此前有效的复测记录保持不变；请刷新页面后重新提交。",
                )
        samples = SamplesInput.from_payload(samples_value.payload)
        try:
            rows = await run_in_threadpool(
                align_retests, parsed, samples, samples_value.version_id
            )
        except RetestError as exc:
            return _render_retest_error(
                request, f"{exc}。整批未接收。"
            )

        filename = (
            retest_file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
            or "未命名.csv"
        )
        try:
            result = await run_in_threadpool(
                store.apply_retests, samples_value.version_id, rows, filename
            )
        except RetestConflictError as exc:
            return _render_retest_error(
                request, f"{exc}。整批拒收，此前有效的复测记录保持不变。"
            )
        except RetestStaleVersionError as exc:
            return _render_retest_error(request, str(exc))
        except StoreError as exc:
            return _render_retest_error(
                request,
                f"复测写入失败，整批未生效：{exc}。此前有效的复测记录保持不变。",
            )

        if result.applied == 0:
            message = (
                f"本次 {result.replayed} 行与已有修订内容完全一致，"
                "均为重复提交，未重复生效。"
            )
        else:
            message = (
                f"复测订正已生效：{result.applied} 行新修订"
                + (
                    f"，{result.replayed} 行重复提交未重复生效"
                    if result.replayed
                    else ""
                )
                + f"；绑定成品 v{result.samples_version}。"
            )
        return redirect_notice(message, "ok", "retests")

    def _render_retest_error(request: Request, message: str) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    @app.post("/compute")
    async def compute(request: Request) -> Response:
        snapshot = await run_in_threadpool(store.snapshot)
        if snapshot is None:
            return redirect_notice(
                "基准与成品输入都必须先上传，才能计算。", "error", "upload"
            )

        try:
            baseline, samples = payload_to_inputs(snapshot)
            payload = await run_in_threadpool(
                compute_report, baseline, samples
            )
        except (ComputeError, InputValidationError) as exc:
            return redirect_notice(str(exc), "error", "report")
        except Exception as exc:  # pragma: no cover - defensive
            return redirect_notice(
                f"计算失败，未产生新报告：{type(exc).__name__}: {exc}",
                "error",
                "report",
            )

        try:
            report_id = await run_in_threadpool(
                store.publish_report, snapshot, payload
            )
        except StoreError as exc:
            return redirect_notice(
                f"报告写入失败，仍显示上一份报告：{exc}", "error", "report"
            )

        if report_id is None:
            return redirect_notice(
                "计算期间输入已被更新，本次结果与当前输入不符，已丢弃，"
                "未发布报告。请基于新输入重新计算。",
                "error",
                "report",
            )

        return redirect_notice(
            f"报告 #{report_id} 已生成并发布（{payload['sample_count']} 个样本，"
            f"{payload['non_unique_count']} 个无法唯一确定）。",
            "ok",
            "report",
        )

    # -- exports ---------------------------------------------------------

    @app.get("/reports/{report_id}/export.csv")
    def export_report(report_id: int) -> Response:
        record = store.get_report(report_id)
        if record is None:
            return PlainTextResponse("报告不存在", status_code=404)
        payload = record.payload
        elements = payload["elements"]
        header = ["sample_id", "unique"]
        for ore in payload["ores"]:
            header.append(f"{ore}_weight")
        for ore in payload["ores"]:
            header.append(f"{ore}_weight_min")
            header.append(f"{ore}_weight_max")
        for element in elements:
            header.append(f"measured_{element}")
        for element in elements:
            header += [
                f"predicted_{element}",
                f"predicted_{element}_min",
                f"predicted_{element}_max",
            ]
        for element in elements:
            header += [
                f"residual_{element}",
                f"residual_{element}_min",
                f"residual_{element}_max",
            ]
        header += ["sse", "note"]

        def triple(point_value, interval):
            """One element's (value, min, max) cells.

            A non-unique optimum has no single value, so the value cell is
            left empty and only the exact attainable range is emitted.
            """

            low, high = interval
            return [
                "" if point_value is None else formatting.decimal(point_value, 10),
                formatting.decimal(low, 10),
                formatting.decimal(high, 10),
            ]

        rows = [",".join(_csv_cell(cell) for cell in header)]
        for result in payload["results"]:
            row = [result["sample_id"], "yes" if result["unique"] else "no"]
            if result["unique"]:
                row += [formatting.percent(v, 8) for v in result["weights"]]
            else:
                row += [""] * len(payload["ores"])
            for low, high in result["weight_ranges"]:
                row += [formatting.percent(low, 8), formatting.percent(high, 8)]
            row += [formatting.decimal(v, 10) for v in result["measured"]]
            for i in range(len(elements)):
                row += triple(
                    None if result["predicted"] is None else result["predicted"][i],
                    result["predicted_ranges"][i],
                )
            for i in range(len(elements)):
                row += triple(
                    None if result["residuals"] is None else result["residuals"][i],
                    result["residual_ranges"][i],
                )
            row.append(formatting.decimal(result["sse"], 12))
            row.append(
                ""
                if result["unique"]
                else "最优占比不唯一：占比、预测值与残差仅以下列 min/max 范围给出，"
                "不存在唯一的配料组合"
            )
            rows.append(",".join(_csv_cell(cell) for cell in row))
        return Response(
            "﻿" + "\r\n".join(rows) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ore-report-{report_id}.csv"'
                )
            },
        )

    @app.get("/inputs/{kind}/export.csv")
    def export_input(kind: str) -> Response:
        if kind not in ("baseline", "samples"):
            return PlainTextResponse("输入类型不存在", status_code=404)
        current = store.current_inputs().get(kind)
        if current is None:
            return PlainTextResponse("输入尚未上传", status_code=404)
        if kind == "baseline":
            value = BaselineInput.from_payload(current.payload)
            header = ["ore", *value.elements]
            rows = [
                [ore, *[formatting.decimal(x, 12) for x in row]]
                for ore, row in zip(value.ores, value.rows)
            ]
        else:
            value = SamplesInput.from_payload(current.payload)
            header = ["sample_id", *value.elements]
            rows = [
                [sample_id, *[formatting.decimal(x, 12) for x in row]]
                for sample_id, row in zip(value.sample_ids, value.rows)
            ]
        lines = [",".join(_csv_cell(cell) for cell in header)]
        lines += [",".join(_csv_cell(cell) for cell in row) for row in rows]
        return Response(
            "﻿" + "\r\n".join(lines) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}.csv"'
            },
        )

    @app.get("/api/latest")
    def api_latest() -> JSONResponse:
        state = store.review_state()
        record = state.report
        current = state.current
        retests: dict[str, Any] | None = None
        if state.samples is not None:
            samples = SamplesInput.from_payload(state.samples.payload)
            views = build_sample_views(
                samples,
                state.revisions,
                samples_version=state.samples.version_id,
            )
            report_on_same_version = bool(
                record
                and record.samples_version == state.samples.version_id
            )
            retests = {
                "samples_version": state.samples.version_id,
                "report_version": record.samples_version if record else None,
                "report_on_same_version": report_on_same_version,
                "rows": [
                    view_to_payload(
                        view,
                        state.samples.version_id,
                        report_on_same_version=report_on_same_version,
                    )
                    for view in views
                ],
            }
        if record is None:
            return JSONResponse(
                {
                    "report": None,
                    "retests": retests,
                    "inputs": {
                        kind: (
                            {
                                "version": value.version_id,
                                "filename": value.filename,
                                "created_at": value.created_at,
                            }
                            if value
                            else None
                        )
                        for kind, value in current.items()
                    },
                }
            )
        return JSONResponse(
            {
                "report": {
                    "id": record.report_id,
                    "created_at": record.created_at,
                    "baseline_version": record.baseline_version,
                    "samples_version": record.samples_version,
                    "stale": record.is_stale,
                    "payload": record.payload,
                },
                "retests": retests,
            }
        )

    @app.get("/retests/export.csv")
    def export_retests() -> Response:
        state = store.review_state()
        if state.samples is None:
            return PlainTextResponse("成品输入尚未上传", status_code=404)
        samples = SamplesInput.from_payload(state.samples.payload)
        views = build_sample_views(
            samples,
            state.revisions,
            samples_version=state.samples.version_id,
        )
        header = [
            "sample_id",
            "samples_version",
            "report_version",
            "report_on_same_version",
            "latest_revision",
            "latest_status",
            "changed",
            "needs_review",
            *[f"original_{e}" for e in samples.elements],
            *[f"effective_{e}" for e in samples.elements],
            *[f"difference_{e}" for e in samples.elements],
        ]

        def cells(values):
            if values is None:
                return [""] * len(samples.elements)
            return [formatting.decimal(value, 12) for value in values]

        report = state.report
        report_on_same_version = bool(
            report is not None
            and report.samples_version == state.samples.version_id
        )
        lines = [",".join(_csv_cell(cell) for cell in header)]
        for view in views:
            row = [
                view.sample_id,
                str(state.samples.version_id),
                "" if report is None else str(report.samples_version),
                "yes" if report_on_same_version else "no",
                "" if view.latest_revision is None else str(view.latest_revision),
                view.latest_status or "",
                "yes" if view.changed else "no",
                "yes"
                if view.changed and report_on_same_version
                else "no",
                *cells(view.original),
                *cells(view.effective_values),
                *cells(view.differences),
            ]
            lines.append(",".join(_csv_cell(cell) for cell in row))
        return Response(
            "﻿" + "\r\n".join(lines) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="retests-v{state.samples.version_id}.csv"'
                )
            },
        )

    return app


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\r", "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


class _LabeledError(Exception):
    def __init__(self, label: str, message: str) -> None:
        super().__init__(message)
        self.label = label
        self.message = message


app = create_app()
