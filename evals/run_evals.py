#!/usr/bin/env python3
"""Eval harness — scores trajectories as well as answers.

    python evals/run_evals.py                  # every case, in its declared modes
    python evals/run_evals.py --mode cag       # restrict to one mode
    python evals/run_evals.py --case AC3       # one case
    python evals/run_evals.py --no-judge       # skip the LLM rubric (free, faster)

Each case runs a real conversation through the real agent loop. Assertions read
two sources: the JSONL trace for what the agent *did*, and the final answer for
what it *said*. Cases carrying a `judge` rubric get a second Claude call with a
constrained output schema; where trajectory assertions are decisive they stay
authoritative and the judge is advisory.

Every run costs real API tokens. AGENT_TODAY is pinned from the case file so
fixture offsets resolve identically on every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import yaml  # noqa: E402
from rich.console import Console  # noqa: E402

from agent import config  # noqa: E402
from agent.context import build_provider  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from agent.memory import Conversation  # noqa: E402
from agent.render import Renderer  # noqa: E402
from agent.trace import Tracer  # noqa: E402
from agent.tools.registry import build_dispatcher  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TRACE_DIR = EVAL_DIR / "traces"
REPORT_PATH = EVAL_DIR / "report.md"

JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["pass", "reason"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------
# Running one case
# --------------------------------------------------------------------------


def run_case(case: dict, mode: config.Mode, console: Console) -> dict:
    """Execute one case in one mode and return its trace plus final answer."""
    session_id = f"eval-{case['id']}-{mode.value}"
    tracer = Tracer(session_id, TRACE_DIR, enabled=True)

    # Fresh trace per run — otherwise assertions read yesterday's trajectory.
    if tracer.path.exists():
        tracer.path.unlink()

    # Silence the renderer: the harness prints its own progress.
    renderer = Renderer(console=Console(quiet=True, file=open(os.devnull, "w")))
    dispatch, _ = build_dispatcher(mode, tracer)

    loop = AgentLoop(
        client=anthropic.Anthropic(),
        provider=build_provider(mode),
        dispatch=dispatch,
        renderer=renderer,
        tracer=tracer,
        conversation=Conversation(),
    )

    started = time.monotonic()
    answers = [loop.run_turn(turn) for turn in case["turns"]]
    elapsed = time.monotonic() - started

    records = [
        json.loads(line)
        for line in tracer.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "answers": answers,
        "final": answers[-1],
        "records": records,
        "elapsed_s": round(elapsed, 2),
        "trace_path": tracer.path,
    }


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------


def check_case(case: dict, mode: config.Mode, run: dict) -> list[dict]:
    """Return one result row per assertion."""
    checks: list[dict] = []
    records = run["records"]
    final = run["final"]

    calls = [r for r in records if r["event"] == "tool_call"]
    called = [r["tool"] for r in calls]

    def add(kind: str, ok: bool, detail: str) -> None:
        checks.append({"kind": kind, "ok": ok, "detail": detail})

    # -- trajectory --------------------------------------------------------

    for tool in case.get("expects_tools", []):
        # CAG has no retrieve_kb by design; the KB is in the cached prefix.
        if tool == "retrieve_kb" and mode is config.Mode.CAG:
            add("expects_tool", True, f"{tool} — CAG 模式不適用，略過")
            continue
        add("expects_tool", tool in called, f"應呼叫 {tool}")

    for tool in case.get("forbids_tools", []):
        add("forbids_tool", tool not in called, f"不應呼叫 {tool}")

    expected_order = [
        t for t in case.get("tool_order", [])
        if not (t == "retrieve_kb" and mode is config.Mode.CAG)
    ]
    if expected_order:
        seen = [t for t in called if t in expected_order]
        # Subsequence, not equality — a retry or an extra retrieval between
        # links is fine; the dependency order is what matters.
        it = iter(seen)
        ok = all(t in it for t in expected_order)
        add("tool_order", ok, f"順序 {' → '.join(expected_order)}（實際 {' → '.join(seen) or '無'}）")

    for spec in case.get("expects_tool_input", []):
        matches = [
            c for c in calls
            if c["tool"] == spec["tool"] and spec["contains"] in json.dumps(c["input"], ensure_ascii=False)
        ]
        add("tool_input", bool(matches), f"{spec['tool']} 參數含 {spec['contains']}")

    # -- content -----------------------------------------------------------

    # "a|b" means either spelling satisfies the assertion — the model may write
    # 1,767 / 1767 / NT$1,767 and all three are correct.
    for needle in case.get("must_contain", []):
        variants = needle.split("|")
        add("must_contain", any(v in final for v in variants),
            f"答案含「{' 或 '.join(variants)}」")

    for needle in case.get("must_not_contain", []):
        variants = needle.split("|")
        add("must_not_contain", not any(v in final for v in variants),
            f"答案不含「{' 或 '.join(variants)}」")

    return checks


def judge(case: dict, run: dict, client: anthropic.Anthropic) -> dict | None:
    """LLM rubric for cases where string matching is too brittle."""
    rubric = case.get("judge")
    if not rubric:
        return None

    transcript = "\n\n".join(
        f"顧客：{q}\n助理：{a}" for q, a in zip(case["turns"], run["answers"])
    )

    # Several rubrics ask whether a tool was actually called. The judge cannot
    # see that from the transcript alone — without this it reports an absence it
    # has no way to observe, and fails correct runs. The trace is the authority.
    calls = [r["tool"] for r in run.get("records", []) if r["event"] == "tool_call"]
    trajectory = (
        "本次對話實際呼叫的工具（依序，來自系統追蹤記錄，為權威來源）："
        + ("、".join(calls) if calls else "（未呼叫任何工具）")
    )
    response = client.messages.create(
        model=config.MODEL,
        max_tokens=1000,
        output_config={"format": JUDGE_SCHEMA, "effort": "low"},
        messages=[
            {
                "role": "user",
                "content": (
                    "你是一位嚴格的評測員，正在檢查電商客服 AI 的回覆是否符合規範。\n\n"
                    f"# 評分標準\n{rubric}\n\n"
                    f"# 對話紀錄\n{transcript}\n\n"
                    f"# 工具呼叫記錄\n{trajectory}\n\n"
                    "請依標準判斷是否通過，並用繁體中文簡短說明理由。"
                ),
            }
        ],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(rows: list[dict], consistency: list[dict]) -> None:
    lines = [
        "# Eval results",
        "",
        f"執行時間：{time.strftime('%Y-%m-%d %H:%M:%S')}　"
        f"模型：`{config.MODEL}`　effort：`{config.EFFORT}`",
        "",
        "軌跡斷言讀自 `evals/traces/*.jsonl`，內容斷言讀自最終答案，"
        "`judge` 欄為 LLM 評分（僅用於難以字串比對的案例）。",
        "",
        "| Case | Mode | 軌跡 | 內容 | Judge | 步數 | 秒 | 結果 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        traj = [c for c in row["checks"] if c["kind"] in
                ("expects_tool", "forbids_tool", "tool_order", "tool_input")]
        cont = [c for c in row["checks"] if c["kind"] in ("must_contain", "must_not_contain")]

        def tally(items):
            if not items:
                return "–"
            passed = sum(1 for c in items if c["ok"])
            return f"{passed}/{len(items)}"

        verdict = row["judge"]
        judge_cell = "–" if verdict is None else ("✅" if verdict["pass"] else "❌")
        lines.append(
            f"| {row['id']} | {row['mode']} | {tally(traj)} | {tally(cont)} | {judge_cell} | "
            f"{row['steps']} | {row['elapsed_s']} | {'✅ PASS' if row['ok'] else '❌ FAIL'} |"
        )

    passed = sum(1 for r in rows if r["ok"])
    lines += ["", f"**{passed}/{len(rows)} 通過**", ""]

    if consistency:
        lines += ["## AC7 — 跨模式一致性", ""]
        for entry in consistency:
            mark = "✅" if entry["ok"] else "❌"
            lines.append(f"- {mark} **{entry['id']}** — {entry['reason']}")
        lines.append("")

    # Failure detail, so the report is actionable rather than just a scoreboard.
    failures = [r for r in rows if not r["ok"]]
    if failures:
        lines += ["## 失敗細節", ""]
        for row in failures:
            lines.append(f"### {row['id']} · {row['mode']}")
            for check in row["checks"]:
                if not check["ok"]:
                    lines.append(f"- ❌ {check['detail']}")
            if row["judge"] and not row["judge"]["pass"]:
                lines.append(f"- ❌ judge：{row['judge']['reason']}")
            lines += ["", "```", row["final"][:600], "```", ""]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="執行 acceptance criteria 評測")
    parser.add_argument("--mode", choices=[m.value for m in config.Mode], default=None)
    parser.add_argument("--case", default=None, help="只跑單一 case（例如 AC3）")
    parser.add_argument("--no-judge", action="store_true", help="跳過 LLM 評分")
    args = parser.parse_args(argv)

    console = Console()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]找不到 ANTHROPIC_API_KEY。[/red]")
        return 1

    spec = yaml.safe_load((EVAL_DIR / "cases.yaml").read_text(encoding="utf-8"))
    defaults = spec.get("defaults", {})

    # Pin the clock so fixture offsets resolve identically every run (FR10).
    if defaults.get("pinned_date"):
        os.environ["AGENT_TODAY"] = defaults["pinned_date"]

    cases = spec["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            console.print(f"[red]找不到 case {args.case}[/red]")
            return 1

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    rows: list[dict] = []
    by_case: dict[str, dict[str, str]] = {}

    for case in cases:
        modes = [config.Mode(m) for m in case.get("modes", defaults.get("modes", ["rag"]))]
        if args.mode:
            modes = [m for m in modes if m.value == args.mode]

        for mode in modes:
            console.print(f"[dim]執行 {case['id']} · {mode.value} …[/dim]")
            try:
                run = run_case(case, mode, console)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]  {case['id']} · {mode.value} 執行失敗：{exc}[/red]")
                rows.append({
                    "id": case["id"], "mode": mode.value, "checks": [
                        {"kind": "run", "ok": False, "detail": f"執行失敗：{exc}"}
                    ], "judge": None, "ok": False, "steps": 0, "elapsed_s": 0,
                    "final": "",
                })
                continue

            checks = check_case(case, mode, run)
            verdict = None if args.no_judge else judge(case, run, client)

            ok = all(c["ok"] for c in checks) and (verdict is None or verdict["pass"])
            steps = max((r.get("step", 0) for r in run["records"] if "step" in r), default=0)

            rows.append({
                "id": case["id"], "mode": mode.value, "checks": checks,
                "judge": verdict, "ok": ok, "steps": steps,
                "elapsed_s": run["elapsed_s"], "final": run["final"],
            })
            by_case.setdefault(case["id"], {})[mode.value] = run["final"]

            mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            console.print(f"  {mark}  {case['id']} · {mode.value}  ({run['elapsed_s']}s, {steps} 步)")

    # AC7 — do the two modes actually agree?
    consistency: list[dict] = []
    for case in cases:
        if not case.get("cross_mode_consistency"):
            continue
        answers = by_case.get(case["id"], {})
        if len(answers) < 2:
            continue
        verdict = judge(
            {
                "turns": ["（跨模式一致性檢查）"],
                "judge": (
                    "以下是同一個問題在兩種檢索模式下的回覆。"
                    "通過條件：兩者在實質結論上一致（可以退／不可以退、條件、金額相同），"
                    "用字不同不算失敗。失敗條件：結論相反或關鍵條件不同。"
                ),
            },
            {"answers": [f"[RAG]\n{answers['rag']}\n\n[CAG]\n{answers['cag']}"]},
            client,
        ) if not args.no_judge else None

        consistency.append({
            "id": case["id"],
            "ok": verdict["pass"] if verdict else True,
            "reason": verdict["reason"] if verdict else "（--no-judge，未評分）",
        })

    write_report(rows, consistency)
    passed = sum(1 for r in rows if r["ok"])
    console.print(f"\n[bold]{passed}/{len(rows)} 通過[/bold] → {REPORT_PATH}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
