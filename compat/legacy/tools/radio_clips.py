"""Radio highlight clips: score candidates with an LLM, then cut the ones you pick.

  plan: python tools/radio_clips.py plan --srt EP.srt --media EP.mp3 --out DIR --label 45
        -> DIR/45.candidates.md (scored list to pick from) + DIR/45.plan.json
  refine: python tools/radio_clips.py refine --plan DIR/45.plan.json   (redo the boundary check only)
  cut:  python tools/radio_clips.py cut --plan DIR/45.plan.json --pick 1,3,5 [--with-srt]
        -> DIR/clips/#45-01 标题.mp3 (lossless stream copy) [+ subtitle sliced from the full episode]

Cut points: 1.0s before the first cue, 1.0s after the last cue (measured on hand-cut LosslessCut clips).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.llm_client import call_llm  # noqa: E402
from qwen_asr import clip_planner as cp  # noqa: E402
from qwen_asr.credentials import resolve_llm_api_key, resolve_llm_base_url  # noqa: E402
from qwen_asr.defaults import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL  # noqa: E402

DEFAULT_PROMPT = ROOT / "tools" / "prompts" / "radio_clip_selection.md"
DEFAULT_REFINE_PROMPT = ROOT / "tools" / "prompts" / "radio_clip_boundary.md"


def cmd_plan(args: argparse.Namespace) -> int:
    srt, media, out = Path(args.srt), Path(args.media), Path(args.out)
    cues = cp.parse_srt(srt.read_text(encoding="utf-8-sig"))
    if not cues:
        raise SystemExit(f"no cues in {srt}")
    title = args.title or media.stem.replace("【源】", "")
    label = args.label or title
    out.mkdir(parents=True, exist_ok=True)
    messages = cp.build_messages(cues, prompt=Path(args.prompt).read_text(encoding="utf-8"), title=title)
    api_key = resolve_llm_api_key(args.api_key, args.base_url)
    args.base_url = resolve_llm_base_url(args.base_url, api_key)
    if not api_key:
        raise SystemExit("missing API key (MIMO_API_KEY / LLM_API_KEY)")
    extra_body = None if args.thinking else {"thinking": {"type": "disabled"}}
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            response = call_llm(messages, model=args.model, temperature=args.temperature, base_url=args.base_url,
                                api_key=api_key, llm_extra_body=extra_body, timeout=args.timeout,
                                max_tokens=args.max_tokens)
            text = response.choices[0].message.content or ""
            (out / f"{label}.response.txt").write_text(text, encoding="utf-8")
            plan = cp.build_plan(cp.extract_json_object(text), cues, pre_roll=args.pre_roll,
                                 post_roll=args.post_roll, media_duration=cp.media_duration_seconds(media))
            break
        except Exception as exc:  # noqa: BLE001 - retry any transport/format failure, then give up
            last_error = exc
            print(f"attempt {attempt} failed: {exc}", file=sys.stderr)
    else:
        raise SystemExit(f"plan failed: {last_error}")
    usage = getattr(response, "usage", None)
    plan.update(title=title, label=label, srt=str(srt.resolve()), media=str(media.resolve()), model=args.model,
                seconds=round(time.perf_counter() - started, 1),
                usage={"prompt_tokens": getattr(usage, "prompt_tokens", None),
                       "completion_tokens": getattr(usage, "completion_tokens", None)})
    if args.refine:
        plan = _refine(plan, cues, out, args, api_key)
    _write_plan(plan, cues, out)
    print(f"{label}: {len(plan['candidates'])} candidates in {plan['seconds']}s -> {out / f'{label}.candidates.md'}")
    return 0


def _write_plan(plan: dict, cues: list[cp.Cue], out: Path) -> None:
    label = plan["label"]
    (out / f"{label}.plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{label}.candidates.md").write_text(cp.render_markdown(plan, cues, title=plan["title"]), encoding="utf-8")


def _refine(plan: dict, cues: list[cp.Cue], out: Path, args: argparse.Namespace, api_key: str) -> dict:
    """Second pass: let the model check each candidate's boundaries; keep the plan unchanged on failure."""
    messages = cp.build_refine_messages(cues, cp.candidates_from_plan(plan),
                                        prompt=Path(args.refine_prompt).read_text(encoding="utf-8"))
    extra_body = None if args.thinking else {"thinking": {"type": "disabled"}}
    plan = {**plan, "warnings": [w for w in plan.get("warnings", []) if not w.startswith("boundary refine")]}
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 3):
        try:
            response = call_llm(messages, model=args.model, temperature=0.1, base_url=args.base_url,
                                api_key=api_key, llm_extra_body=extra_body, timeout=args.timeout, max_tokens=4000)
            text = response.choices[0].message.content or ""
            (out / f"{plan['label']}.refine.txt").write_text(text, encoding="utf-8")
            return cp.apply_refinement(plan, cp.extract_json_object(text), cues,
                                       media_duration=cp.media_duration_seconds(Path(plan["media"])))
        except Exception as exc:  # noqa: BLE001 - retry, then keep the unrefined plan
            last_error = exc
            print(f"boundary refine attempt {attempt} failed: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
    plan["warnings"].append(f"boundary refine skipped: {last_error}")
    return plan


def cmd_refine(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cues = cp.parse_srt(Path(plan["srt"]).read_text(encoding="utf-8-sig"))
    api_key = resolve_llm_api_key(args.api_key, args.base_url)
    args.base_url = resolve_llm_base_url(args.base_url, api_key)
    if not api_key:
        raise SystemExit("missing API key (MIMO_API_KEY / LLM_API_KEY)")
    plan = _refine(plan, cues, plan_path.parent, args, api_key)
    _write_plan(plan, cues, plan_path.parent)
    print(f"{plan['label']}: {len(plan.get('boundary_changes', []))} boundary changes, "
          f"warnings={[w for w in plan['warnings'] if w.startswith('boundary refine')]}")
    return 0


def _parse_picks(value: str) -> list[int]:
    return [int(part) for part in value.replace("，", ",").split(",") if part.strip()]


def cmd_cut(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    media = Path(args.media or plan["media"])
    cues = cp.parse_srt(Path(plan["srt"]).read_text(encoding="utf-8-sig"))
    plan = cp.recompute_times(plan, cues, media_duration=cp.media_duration_seconds(media))
    by_number = {c.number: c for c in cp.candidates_from_plan(plan)}
    picks = sorted(by_number) if args.pick == "all" else _parse_picks(args.pick)
    missing = [n for n in picks if n not in by_number]
    if missing:
        raise SystemExit(f"unknown candidate numbers: {missing}")
    out = Path(args.out) if args.out else plan_path.parent / "clips"
    for spec in args.adjust or []:
        number, start_cue, end_cue = cp.parse_adjust(spec)
        if number not in by_number:
            raise SystemExit(f"--adjust: unknown candidate {number}")
        by_number[number] = cp.adjust_candidate(by_number[number], cues, start_cue, end_cue,
                                                pre_roll=plan.get("pre_roll_s", cp.DEFAULT_PRE_ROLL_S),
                                                post_roll=plan.get("post_roll_s", cp.DEFAULT_POST_ROLL_S),
                                                media_duration=cp.media_duration_seconds(media))
    for number in picks:
        candidate = by_number[number]
        name = cp.clip_basename(str(plan["label"]), candidate)
        target = out / f"{name}.mp3"
        if target.exists() and not args.overwrite:
            print(f"skip existing {target}")
            continue
        cp.cut_clip(media, target, candidate.start, candidate.end)
        if args.with_srt:
            sliced = cp.slice_cues(cues, candidate.start, candidate.end)
            (out / f"【ai】{name}.srt").write_text(cp.render_srt(sliced), encoding="utf-8")
        print(f"#{number} {cp.format_clock(candidate.start)}-{cp.format_clock(candidate.end)} -> {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    llm = argparse.ArgumentParser(add_help=False)
    llm.add_argument("--model", default=DEFAULT_LLM_MODEL)
    llm.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    llm.add_argument("--api-key")
    llm.add_argument("--timeout", type=float, default=600)
    llm.add_argument("--retries", type=int, default=1)
    llm.add_argument("--thinking", action="store_true", help="keep the model's thinking mode on")
    llm.add_argument("--refine-prompt", default=str(DEFAULT_REFINE_PROMPT))
    plan = sub.add_parser("plan", parents=[llm], help="score clip candidates with an LLM")
    plan.add_argument("--srt", required=True)
    plan.add_argument("--media", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--title")
    plan.add_argument("--label", help="short id used in file names, e.g. the episode number")
    plan.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    plan.add_argument("--temperature", type=float, default=0.3)
    plan.add_argument("--max-tokens", type=int, default=16000)
    plan.add_argument("--no-refine", dest="refine", action="store_false",
                      help="skip the second LLM pass that checks clip boundaries")
    plan.add_argument("--pre-roll", type=float, default=cp.DEFAULT_PRE_ROLL_S)
    plan.add_argument("--post-roll", type=float, default=cp.DEFAULT_POST_ROLL_S)
    plan.set_defaults(func=cmd_plan)
    refine = sub.add_parser("refine", parents=[llm], help="(re)run the boundary check on an existing plan")
    refine.add_argument("--plan", required=True)
    refine.set_defaults(func=cmd_refine)
    cut = sub.add_parser("cut", help="cut picked candidates to mp3")
    cut.add_argument("--plan", required=True)
    cut.add_argument("--pick", required=True, help="candidate numbers, e.g. 1,3,5 or all")
    cut.add_argument("--media", help="override the media path stored in the plan")
    cut.add_argument("--out", help="default: <plan dir>/clips")
    cut.add_argument("--adjust", action="append", metavar="N:START-END",
                     help="override a candidate's cue range, e.g. 2:315-411 (repeatable)")
    cut.add_argument("--with-srt", action="store_true", help="also slice the full-episode subtitle")
    cut.add_argument("--overwrite", action="store_true")
    cut.set_defaults(func=cmd_cut)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
