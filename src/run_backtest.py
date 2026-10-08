"""Phase 7: backtest the strategies on your downloaded history.

Needs:  python src/download_history.py   (candles)   and ideally   python src/measure_markup.py
Examples:
    python src/run_backtest.py
    python src/run_backtest.py --strategy breakout --hold 5 15 60
    python src/run_backtest.py --walk-forward --folds 5
    python src/run_backtest.py --score-holdout          # one look at the reserved holdout
    python src/run_backtest.py --plan fixed_fraction --balance 1000

Data is split three ways:
    dev        everything before the holdout; train/test or walk-forward folds live here
    holdout    the final --holdout share, reserved untouched so parameters can be chosen
               without seeing it. Scored only with --score-holdout, and only for
               candidates already selected on dev.
    quotes     logs/proposals.csv, the real quotes Deriv offered, replayed separately

Entry modes, reported side by side where they differ:
    non-overlap  no new contract until the current one expires (independent entries)
    overlap      portfolio mode: contracts may overlap, capped by --max-open
If a result disappears in non-overlapping mode, dependence was making it look stronger
than it was. `indep` is the number of independent expiry blocks behind `trades`.

Promotion needs every criterion, not a positive average: enough independent trades,
a block-bootstrap 95% CI above zero, stability across windows, no single window
carrying the profit, survival of a worse-execution rerun, a positive holdout, and a
positive replay of the quotes Deriv actually quoted.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.cost_model import CostModel, MEASURED, latest_markup_csv, load_markup_csv
from backtest.engine import keep_nonoverlapping_entries, run_backtest
from backtest.metrics import direction_test, summarize
from backtest.proposal_replay import replay_report
from backtest.walkforward import (blocksize_for, purged_walk_forward,
                                  summarize_fold_trades)
from config import (CANDLE_SECONDS, DATA_DIR, LOG_DIR, MAX_MARKUP_PCT,
                    MAX_OPEN_POSITIONS,
                    MAX_RISK_PCT_PER_TRADE, MAX_STAKE, MIN_STAKE, RISK_FRACTION, SYMBOL)
from data_feed import load_candles_csv
from staking import make_plan
from strategies import STRATEGIES, get_strategy

MODES = ("non-overlap", "overlap")

# A strategy is only a candidate with a real sample behind it; below this the
# conclusion is postponed rather than failed, because absence of evidence cuts both ways.
MIN_OOS_TRADES = 300


class StressedCostModel:
    """The same quotes, priced as if execution came out `extra_pct` points worse."""

    def __init__(self, base, extra_pct: float):
        self._base = base
        self.extra_pct = extra_pct

    def markup_pct(self, seconds, level):
        return self._base.markup_pct(seconds, level) + self.extra_pct


def build_cost_model(path_arg):
    path = path_arg or latest_markup_csv(LOG_DIR)
    table = dict(MEASURED)
    source = "built-in table (measured 2026-10-06)"
    if path:
        measured = load_markup_csv(path, SYMBOL)
        if measured:
            table.update(measured)
            source = f"{path} (+ built-in table for missing expiries)"
    return CostModel(table), source


def pct(x):
    return "   n/a" if x is None else f"{x * 100:6.1f}"


def fold_checks(outcome, oos_sets, walk_forward):
    """(stable across windows, one window dominates the profit, dominance test applies).

    Profit must be spread over several windows: a single fold carrying the whole
    result is a lucky patch, not an edge.
    """
    known = [outcome[s] for s in oos_sets if outcome.get(s)]
    profits = [o[0]["total_profit"] for o in known]

    if not walk_forward:
        tr, te = outcome.get("train"), outcome.get("test")
        stable = bool(tr and te and tr[0]["roi"] > 0 and te[0]["roi"] > 0
                      and tr[1]["p_value"] < 0.05 and te[1]["p_value"] < 0.05)
        return stable, False, False

    stable = bool(known) and sum(1 for p in profits if p > 0) * 2 >= len(known)
    checkable = len(profits) > 1
    net = sum(profits)
    dominant = bool(checkable and net > 0 and max(profits) > 0.5 * net)
    return stable, dominant, checkable


def mark(value: bool, applies: bool = True) -> str:
    if not applies:
        return "n/a"
    return "ok" if value else "NO"


def judge(independent, ci_ok, stable, dominant, checkable,
          hold_ok, hold_applies, stress_ok, stress_applies, quotes_ok,
          min_oos_trades):
    """The promotion decision, kept separate so it can be tested.

    Order matters: a thin sample postpones instead of failing, because absence of
    evidence cuts both ways, and every other criterion must pass before a strategy is
    promoted at all. `quotes_ok` is None when the proposal replay has not been made,
    which yields CANDIDATE* rather than CONFIRMED.

    CONFIRMED is reserved for a run where every check actually *ran and passed* --
    real quotes, a scored holdout and a positive stress rerun. A criterion that never
    applied cannot be reported as one that passed, so an unscored holdout or a disabled
    stress rerun downgrades CONFIRMED to CANDIDATE* (it never promotes a failure).
    """
    enough = independent >= min_oos_trades
    flags = {"enough": enough, "ci": bool(ci_ok), "stable": bool(stable),
             "dominant": bool(dominant), "checkable": bool(checkable),
             "hold": bool(hold_ok), "hold_applies": bool(hold_applies),
             "stress": bool(stress_ok), "stress_applies": bool(stress_applies),
             "quotes": quotes_ok}
    if not enough:
        return "postpone", flags
    if not (ci_ok and stable and not (dominant and checkable)):
        return "fail", flags
    if hold_applies and not hold_ok:
        return "fail", flags
    if stress_applies and not stress_ok:
        return "fail", flags
    if quotes_ok is False:
        return "fail", flags
    if quotes_ok is True and hold_applies and stress_applies:
        return "CONFIRMED", flags
    return "CANDIDATE*", flags


def make_segments(candles, args, max_hold):
    """The out-of-sample windows to report on, plus the segment names that count.

    A purge of `max_hold` candles always sits between a train window and the test
    window that follows it, in both split styles, so no training label can reach into
    the period being tested. `purged_walk_forward`'s test_size must be computed from
    what is left *after* the purge, otherwise the last fold is silently dropped and
    `--folds 4` returns three.
    """
    if not args.walk_forward:
        cut = int(len(candles) * args.split)
        test_start = cut + max_hold
        if test_start >= len(candles):
            raise SystemExit(
                f"Not enough data for a {max_hold}-candle purge after a {cut}-candle "
                f"training window out of {len(candles)}. Lower --split or --hold."
            )
        segments = {"train": candles.iloc[:cut].reset_index(drop=True),
                    "test": candles.iloc[test_start:].reset_index(drop=True)}
        return segments, ["test"]

    train_size = int(len(candles) * args.split)
    usable = len(candles) - train_size - max_hold
    if usable < args.folds:
        raise SystemExit(
            f"Not enough data for {args.folds} walk-forward folds of at least 1 candle "
            f"with a purge of {max_hold} ({usable} candles left). "
            f"Lower --folds or --split."
        )
    test_size = max(1, usable // args.folds)
    folds = list(purged_walk_forward(len(candles), train_size, test_size,
                                     purge=max_hold, step=test_size))[:args.folds]
    if not folds:
        raise SystemExit(
            f"Not enough data for {args.folds} walk-forward folds of {test_size} candles "
            f"with a purge of {max_hold}. Lower --folds or --split."
        )
    segments = {f"fold{i + 1}": candles.iloc[f.test_start:f.test_end].reset_index(drop=True)
                for i, f in enumerate(folds)}
    return segments, list(segments)


def window_problem(candles, args, max_hold) -> "str | None":
    """Why `max_hold` cannot be tested on this much data, or None if it can be.

    A contract has to expire *inside* its test window. Fold geometry is therefore per
    holding period, not per run: sizing every fold for the longest hold in the sweep
    starves the short ones, and a fold shorter than the hold itself produces no
    signals at all -- which reads as "the strategy never fired" rather than "this
    window is too small to test it in".
    """
    cut = int(len(candles) * args.split)
    if not args.walk_forward:
        if cut + max_hold >= len(candles):
            return (f"a {max_hold}-candle purge after a {cut}-candle training window "
                    f"uses up all {len(candles)} candles")
        return None
    usable = len(candles) - cut - max_hold
    test_size = max(1, usable // args.folds) if usable > 0 else 0
    if test_size <= max_hold:
        return (f"walk-forward test windows would be {test_size} candles, shorter than "
                f"the {max_hold}-candle hold, so no contract could expire inside one "
                f"(lower --folds, lower --split, or download more history)")
    return None


def evaluate(name, params, frame, cost_model, args, overlap, max_open):
    """One strategy x hold x segment x mode -> (summary, direction test)."""
    strat = get_strategy(name, **params)
    plan = (make_plan("fixed", amount=args.stake) if args.plan == "fixed"
            else make_plan("fixed_fraction", fraction=RISK_FRACTION))
    res = run_backtest(frame, strat, SYMBOL, cost_model, plan, args.balance,
                       MIN_STAKE if args.plan != "fixed" else 0.4, MAX_STAKE,
                       MAX_RISK_PCT_PER_TRADE, overlap=overlap,
                       max_open_positions=max_open)
    m = summarize(res.trades, res.balances)
    if m["trades"] == 0:
        return strat, res, m, None
    return strat, res, m, direction_test(res.trades, args.shuffles)


def mean_roi(name, params, frames, cost_model, args, overlap, max_open):
    """Mean per-trade ROI over `frames`, or None if none of them traded."""
    rois = []
    for frame in frames:
        _, res, _, _ = evaluate(name, params, frame, cost_model, args, overlap, max_open)
        rois.extend(tr.roi for tr in res.trades)
    return sum(rois) / len(rois) if rois else None


def export_report(path, summaries, source, variants):
    """Write the train / out_of_sample JSON that go_live_gate.py reads.

    Among the variants that were tried we report the one with the best out-of-sample
    result -- which is exactly why `variants_tested` is written next to it: the gate
    divides its significance level by that number, so looking at many variants cannot
    make luck look like skill.
    """
    best = None
    for name, params, hold, mode, outcome in summaries:
        train, test = outcome.get("train"), outcome.get("test")
        if not train or not test:
            continue
        if best is None or test[0]["roi"] > best[3][0]["roi"]:
            best = (name, hold, mode, test, train)
    if best is None:
        print("\n--export: no variant had both a train and a test segment; nothing written.")
        return
    name, hold, mode, test, train = best
    m, t = test
    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "strategy": name,
        "hold_candles": hold,
        "mode": mode,
        "cost_model_source": source,
        "variants_tested": variants,
        "train": {"trades": train[0]["trades"], "roi": train[0]["roi"],
                  "p_value": train[1]["p_value"] if train[1] else None},
        "out_of_sample": {"trades": m["trades"], "roi": m["roi"], "p_value": t["p_value"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nExported {name} hold={hold} {mode} to {path}")
    print(f"  out-of-sample {m['trades']} trades, ROI {m['roi']:+.1%}, p = {t['p_value']:.4f}; "
          f"{variants} variants tried.")
    if str(source).startswith("built-in"):
        print("  NOTE: the cost model is the built-in table, so go_live_gate.py will refuse it.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all", help="all or one of " + ", ".join(STRATEGIES))
    ap.add_argument("--hold", type=int, nargs="+", help="expiry in candles; several values = a sweep")
    ap.add_argument("--strike-atr", type=float, default=None, help="strike distance in ATRs")
    ap.add_argument("--split", type=float, default=0.7, help="share of dev data used as train")
    ap.add_argument("--plan", default="fixed", help="fixed or fixed_fraction")
    ap.add_argument("--stake", type=float, default=1.0, help="stake for the fixed plan")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--shuffles", type=int, default=2000)
    ap.add_argument("--markup-csv", default=None)
    ap.add_argument("--modes", choices=("non-overlap", "overlap", "both"), default="both",
                    help="entry modes to report (default: both, so overlap dependence is visible)")
    ap.add_argument("--max-open", type=int, default=None,
                    help="concurrent contract cap in portfolio mode "
                         f"(default: config MAX_OPEN_POSITIONS = {MAX_OPEN_POSITIONS})")
    ap.add_argument("--walk-forward", action="store_true",
                    help="replace the single train/test split with purged walk-forward folds")
    ap.add_argument("--folds", type=int, default=4, help="number of walk-forward test folds")
    ap.add_argument("--holdout", type=float, default=0.15,
                    help="final share of the data reserved untouched (default 0.15, 0 disables)")
    ap.add_argument("--score-holdout", action="store_true",
                    help="score the reserved holdout once, for candidates only")
    ap.add_argument("--stress-pct", type=float, default=3.0,
                    help="extra markup points assumed when testing that profit survives "
                         "worse execution (0 disables)")
    ap.add_argument("--replay", default=None, metavar="CSV",
                    help="proposal log to replay (default: logs/proposals.csv if present)")
    ap.add_argument("--no-replay", action="store_true", help="skip the proposal replay check")
    ap.add_argument("--min-oos-trades", type=int, default=MIN_OOS_TRADES,
                    help="independent out-of-sample trades needed before a verdict "
                         f"(default {MIN_OOS_TRADES}); raise it, do not loosen it")
    ap.add_argument("--export", metavar="PATH", nargs="?", const="backtest_report.json",
                    default=None,
                    help="write the train / out_of_sample numbers as JSON for go_live_gate.py "
                         "(default reports/backtest_report.json)")
    args = ap.parse_args()

    if args.export is not None and args.walk_forward:
        raise SystemExit(
            "--export reports a single train / out-of-sample split, which is what "
            "go_live_gate.py reads. Run it without --walk-forward."
        )

    path = DATA_DIR / f"{SYMBOL}_{CANDLE_SECONDS}s.csv"
    if not path.exists():
        raise SystemExit(f"No data at {path}. Run: python src/download_history.py")
    candles = load_candles_csv(path)
    cost_model, source = build_cost_model(args.markup_csv)

    if not 0.0 <= args.holdout < 1.0:
        raise SystemExit("--holdout must be in [0, 1)")
    hold_n = int(len(candles) * args.holdout)
    dev = candles.iloc[:len(candles) - hold_n].reset_index(drop=True) if hold_n else candles
    hold_frame = candles.iloc[len(candles) - hold_n:].reset_index(drop=True) if hold_n else None

    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    combos = []
    for name in names:
        for hold in (args.hold or [None]):
            params = {}
            if hold is not None:
                params["hold_candles"] = hold
            if args.strike_atr is not None:
                params["strike_atr"] = args.strike_atr
            combos.append((name, params))
    max_hold = max(get_strategy(name, **params).hold_candles for name, params in combos)

    _, oos_sets = make_segments(dev, args, max_hold)  # names are the same for every hold
    modes = list(MODES) if args.modes == "both" else [args.modes]
    max_open = args.max_open if args.max_open is not None else MAX_OPEN_POSITIONS

    # Each holding period gets its own purge and its own fold windows.
    segments_by_hold, unsized_holds = {}, {}
    for hold in sorted({get_strategy(n, **p).hold_candles for n, p in combos}):
        why = window_problem(dev, args, hold)
        if why:
            unsized_holds[hold] = why
        else:
            segments_by_hold[hold] = make_segments(dev, args, hold)[0]

    print(f"{SYMBOL}: {len(candles)} candles of {CANDLE_SECONDS}s "
          f"({len(dev)} development, {hold_n} reserved), "
          + ", ".join(oos_sets if args.walk_forward else ["train", "test"]))
    if args.walk_forward:
        print(f"Fold windows sized per holding period, each purged by its own hold; "
              f"longest hold in this run is {max_hold}. ")
    else:
        print(f"Purged by each hold's own holding period. ")
    print(f"Prices modelled as fair value x (1 + measured markup). Markup source: {source}")
    for hold, why in unsized_holds.items():
        print(f"SKIP hold={hold}: {why}.")
    if hold_n:
        scored = "scored once for candidates only" if args.score_holdout else \
                 "not evaluated; use --score-holdout to look"
        print(f"Final holdout: {hold_n} candles ({args.holdout:.0%}), {scored}.")
    if max_open <= 1 and args.modes == "overlap":
        raise SystemExit("--modes overlap needs --max-open greater than 1.")
    if max_open <= 1 and args.modes == "both":
        print("Contract cap is 1, so portfolio mode cannot overlap and would repeat the "
              "non-overlapping rows exactly; skipped. Pass --max-open N (N > 1) to run it.")
        modes = ["non-overlap"]
    print()

    header = (f"{'strategy':15}{'hold':>5}{'mode':>12}{'set':>7}{'trades':>7}{'indep':>7}"
              f"{'win%':>6}{'ROI%':>7}{'grossROI%':>10}{'markup%':>8}{'maxDD%':>7}{'p-value':>8}")
    print(header)
    print("-" * len(header))

    all_rows, summaries, tests_run = [], [], 0
    gate_skipped, risk_skipped = 0, {}
    for name, params in combos:
        hold_label = get_strategy(name, **params).hold_candles
        segments = segments_by_hold.get(hold_label)
        if segments is None:
            continue
        for mode in modes:
            overlap = mode == "overlap"
            outcome = {}
            for seg_name, frame in segments.items():
                strat, res, m, t = evaluate(name, params, frame, cost_model, args,
                                            overlap, max_open)
                gate_skipped += res.gate_rejected
                for reason, count in res.risk_rejected.items():
                    risk_skipped[reason] = risk_skipped.get(reason, 0) + count
                rows = [{"strategy": name, "hold": strat.hold_candles, "mode": mode,
                         "set": seg_name, **tr.__dict__} for tr in res.trades]
                all_rows.extend(rows)
                if m["trades"] == 0:
                    print(f"{name:15}{strat.hold_candles:>5}{mode:>12}{seg_name:>7}"
                          f"{0:>7}{0:>7}   (no trades)")
                    outcome[seg_name] = None
                    continue
                tests_run += 1
                print(f"{name:15}{strat.hold_candles:>5}{mode:>12}{seg_name:>7}"
                      f"{m['trades']:>7}{res.independent_blocks():>7}"
                      f"{m['win_rate'] * 100:6.1f}{pct(m['roi']):>7}{pct(m['gross_roi']):>10}"
                      f"{m['avg_markup_pct']:8.1f}{m['max_drawdown_pct']:7.1f}{t['p_value']:8.3f}")
                outcome[seg_name] = (m, t)
            summaries.append((name, params, hold_label, mode, outcome))

    # ---------------------------------------------------- block-bootstrap CIs
    print("\nBLOCK BOOTSTRAP 95% CI ON MEAN NET ROI (out-of-sample trades, after costs)")
    ci_header = (f"{'strategy':15}{'hold':>5}{'mode':>12}{'trades':>7}{'block':>7}"
                 f"{'mean%':>8}{'low%':>8}{'high%':>8}{'above 0?':>10}")
    print(ci_header)
    print("-" * len(ci_header))
    pooled = {}
    for name, params, hold, mode, outcome in summaries:
        rows = [r for r in all_rows
                if r["strategy"] == name and r["hold"] == hold
                and r["mode"] == mode and r["set"] in oos_sets]
        block = blocksize_for(hold, len(rows))
        ci = summarize_fold_trades(pd.DataFrame(rows), returncolumn="roi", blocksize=block)
        # entry indexes restart in each window, so de-overlap window by window
        independent = sum(
            len(keep_nonoverlapping_entries(
                sorted(r["index"] for r in rows if r["set"] == seg), hold))
            for seg in oos_sets
        )
        pooled[(name, hold, mode)] = {**ci, "independent": independent}
        verdict = "yes" if ci["profitable"] else "NO"
        if ci["trades"] == 0:
            print(f"{name:15}{hold:>5}{mode:>12}{0:>7}{0:>7}   (no out-of-sample trades)")
        else:
            print(f"{name:15}{hold:>5}{mode:>12}{ci['trades']:>7}{block:>7}"
                  f"{ci['mean'] * 100:8.2f}{ci['cilow'] * 100:8.2f}{ci['cihigh'] * 100:8.2f}"
                  f"{verdict:>10}")

    # --------------------------------------------------- signals that never traded
    total_risk = sum(risk_skipped.values())
    if gate_skipped or total_risk:
        print("\nSIGNALS NOT TRADED (they are NOT in the trade counts above)")
        if gate_skipped:
            print(f"  {gate_skipped:>6}  premium gate: the quote was more than "
                  f"{MAX_MARKUP_PCT:.1f}% over fair value")
        for reason, count in sorted(risk_skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>6}  {reason[:110]}")
    else:
        print("\nSIGNALS NOT TRADED: none -- every signal the strategy produced was taken.")

    # ------------------------------------------------- holdout + execution stress
    prelim = []
    for name, params, hold, mode, outcome in summaries:
        info = pooled[(name, hold, mode)]
        stable, dominant, _ = fold_checks(outcome, oos_sets, args.walk_forward)
        if (info["independent"] >= args.min_oos_trades and info["profitable"]
                and stable and not dominant):
            prelim.append((name, params, hold, mode))
    prelim_keys = {(n, h, m) for n, _, h, m in prelim}
    hold_scored = args.score_holdout and hold_n > 0
    stressed = args.stress_pct > 0

    robust = {}
    if args.stress_pct > 0 and prelim:
        print(f"\nEXECUTION STRESS: same entries re-priced {args.stress_pct:.1f} points "
              "of markup worse than measured")
        for name, params, hold, mode in prelim:
            overlap = mode == "overlap"
            roi = mean_roi(name, params, [segments_by_hold[hold][s] for s in oos_sets],
                           StressedCostModel(cost_model, args.stress_pct),
                           args, overlap, max_open)
            robust[(name, hold, mode)] = roi
            print(f"  {name} hold={hold} {mode}: stressed ROI "
                  f"{'n/a (no trades)' if roi is None else f'{roi * 100:.2f}%'}")

    if args.score_holdout and hold_frame is not None and prelim:
        print(f"\nHOLDOUT CONFIRMATION: {hold_n} candles never used for selection")
        for name, params, hold, mode in prelim:
            roi = mean_roi(name, params, [hold_frame], cost_model, args,
                           mode == "overlap", max_open)
            robust.setdefault((name, hold, mode), None)
            robust[(name, hold, mode, "holdout")] = roi
            print(f"  {name} hold={hold} {mode}: holdout ROI "
                  f"{'n/a (no trades)' if roi is None else f'{roi * 100:.2f}%'}")

    # ------------------------------------------------------- proposal replay
    quotes_ok = None
    if not args.no_replay:
        replay_path = args.replay or LOG_DIR / "proposals.csv"
        info = replay_report(replay_path, candles, SYMBOL)
        print("\nPROPOSAL REPLAY: contracts settled from the quotes Deriv actually offered")
        if info is None:
            print(f"  no usable proposal log at {replay_path}; nothing to replay.")
            print("  The live bot writes one via logs/proposals.csv, so this check is")
            print("  simply not made yet -- it is not a pass.")
        elif info["settled"] == 0:
            print(f"  {info['quotes']} quotes logged at {info['path']}, none settleable "
                  "against this candle history.")
        else:
            quotes_ok = info["profitable"]
            print(f"  {info['quotes']} quotes logged, {info['settled']} settled "
                  f"(block {info['block']}): mean ROI {info['mean'] * 100:.2f}%, "
                  f"95% CI [{info['cilow'] * 100:.2f}%, {info['cihigh'] * 100:.2f}%] "
                  f"-> {'above zero' if quotes_ok else 'NOT above zero'}")
            print("  Priced with no model: payoff = payoutperpoint * intrinsic, minus askprice.")

    # ------------------------------------------------------ pass/fail criteria
    quote_label = "n/a" if quotes_ok is None else ("ok" if quotes_ok else "NO")
    hold_state = ("scored" if args.score_holdout
                  else "reserved, not scored" if hold_n else "off")
    print(f"\nPASS / FAIL CRITERIA (every column must pass; under {args.min_oos_trades} "
          "independent trades postpones the call)")
    print(f"  quotes={quote_label}   holdout={hold_state}   "
          f"stress={'off' if args.stress_pct <= 0 else f'{args.stress_pct:.1f}pt'}")
    crit_header = (f"{'strategy':15}{'hold':>5}{'mode':>12}{'indepOOS':>9}"
                   f"{'>=' + str(args.min_oos_trades):>7}"
                   f"{'CI>0':>6}{'stable':>8}{'noDominant':>12}{'holdout':>8}{'stress':>7}"
                   f"{'quotes':>7}{'verdict':>12}")
    print(crit_header)
    print("-" * len(crit_header))

    candidates, postponed = [], []
    for name, params, hold, mode, outcome in summaries:
        key = (name, hold, mode)
        info = pooled[key]
        independent, ci_ok = info["independent"], info["profitable"]
        enough = independent >= args.min_oos_trades
        stable, dominant, checkable = fold_checks(outcome, oos_sets, args.walk_forward)

        hold_roi = robust.get((name, hold, mode, "holdout"))
        stress_roi = robust.get((name, hold, mode))
        verdict, flags = judge(
            independent=independent, ci_ok=ci_ok, stable=stable, dominant=dominant,
            checkable=checkable,
            hold_ok=hold_roi is not None and hold_roi > 0,
            hold_applies=hold_scored and key in prelim_keys,
            stress_ok=stress_roi is not None and stress_roi > 0,
            stress_applies=stressed and key in prelim_keys,
            quotes_ok=quotes_ok, min_oos_trades=args.min_oos_trades,
        )
        if verdict == "postpone":
            postponed.append(key)
        elif verdict in ("CANDIDATE*", "CONFIRMED"):
            candidates.append(key)

        print(f"{name:15}{hold:>5}{mode:>12}{independent:>9}"
              f"{mark(flags['enough']):>7}{mark(flags['ci']):>6}{mark(flags['stable']):>8}"
              f"{mark(not dominant, checkable):>12}"
              f"{mark(flags['hold'], flags['hold_applies']):>8}"
              f"{mark(flags['stress'], flags['stress_applies']):>7}"
              f"{quote_label:>7}{verdict:>12}")

    print("\nHOW TO READ THIS")
    print("  ROI% is the average return per 1.0 staked AFTER Deriv's markup; grossROI% is before it.")
    print("  indep is how many independent expiry blocks back `trades`; if indep << trades,")
    print("    the contracts were overlapping and the sample is smaller than it looks.")
    print("  p-value: chance that random directions at the same entries would have done as well.")
    print("    It says nothing about entries, holding period, strike selection or pricing.")
    print("  The block-bootstrap CI treats trades as chronological and resamples blocks, so it")
    print("    does not assume independent trades. A strategy passes only if its lower bound > 0.")
    print(f"  You ran {tests_run} tests, so a few p-values below 0.05 are expected by luck alone.")
    if not args.walk_forward:
        print("  This is a single train/test split; after this many looks, the test segment is")
        print("    validation data. Re-run with --walk-forward before believing any result.")
    print("  CANDIDATE* means every check that could be run passed, but the proposal replay")
    print("    has not been made. CONFIRMED also needs real quotes, a scored holdout and a")
    print("    stress rerun to come back positive.")
    if candidates:
        print(f"\nCandidates: {candidates}")
        print("  Treat these as leads to re-test on fresh data, not as proven edges.")
    if postponed:
        print(f"\nPostponed (fewer than {args.min_oos_trades} independent out-of-sample trades):")
        for name, hold, mode in postponed:
            print(f"  {name} hold={hold} {mode}")
        print("  Collect more history before drawing a conclusion, in either direction.")
    if not candidates and not postponed:
        print("\nNo strategy cleared costs with an out-of-sample CI above zero.")

    if args.export is not None:
        export_path = Path(args.export)
        if not export_path.is_absolute():
            export_path = Path(__file__).resolve().parent.parent / "reports" / export_path
        export_report(export_path, summaries, source, len(combos) * len(modes))

    if all_rows:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = LOG_DIR / f"backtest_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nAll trades saved to {out}")


if __name__ == "__main__":
    main()
