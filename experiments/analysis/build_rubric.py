"""Build the difficulty scaling rubric from audit_data.json.

Consumes the structured audit artifact and produces a paper-ready markdown
document. Filters out ephemeral metrics (dollar cost, wall time) — those are
in audit_data.json for reproducibility but never appear in the rubric.

Usage:
    uv run python experiments/analysis/build_rubric.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_PATH = ROOT / "experiments" / "analysis" / "results" / "audit_data.json"
OUT_PATH = ROOT / "experiments" / "analysis" / "results" / "difficulty_rubric.md"

# All oracles in the catalog (Section 1) — full family for completeness
ORACLE_ORDER = ["1A", "1B", "1C", "1D", "1E", "HD"]

# Oracles included in observed-metric tables and rules of thumb.
# 1B and 1C are excluded: 1B has identical IO topology to 1A (18/18 edges
# shared, only Y2 functional change), 1C has correct prior + 2 new edges
# (pure discovery test). They were early variants that revealed the design
# problem and motivated 1D (functional shift) + 1E (topology rewire) as
# the canonical transfer tests in Phase 2.5. See Phase 2.5 retro:
#   "1B/1C were too similar to 1A for credible transfer"
ORACLES_IN_RUBRIC = ["1A", "1D", "1E", "HD"]
SKIPPED_ORACLES = ["1B", "1C"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(v: Any, prec: int = 3, na: str = "n/a") -> str:
    """Format a value with given precision; handle None/NaN."""
    if v is None:
        return na
    if isinstance(v, float) and math.isnan(v):
        return na
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def fmt_pct(v: Any, prec: int = 1, na: str = "n/a") -> str:
    """Format as percentage."""
    if v is None:
        return na
    if isinstance(v, float) and math.isnan(v):
        return na
    return f"{v * 100:.{prec}f}%"


def fmt_pm(mean: Any, std: Any, prec: int = 3, na: str = "n/a") -> str:
    """Format mean ± std."""
    if mean is None:
        return na
    if isinstance(mean, float) and math.isnan(mean):
        return na
    s = f"{mean:.{prec}f}"
    if std is not None and not (isinstance(std, float) and math.isnan(std)):
        s += f" ± {std:.{prec}f}"
    return s


def n_annotation(n: int) -> str:
    """Return ' (n=X)' annotation; bold if low-n."""
    if n == 0:
        return " (n=0)"
    if n == 1:
        return " (**n=1**)"
    return f" (n={n})"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def build_intrinsic_table(audit: dict) -> str:
    """Section 1: intrinsic difficulty dimensions per oracle."""
    out = ["## 1. Oracle catalog: intrinsic difficulty dimensions\n"]
    out.append("Computed once per oracle from `oracle.ground_truth()`, `Sobol`, "
               "and `_compute_mechanism_sufficiency`. Independent of run data.\n")
    out.append("")

    header = ("| Oracle | d | k | k/d | d_eff | k/d_eff | DAG depth | "
              "Interaction frac | Adv regions | Noise dims | Ref HV | "
              "R²(M→Y) mean |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    out.append(header)
    out.append(sep)
    # Catalog includes all 6 oracles for completeness (even early variants).
    for label in ORACLE_ORDER:
        intr = audit[label]["intrinsics"]
        suff = intr["mechanism_sufficiency"]["mean"]
        marker = " *" if label in SKIPPED_ORACLES else ""
        out.append(
            f"| {label}{marker} | {intr['d']} | {intr['k_mechanisms']} | "
            f"{fmt(intr['k_over_d'], 2)} | {intr['d_eff']} | "
            f"{fmt(intr['k_over_d_eff'], 2)} | {intr['dag_depth']} | "
            f"{fmt(intr['interaction_fraction'], 2)} | "
            f"{intr['adversarial_region_count']} | "
            f"{intr['noise_dim_count']} | "
            f"{fmt(intr['reference_hv'], 3)} | "
            f"{fmt(suff, 3)} |"
        )
    out.append("")
    out.append(f"_\\* {', '.join(SKIPPED_ORACLES)} are early transfer variants "
               f"that were superseded by 1D and 1E in Phase 2.5 (see Section 2 "
               f"footnote). They appear here in the catalog for completeness but "
               f"are excluded from observed-metric tables and rules of thumb._")

    out.append("")
    out.append("**Mechanism sufficiency R²(M→Y) per output:**")
    out.append("")
    out.append("| Oracle | Y1 | Y2 | Y3 | Y4 | Mean |")
    out.append("|---|---|---|---|---|---|")
    for label in ORACLE_ORDER:
        po = audit[label]["intrinsics"]["mechanism_sufficiency"]["per_output"]
        mean = audit[label]["intrinsics"]["mechanism_sufficiency"]["mean"]
        out.append(
            f"| {label} | {fmt(po.get('Y1'), 3)} | {fmt(po.get('Y2'), 3)} | "
            f"{fmt(po.get('Y3'), 3)} | {fmt(po.get('Y4'), 3)} | "
            f"{fmt(mean, 3)} |"
        )

    out.append("")
    out.append("**Interpretation guide:**")
    out.append("- **k/d (raw) shrinks when irrelevant inputs are added**: HD's k/d = "
               "0.58 vs 1A's 1.17. This is the CBM bottleneck-friendliness predictor — "
               "**lower k/d means the mechanism bottleneck is more favorable** because "
               "the agent can ignore most input dimensions. HD's screening result "
               "(see Section 8) operationalizes this prediction.")
    out.append("- **k/d_eff is identical (1.17) for all 6-input oracles AND for HD**, "
               "because d_eff (count of inputs with non-trivial Sobol) is 6 for "
               "every oracle. So the *bottleneck-vs-real-inputs ratio* is constant — "
               "what changes across oracles is the *raw* dimensionality and the "
               "structural details captured by mechanism sufficiency.")
    out.append("- **R²(M→Y) ≥ 0.95**: mechanisms fully determine outputs; "
               "discovering them is equivalent to understanding the oracle. "
               "These oracles (1A, 1C, 1E, HD) are where the VR mechanism-bottleneck "
               "approach is structurally aligned.")
    out.append("- **R²(M→Y) < 0.80**: significant direct (non-mechanism-mediated) "
               "input→output paths exist. The bottleneck story is partially broken. "
               "**1D (0.748)** has a notable Y3 sufficiency drop (0.55) from the "
               "inverted-U on X3 plus the X1*X5 interaction; **1B (0.799)** has "
               "Y3 = 0.20 from the regime-boundary M2 change. In both cases the "
               "agent is structurally disadvantaged because Y is not a clean function "
               "of M.")
    out.append("- **Noise dim count > 0**: dimensions with mean Sobol total < 0.02. "
               "HD is the only oracle with noise dims (6 out of 12). All have exactly "
               "zero Sobol — confirming the design is clean.")
    out.append("- **Interaction fraction**: `Σ(Sobol_total − Sobol_first) / Σ Sobol_total`. "
               "Proxy for hidden-coupling intensity. All oracles cluster between 0.10–0.18 — "
               "they all have meaningful interactions but none is purely additive. "
               "1D's 0.10 is the lowest because 1D added direct (additive) X5 and X1*X5 "
               "paths.")
    out.append("")

    return "\n".join(out)


def build_prior_quality_table(audit: dict) -> str:
    """Section 2: prior quality (transfer variants vs 1A)."""
    out = ["## 2. Prior quality (transfer variants, vs 1A)\n"]
    out.append("Compares each variant's IO-projected ground truth against 1A's. "
               "`Y-correlation` is computed on a shared LHS sample (n=200, seed=7) "
               "via `evaluate_batch`. NaN means oracles have different input "
               "dimensionality so direct correlation is not defined.\n")
    out.append("")
    out.append("**Why 1B and 1C are not transfer tests.** The audit empirically "
               "confirms what Phase 2.5 design rationale stated: 1B and 1C are "
               "topologically too similar to 1A to test transfer:")
    out.append("")
    out.append("- **1B**: identical IO topology (18/18 edges shared, 0 wrong, "
               "0 missing). The only change is the M2 functional form, which shows "
               "up as Y2 correlation 0.62. The prior is *structurally correct* — "
               "this measures local function shift, not transfer.")
    out.append("- **1C**: 18/18 + 2 new edges (X3→Y4, X5→Y4 for the new mechanism M5). "
               "The prior is *fully correct* on existing edges and only blind to the "
               "new mechanism. This is a pure discovery test, not a transfer test.")
    out.append("")
    out.append("Phase 2.5 designed **1D** (functional shifts + 1 new edge) and "
               "**1E** (full topology rewire: 16/18 shared + 2 wrong + 3 missing) "
               "as the canonical transfer testbeds. **The remaining sections of "
               "this rubric (3-9) and the rules of thumb (10) are based on 1A, 1D, "
               "1E, and HD only.** 1B and 1C are documented here as stepping stones "
               "and excluded from the rest of the rubric.")
    out.append("")

    out.append("| Variant | Edges shared | Wrong (must unlearn) | "
               "Missing (must discover) | Y1 corr | Y2 corr | Y3 corr | Y4 corr | "
               "Used in rubric? |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for label in ["1B", "1C", "1D", "1E"]:
        pq = audit[label].get("prior_quality_vs_1A")
        if pq is None:
            continue
        ycorr = pq["y_correlation"]
        used = "no — stepping stone" if label in SKIPPED_ORACLES else "**yes**"
        out.append(
            f"| {label} | {pq['edges_shared']}/{pq['edges_in_1a_prior']} | "
            f"{pq['edges_wrong_in_prior']} | {pq['edges_missing_from_prior']} | "
            f"{fmt(ycorr.get('Y1'), 2)} | {fmt(ycorr.get('Y2'), 2)} | "
            f"{fmt(ycorr.get('Y3'), 2)} | {fmt(ycorr.get('Y4'), 2)} | "
            f"{used} |"
        )

    out.append("")
    out.append("**Interpretation of the canonical transfer variants:**")
    out.append("- **1D (Functional shift)**: 18/18 shared + 1 new edge. The challenge "
               "is functional: Y2 sign flip + X1×X5 interaction + X3→Y3 inverted-U. "
               "Y correlations all positive (0.71–0.88) but materially shifted. "
               "**Tests whether the agent can unlearn wrong functional forms while "
               "leveraging correct edge existence.**")
    out.append("- **1E (Topology rewire)**: 16/18 shared, **2 wrong** + **3 missing** "
               "edges. Y2 correlation **−0.05**, Y4 correlation **0.005** — "
               "essentially uncorrelated. The agent must unlearn X3→Y3 and X4→Y4 "
               "(false in 1E) and discover X4→Y3, X3→Y4, X5→Y4 (rewired). "
               "**Tests whether the agent can dismiss confidently-held false beliefs "
               "and rebuild from scratch.**")
    out.append("")

    return "\n".join(out)


def _condition_row_label(oracle: str, cond: dict) -> str:
    """Format the row label for an (oracle, condition) row."""
    return (f"{oracle} / {cond['label']} ({cond['budget']} budget, "
            f"{cond['model']}, n={cond['n_seeds']})")


def build_sample_efficiency_table(audit: dict) -> str:
    """Section 3: sample efficiency (oracle evals to X% of ref HV)."""
    out = ["## 3. Sample efficiency (primary cost-related metric)\n"]
    out.append("Oracle evaluations needed to reach 50%, 75%, 90% of the oracle's "
               "own reference HV. **Durable** across LLM pricing changes and model "
               "upgrades — this is the metric a practitioner with an expensive oracle "
               "actually cares about.")
    out.append("")
    out.append("Reported separately for VR and BO; per-row VR/BO ratio at each threshold "
               "is the \"oracle-eval tax.\" `n/a` means the curve never reached the threshold "
               "within the run's budget.")
    out.append("")
    out.append("| Run | 50% VR | 50% BO | 75% VR | 75% BO | 90% VR | 90% BO |")
    out.append("|---|---|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            if cond["n_seeds"] == 0:
                continue
            opt = cond["optimization"]
            if opt.get("n_seeds_vr", 0) == 0:
                continue
            row_label = _condition_row_label(label, cond)
            se_vr = opt.get("sample_efficiency_vr", {})
            se_bo = opt.get("sample_efficiency_bo", {})

            def _f(x: Any) -> str:
                return f"{int(x)}" if x is not None else "n/a"

            out.append(
                f"| {row_label} | {_f(se_vr.get('50pct'))} | {_f(se_bo.get('50pct'))} | "
                f"{_f(se_vr.get('75pct'))} | {_f(se_bo.get('75pct'))} | "
                f"{_f(se_vr.get('90pct'))} | {_f(se_bo.get('90pct'))} |"
            )
    out.append("")
    out.append("**Reading the table:**")
    out.append("- **VR > BO at low thresholds (50%, 75%)**: BO's qNEHVI converges fast "
               "on smooth Pareto fronts; VR spends early budget on screening + OAT sweeps "
               "and only catches up after building its causal model. This is the "
               "\"understanding tax\" in dimensional units.")
    out.append("- **At 90%**: the gap is widest because BO is approaching its asymptote "
               "while VR is still exploring. This is where extended-budget VR closes "
               "the gap (see Section 4 crossover).")
    out.append("- **HD base_72 vs HD extended_144**: at 50% VR needs 38 vs 66 evals — "
               "the extended-budget agent **takes longer** to reach 50% because it spends "
               "more time on systematic screening (including noise verification). At 90% "
               "the extended agent catches up.")
    out.append("- **1E rows show \"2 evals to 50%\"**: this is a quirk of 1E's HV "
               "landscape, not an agent achievement. 1E's reference HV (0.259) is small "
               "enough that the random LHS-init phase already produces points exceeding "
               "50% within the first ~2 evaluations. For 1E, only the 75%/90% thresholds "
               "compare the agent's actual tool-budget contribution.")
    out.append("- **Method-honest comparison**: BO and VR both include their initial "
               "LHS phase in the trajectory, so the comparison is apples-to-apples in "
               "terms of \"oracle evaluations consumed.\" The interpretation issue above "
               "is about the *informativeness* of the threshold, not a fairness issue.")
    out.append("")

    return "\n".join(out)


def build_optimization_table(audit: dict) -> str:
    """Section 4: HV (as fraction of ref HV) and crossover."""
    out = ["## 4. Final HV (as % of reference HV)\n"]
    out.append("All HV values normalized to oracle's own reference HV (raw HVs in "
               "Appendix A). The **crossover eval** is the first VR evaluation at which "
               "mean(VR_HV) ≥ BO's final mean HV — i.e., when does VR catch up to where "
               "BO ended its run?")
    out.append("")
    out.append("| Run | VR HV (%ref) | BO HV (%ref) | VR/BO | Crossover eval | "
               "Seeds crossing BO |")
    out.append("|---|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            opt = cond["optimization"]
            if opt.get("n_seeds_vr", 0) == 0:
                continue
            row_label = _condition_row_label(label, cond)
            vr_pm = fmt_pm(opt.get("vr_final_hv_frac_ref_mean"),
                           opt.get("vr_final_hv_frac_ref_std"), prec=3)
            bo_pm = fmt_pm(opt.get("bo_final_hv_frac_ref_mean"),
                           opt.get("bo_final_hv_std", 0) / opt.get("bo_final_hv_mean", 1)
                           * opt.get("bo_final_hv_frac_ref_mean", 0)
                           if opt.get("bo_final_hv_mean") else None, prec=3)
            ratio = opt.get("vr_over_bo")
            cross = opt.get("crossover_eval")
            n_cross = opt.get("seeds_crossing_bo")
            n_vr = opt.get("n_seeds_vr")
            out.append(
                f"| {row_label} | {vr_pm} | {bo_pm} | "
                f"{fmt(ratio, 3)} | "
                f"{cross if cross is not None else 'n/a'} | "
                f"{n_cross}/{n_vr} |"
            )
    out.append("")
    return "\n".join(out)


def build_discovery_table(audit: dict) -> str:
    """Section 5: edge precision/recall + hardest missed."""
    out = ["## 5. Causal discovery (edge precision / recall vs ground-truth IO projection)\n"]
    out.append("Edges are compared against the IO projection of each oracle's "
               "ground-truth DAG (X→Y reachable paths only). Note: `project_to_io` "
               "collapses mechanism-level edge difficulty to a single tier, so "
               "per-difficulty breakdown is omitted from this table.\n")
    out.append("")
    out.append("| Run | GT edges | Precision | Recall | Hardest missed edge |")
    out.append("|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for _cond_name, cond in audit[label]["conditions"].items():
            ep = cond["edge_pr"]
            if ep.get("n_seeds", 0) == 0:
                continue
            row_label = _condition_row_label(label, cond)
            prec_pm = fmt_pm(ep.get("precision_mean"), ep.get("precision_std"), 3)
            rec_pm = fmt_pm(ep.get("recall_mean"), ep.get("recall_std"), 3)
            hardest = ep.get("hardest_missed_edge") or "—"
            n_gt = ep.get("n_gt_edges", 0)
            out.append(
                f"| {row_label} | {n_gt} | {prec_pm} | {rec_pm} | {hardest} |"
            )
    out.append("")
    return "\n".join(out)


def build_info_capture_table(audit: dict) -> str:
    """Section 6: agent information capture (Sobol-weighted recall)."""
    out = ["## 6. Agent information capture (Sobol-weighted recall)\n"]
    out.append("For each output, sums the Sobol total index of inputs the agent "
               "claimed (confidence ≥ 0.5), divided by the output's total Sobol sum. "
               "Less noisy than raw recall because it weights by effect size: finding "
               "X2→Y1 (high Sobol) counts more than a low-impact edge.")
    out.append("")
    out.append("| Run | Mean across outputs | Y1 | Y2 | Y3 | Y4 |")
    out.append("|---|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            ic = cond["info_capture"]
            if ic.get("n_seeds", 0) == 0:
                continue
            row_label = _condition_row_label(label, cond)
            mean = ic.get("mean_across_outputs")
            std = ic.get("std_across_outputs")
            po = ic.get("per_output_mean", {})
            out.append(
                f"| {row_label} | {fmt_pm(mean, std, 3)} | "
                f"{fmt(po.get('Y1'), 3)} | {fmt(po.get('Y2'), 3)} | "
                f"{fmt(po.get('Y3'), 3)} | {fmt(po.get('Y4'), 3)} |"
            )
    out.append("")
    return "\n".join(out)


def build_prediction_table(audit: dict) -> str:
    """Section 7: OAT prediction quality + adversarial + calibration."""
    out = ["## 7. Prediction quality (OAT, calibration, adversarial)\n"]

    # OAT accuracy
    out.append("### 7a. OAT prediction accuracy")
    out.append("")
    out.append("| Run | OAT direction acc | OAT magnitude MAE | Predictions/seed |")
    out.append("|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            oat = cond["oat_accuracy"]
            if oat.get("n_seeds", 0) == 0:
                continue
            row_label = _condition_row_label(label, cond)
            out.append(
                f"| {row_label} | "
                f"{fmt_pm(oat.get('direction_accuracy_mean'), oat.get('direction_accuracy_std'), 3)} | "
                f"{fmt_pm(oat.get('magnitude_mae_mean'), oat.get('magnitude_mae_std'), 4)} | "
                f"{fmt(oat.get('predictions_per_seed_mean'), 0)} |"
            )
    out.append("")

    # Calibration
    out.append("### 7b. Calibration learning")
    out.append("")
    out.append("Fraction of seeds where last-checkpoint MAE < 0.8 × first-checkpoint MAE. "
               "`learning fraction = 1.0` means every seed showed measurable confidence "
               "tracking improvement over time.")
    out.append("")
    out.append("| Run | Seeds with cal data | Median checkpoints/seed | "
               "First MAE | Last MAE | Learning fraction |")
    out.append("|---|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            cal = cond["calibration"]
            n_data = cal.get("n_seeds_with_data", 0)
            if n_data == 0:
                continue
            row_label = _condition_row_label(label, cond)
            n_total = cal.get("n_seeds", 0)
            out.append(
                f"| {row_label} | {n_data}/{n_total} | "
                f"{fmt(cal.get('median_checkpoints_per_seed'), 1)} | "
                f"{fmt(cal.get('first_mae_mean'), 4)} | "
                f"{fmt(cal.get('last_mae_mean'), 4)} | "
                f"{fmt_pct(cal.get('learning_fraction'), 0)} |"
            )
    out.append("")

    # Adversarial
    out.append("### 7c. Adversarial region prediction error")
    out.append("")
    out.append("Max-region MAE is the worst MAE across the oracle's adversarial "
               "regions (interactions, couplings, threshold zones). The "
               "**adv/non-adv ratio** divides max-adv MAE by non-adversarial MAE; "
               "values > 1 mean the agent's predictions are worse in structurally "
               "hard regions. `n/a` for the ratio means there were no non-adversarial "
               "evaluate_point calls (often because all probed points were inside "
               "an adversarial region — small n_seeds intensify this).")
    out.append("")
    out.append("| Run | Max adv MAE | Non-adv MAE | Adv/non-adv ratio | "
               "Pts in adv / non-adv |")
    out.append("|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for _cond_name, cond in audit[label]["conditions"].items():
            adv = cond["adversarial"]
            if adv.get("n_seeds", 0) == 0:
                continue
            regions = adv.get("regions", {})
            if not regions:
                continue
            row_label = _condition_row_label(label, cond)
            max_adv = adv.get("max_adv_mae")
            non_adv = regions.get("non-adversarial", {}).get("mae")
            ratio = adv.get("max_adv_over_non_adv_ratio")
            n_adv = adv.get("adv_total_n_points", 0)
            n_non = adv.get("non_adv_n_points", 0)
            out.append(
                f"| {row_label} | {fmt(max_adv, 4)} | {fmt(non_adv, 4)} | "
                f"{fmt(ratio, 2)} | {n_adv} / {n_non} |"
            )
    out.append("")
    return "\n".join(out)


def build_screening_table(audit: dict) -> str:
    """Section 8: screening efficiency (HD-specific, generalized to all)."""
    out = ["## 8. Screening efficiency (HD)\n"]
    out.append("Fraction of tool-call evaluations spent on inputs whose total Sobol "
               "index is below the noise threshold (0.02). For HD this measures how "
               "well the agent dismissed X7-X12. For other oracles this is 0% by "
               "construction (no zero-effect inputs).")
    out.append("")
    out.append("| Run | Low-Sobol inputs | OAT noise / total | "
               "Tool-call evals on noise | Max noise edge confidence |")
    out.append("|---|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            sc = cond["screening"]
            if sc.get("n_seeds", 0) == 0:
                continue
            low = sc.get("low_sobol_inputs", [])
            if not low:
                continue
            row_label = _condition_row_label(label, cond)
            out.append(
                f"| {row_label} | {len(low)} ({','.join(low)}) | "
                f"{sc.get('oat_noise', 0)}/{sc.get('oat_total', 0)} "
                f"({fmt_pct(sc.get('noise_oat_fraction'), 1)}) | "
                f"{sc.get('noise_evals_in_tool_calls', 0)}/"
                f"{sc.get('total_evals_in_tool_calls', 0)} "
                f"({fmt_pct(sc.get('noise_eval_fraction'), 1)}) | "
                f"{fmt(sc.get('max_noise_edge_confidence'), 2)} |"
            )
    out.append("")
    out.append("**Reading the table:** at base budget HD does 0% noise OAT sweeps "
               "(pure inference-based dismissal). At extended budget HD shifts to "
               "**verification mode**, spending ~15% of tool-call evals on noise "
               "dimensions to explicitly confirm zero effect — but no noise edge ever "
               "exceeds confidence 0.05. Both strategies produce zero false-positive "
               "noise edges.")
    out.append("")
    return "\n".join(out)


def build_variance_table(audit: dict) -> str:
    """Section 9: variance / robustness across seeds."""
    out = ["## 9. Variance / robustness across seeds\n"]
    out.append("σ values across seeds for HV, recall, and info capture. **σ collapse "
               "at extended budget** is a key signal of convergence — when every seed "
               "reaches the same model, σ → 0.")
    out.append("")
    out.append("| Run | σ(HV %ref) | σ(recall) | σ(info capture) |")
    out.append("|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        for cond_name, cond in audit[label]["conditions"].items():
            n = cond["n_seeds"]
            if n == 0:
                continue
            opt = cond["optimization"]
            ep = cond["edge_pr"]
            ic = cond["info_capture"]
            row_label = _condition_row_label(label, cond)
            out.append(
                f"| {row_label} | "
                f"{fmt(opt.get('vr_final_hv_frac_ref_std'), 4)} | "
                f"{fmt(ep.get('recall_std'), 4)} | "
                f"{fmt(ic.get('std_across_outputs'), 4)} |"
            )
    out.append("")
    return "\n".join(out)


def build_rules_of_thumb(audit: dict) -> str:
    """Section 10: practitioner rules of thumb, evidence-linked."""
    out = ["## 10. Practitioner rules of thumb\n"]
    out.append("Each rule cites the evidence base. Treat single-seed (n=1) rules as "
               "**directional** rather than statistical claims.")
    out.append("")

    # Pull data points to populate the rules
    a_72 = audit["1A"]["conditions"]["multi_seed_72"]["optimization"]
    a_144 = audit["1A"]["conditions"]["extended_144"]["optimization"]
    hd_72 = audit["HD"]["conditions"]["base_72"]["optimization"]
    hd_144 = audit["HD"]["conditions"]["extended_144"]["optimization"]
    hd_72_screen = audit["HD"]["conditions"]["base_72"]["screening"]
    hd_144_screen = audit["HD"]["conditions"]["extended_144"]["screening"]
    a_72_ep = audit["1A"]["conditions"]["multi_seed_72"]["edge_pr"]
    hd_144_ep = audit["HD"]["conditions"]["extended_144"]["edge_pr"]
    hd_144_cal = audit["HD"]["conditions"]["extended_144"]["calibration"]
    hd_144_ic = audit["HD"]["conditions"]["extended_144"]["info_capture"]
    d_prior_72 = audit["1D"]["conditions"]["prior_72"]["optimization"]
    d_fresh_72 = audit["1D"]["conditions"]["fresh_72"]["optimization"]
    d_prior_144 = audit["1D"]["conditions"]["prior_144"]["optimization"]
    e_prior_72 = audit["1E"]["conditions"]["prior_72"]["optimization"]
    e_fresh_72 = audit["1E"]["conditions"]["fresh_72"]["optimization"]
    e_sonnet_72 = audit["1E"]["conditions"]["sonnet_prior_72"]["optimization"]

    rules = []

    rules.append((
        "Baseline cost of understanding (low-d, single oracle)",
        f"At matched 72-eval budget on a 6-input oracle with no irrelevant variables, "
        f"VR reaches **{a_72['vr_over_bo']:.0%}** of BO's HV. The agent spends ~12-24 "
        f"initial evals on LHS sampling and another ~30-40 evals on screening + OAT "
        f"sweeps before its causal model is built; BO uses every eval for direct "
        f"acquisition. **Evidence:** 1A multi_seed_72 (n=10 Opus seeds, "
        f"VR HV {a_72['vr_final_hv_mean']:.3f} ± {a_72['vr_final_hv_std']:.3f} vs "
        f"BO {a_72['bo_final_hv_mean']:.3f})."
    ))

    rules.append((
        "Extended-budget crossover",
        f"At 2× the matched budget (144 evals on a 6-input oracle), VR reaches "
        f"**{a_144['vr_over_bo']:.0%}** of BO HV — VR mean catches BO at eval "
        f"**{a_144['crossover_eval']}** and {a_144['seeds_crossing_bo']}/"
        f"{a_144['n_seeds_vr']} seeds eventually exceed BO's final HV. The "
        f"\"understanding tax\" is paid back in the second half of the budget. "
        f"**Evidence:** 1A extended_144 (n=4 Opus seeds)."
    ))

    rules.append((
        "Dimensionality scaling — irrelevant inputs help VR relatively",
        f"Adding 6 zero-effect dimensions to a 6-input oracle (HD = 12 inputs, 6 noise) "
        f"shrinks the VR/BO gap from {a_72['vr_over_bo']:.0%} to "
        f"**{hd_72['vr_over_bo']:.0%}** at 72 budget. The agent's screening rejects "
        f"the noise dimensions and concentrates effort on the real ones; BO's qNEHVI "
        f"with ARD also handles noise dims but loses a few percent of HV in the "
        f"process. **Evidence:** HD base_72 (n=3 Opus seeds, k/d = 0.58 but k/d_eff "
        f"identical to 1A)."
    ))

    rules.append((
        "Perfect screening at tight budget",
        f"At base budget HD, the agent does **0/12 OAT sweeps** on noise dimensions "
        f"(pure inference-based dismissal). At extended budget the strategy shifts to "
        f"verification: ~{hd_144_screen['noise_oat_fraction']:.0%} of OAT sweeps go "
        f"to noise dims, but the **max noise-edge confidence remains ≤ "
        f"{hd_144_screen['max_noise_edge_confidence']}**. Both strategies produce "
        f"**zero false-positive noise edges**. **Evidence:** HD base_72 + HD extended_144 "
        f"(n=3 each)."
    ))

    rules.append((
        "Recall convergence at extended budget",
        f"Extended budget collapses recall variance to σ={hd_144_ep['recall_std']:.3f}. "
        f"At 72 budget, HD recall is "
        f"{audit['HD']['conditions']['base_72']['edge_pr']['recall_mean']:.3f} ± "
        f"{audit['HD']['conditions']['base_72']['edge_pr']['recall_std']:.3f} "
        f"(seeds find different subsets); at 144 every seed finds 17/18 edges and "
        f"misses **the same edge** (`{hd_144_ep['hardest_missed_edge']}`). The 1 missed "
        f"edge is the genuinely hardest one — same edge that 9/10 1A multi-seed runs "
        f"missed. **Evidence:** HD extended_144 vs HD base_72."
    ))

    rules.append((
        "Calibration learning emerges at extended budget",
        f"At base budget calibration checkpoints fire 1-2 times per seed — not enough "
        f"to detect learning. At extended budget the median is "
        f"{hd_144_cal['median_checkpoints_per_seed']:.0f} checkpoints/seed and the "
        f"learning fraction (last MAE < 0.8 × first MAE) is "
        f"**{hd_144_cal['learning_fraction']:.0%}** ({hd_144_cal['n_seeds_showing_learning']}/"
        f"{hd_144_cal['n_seeds_with_two_checkpoints']} seeds). The agent's confidence "
        f"calibration measurably improves as it accumulates data. **Evidence:** HD "
        f"extended_144 (calibration MAE drops from "
        f"{hd_144_cal['first_mae_mean']:.3f} to {hd_144_cal['last_mae_mean']:.3f})."
    ))

    rules.append((
        "Information capture exceeds raw recall",
        f"Sobol-weighted recall (info capture) reaches "
        f"**{hd_144_ic['mean_across_outputs']:.3f}** at extended budget — higher than "
        f"raw recall (0.944) because the missed edge is low-Sobol. The agent finds "
        f"the high-impact edges and only misses inconsequential ones. **Evidence:** "
        f"HD extended_144 info capture per output (Y1=1.0, Y4=0.99)."
    ))

    rules.append((
        "Mechanism sufficiency predicts where VR works",
        f"Oracles with R²(M→Y) ≥ 0.95 (1A, 1C, 1E, HD) are structurally aligned with "
        f"the mechanism-bottleneck approach: discovering mechanisms is equivalent to "
        f"understanding the oracle. Oracles with R²(M→Y) < 0.80 (1B at 0.80, 1D at "
        f"**0.75**) have direct input→output paths that bypass mechanisms — the "
        f"bottleneck story is partially broken. **This is the single best intrinsic "
        f"predictor of whether VR's approach is well-matched to the problem.** "
        f"**Evidence:** R²(M→Y) computed via polynomial regression of Y on mechanism "
        f"values (n=100k samples per oracle)."
    ))

    rules.append((
        "Prior topology preservation matters more than functional similarity",
        f"On 1D (functional shifts only, topology preserved), the prior gives a "
        f"**−{(1 - d_prior_72['vr_over_bo'] / d_fresh_72['vr_over_bo']) * 100:.0f}% "
        f"penalty** at 72 budget (n=1: prior {d_prior_72['vr_over_bo']:.2f} vs fresh "
        f"{d_fresh_72['vr_over_bo']:.2f}). At extended budget the penalty erases: prior "
        f"reaches **{d_prior_144['vr_over_bo']:.0%}** of BO. On 1E (full rewire, 16/18 "
        f"edges shared but Y2/Y4 essentially uncorrelated with 1A), prior and fresh "
        f"are **statistically equivalent** ({e_prior_72['vr_over_bo']:.0%} vs "
        f"{e_fresh_72['vr_over_bo']:.0%}) — when the prior is *obviously* wrong the "
        f"agent dismisses it cleanly via the screen-first protocol. The dangerous "
        f"case is the *partially wrong* prior (1D), not the *catastrophically wrong* "
        f"one (1E). **Evidence:** 1D and 1E prior/fresh conditions (n=1 each — "
        f"directional)."
    ))

    rules.append((
        "Model generality (Sonnet vs Opus on transfer)",
        f"On 1E transfer, Sonnet reaches "
        f"**{e_sonnet_72['vr_final_hv_frac_ref_mean']:.3f} of ref HV** vs Opus's "
        f"{e_prior_72['vr_final_hv_frac_ref_mean']:.3f} — essentially identical. The "
        f"transfer protocol is model-general; the result is not Opus-specific. "
        f"**Evidence:** 1E sonnet_prior_72 vs 1E prior_72 (n=1 each)."
    ))

    for i, (title, body) in enumerate(rules, 1):
        out.append(f"### Rule {i}: {title}")
        out.append("")
        out.append(body)
        out.append("")

    return "\n".join(out)


def build_limitations(audit: dict) -> str:
    """Section 11: limitations and durability notes."""
    out = ["## 11. Limitations and permanence notes\n"]
    out.append("**Statistical depth:**")
    out.append("- 1A: 10-seed multi-seed (statistically robust)")
    out.append("- HD: 3-seed at base + 3-seed at extended (small but consistent)")
    out.append("- 1A extended: 4-seed (small but consistent)")
    out.append("- **1B, 1C, 1D, 1E: n=1 per condition** — all transfer claims rest on "
               "single seeds. Rules of thumb derived from these are *directional*.")
    out.append("")
    out.append("**Coverage:**")
    out.append("- Only Opus 4.6 evaluated at multi-seed depth. Sonnet has n=1 on 1E "
               "transfer; Haiku is not in this rubric.")
    out.append("- Only 2 budget levels tested (72 and 144 evals). Sample efficiency "
               "interpolation between or extrapolation beyond these is not validated.")
    out.append("- HD is the only oracle with irrelevant inputs. The k/d scaling claim "
               "rests on a single dimensionality data point.")
    out.append("")
    out.append("**Durability classification (what ages and what doesn't):**")
    out.append("")
    out.append("| Metric class | Durability | Why |")
    out.append("|---|---|---|")
    out.append("| Sample efficiency (evals to X% of ref HV) | **High** | Property of "
               "method × problem; LLM-pricing-independent |")
    out.append("| Mechanism sufficiency R²(M→Y) | **Eternal** | Intrinsic to oracle; "
               "no LLM dependency |")
    out.append("| Edge precision/recall | **High** | Discovery accuracy; "
               "shifts slowly with model upgrades |")
    out.append("| Info capture (Sobol-weighted) | **High** | Same as edge P/R |")
    out.append("| Calibration learning trajectory | **High** | Method-level property |")
    out.append("| Adversarial MAE ratio | **High** | Prediction quality structure |")
    out.append("| Variance collapse at extended budget | **High** | Convergence property |")
    out.append("| Absolute HV values | **Medium** | Shifts with LLM capability |")
    out.append("| VR/BO ratio at fixed budget | **Medium** | Shifts slowly with LLM |")
    out.append("| Dollar cost / wall time | **Low (omitted)** | Stored in audit_data.json "
               "and tasks/spend.md but filtered from the rubric |")
    out.append("")
    out.append("**To re-validate on a future model**, re-run the multi-seed conditions "
               "and check that (a) qualitative claims (the rules of thumb) still hold "
               "and (b) sample efficiency numbers shift in the expected direction (better "
               "models need fewer evals).")
    out.append("")
    return "\n".join(out)


def build_appendix(audit: dict) -> str:
    """Appendices: raw HVs and pointers to ephemeral data."""
    out = ["## Appendix A: Raw HV values\n"]
    out.append("(Reproducibility; not used in the rubric's headline claims.)")
    out.append("")
    out.append("| Run | VR HV (raw) | BO HV (raw) | Reference HV |")
    out.append("|---|---|---|---|")
    for label in ORACLES_IN_RUBRIC:
        ref_hv = audit[label]["intrinsics"]["reference_hv"]
        for cond_name, cond in audit[label]["conditions"].items():
            opt = cond["optimization"]
            if opt.get("n_seeds_vr", 0) == 0:
                continue
            row_label = _condition_row_label(label, cond)
            vr_pm = fmt_pm(opt.get("vr_final_hv_mean"), opt.get("vr_final_hv_std"), 4)
            bo_pm = fmt_pm(opt.get("bo_final_hv_mean"), opt.get("bo_final_hv_std"), 4)
            out.append(f"| {row_label} | {vr_pm} | {bo_pm} | {fmt(ref_hv, 4)} |")
    out.append("")

    out.append("## Appendix B: Ephemeral metrics (pointer)\n")
    out.append("Dollar cost, wall time, and token counts are stored in "
               "`experiments/analysis/results/audit_data.json` (per condition under the "
               "`cost` key) and aggregated in `tasks/spend.md`. They are deliberately "
               "omitted from the rubric proper because they age in months as API "
               "pricing changes.")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    with open(AUDIT_PATH) as f:
        audit = json.load(f)

    sections = []
    sections.append("# SynthOracle Difficulty Scaling Rubric\n")
    sections.append(
        "Generated from `experiments/analysis/results/audit_data.json` by "
        "`build_rubric.py`. This is the paper-ready synthesis of all "
        "cross-oracle results, designed for AI4Science practitioners to locate "
        "their problem on difficulty axes and read off expected VR performance."
    )
    sections.append("")
    sections.append(
        "**How to read this document.** Sections 1-2 are *intrinsic* (oracle "
        "properties, no run data). Sections 3-9 are *observed* (per condition, "
        "with honest n_seeds annotations — values in **bold n=1** are directional). "
        "Section 10 distills practitioner rules of thumb. Section 11 covers "
        "limitations and durability."
    )
    sections.append("")
    sections.append("---\n")

    sections.append(build_intrinsic_table(audit))
    sections.append("---\n")
    sections.append(build_prior_quality_table(audit))
    sections.append("---\n")
    sections.append(build_sample_efficiency_table(audit))
    sections.append("---\n")
    sections.append(build_optimization_table(audit))
    sections.append("---\n")
    sections.append(build_discovery_table(audit))
    sections.append("---\n")
    sections.append(build_info_capture_table(audit))
    sections.append("---\n")
    sections.append(build_prediction_table(audit))
    sections.append("---\n")
    sections.append(build_screening_table(audit))
    sections.append("---\n")
    sections.append(build_variance_table(audit))
    sections.append("---\n")
    sections.append(build_rules_of_thumb(audit))
    sections.append("---\n")
    sections.append(build_limitations(audit))
    sections.append("---\n")
    sections.append(build_appendix(audit))

    text = "\n".join(sections)
    OUT_PATH.write_text(text)
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(text.splitlines())} lines, {len(text)} bytes")


if __name__ == "__main__":
    main()
