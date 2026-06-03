"""Append SHAP and statistical interpretation sections to analysis.ipynb."""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "analysis.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Section 10 — SHAP on the stacking meta-classifier
# ---------------------------------------------------------------------------

S10_HEADER = """\
## Section 10 — SHAP Interpretation of the Ensemble

We apply **SHAP (SHapley Additive exPlanations)** to the logistic-stacking
meta-classifier learned in Section 9.5. The meta-classifier has exactly two
inputs — the SBI score and the Xception score — so SHAP values give a clean,
quantitative answer to the question *"how does the ensemble combine the two
models' evidence?"*

- A **positive** SHAP value for SBI on a video means SBI's score pushed the
  ensemble's decision *towards FAKE* for that video.
- A **negative** SHAP value means it pushed the decision *towards REAL*.
- The magnitude is in the same units as the meta-classifier's log-odds output.

Note: this is interpretability *at the ensemble level*. For interpretability at
the pixel level (which face regions each base model attended to), see Section 7
where we already produced Grad-CAM overlays.
"""

S10_FIT_AND_EXPLAIN = """\
# Section 10.1 — Fit the meta-classifier and build a SHAP explainer
import shap
from sklearn.linear_model import LogisticRegression

# Re-fit the same logistic stacker on Celeb-DF validation predictions.
# (Section 9 already did this internally; we reconstruct it here so we have
# a sklearn model object to hand to shap.)
val_X = np.column_stack([val_sbi, val_xc])
meta = LogisticRegression(max_iter=1000)
meta.fit(val_X, val_y)

print(f'Meta-classifier coefficients: SBI={meta.coef_[0][0]:+.4f}  '
      f'Xception={meta.coef_[0][1]:+.4f}  bias={meta.intercept_[0]:+.4f}')

# LinearExplainer is the closed-form SHAP solver for linear models.
# Background = the validation set; we explain the held-out 80% Celeb-DF test slice.
test_X = np.column_stack([celeb_test_sbi, celeb_test_xc])
explainer = shap.LinearExplainer(meta, val_X)
shap_values = explainer.shap_values(test_X)   # shape (N_test, 2)

print(f'SHAP values computed for {shap_values.shape[0]} test videos.')
print(f'Mean |SHAP|: SBI={np.abs(shap_values[:, 0]).mean():.4f}  '
      f'Xception={np.abs(shap_values[:, 1]).mean():.4f}')
"""

S10_SUMMARY_AND_BEESWARM = """\
# Section 10.2 — Global view: which model contributes more across all videos?

feature_names = ['SBI_score', 'Xception_score']
expl = shap.Explanation(
    values=shap_values,
    base_values=np.full(shap_values.shape[0], explainer.expected_value),
    data=test_X,
    feature_names=feature_names,
)

# Bar plot: mean absolute SHAP per feature (overall contribution magnitude).
plt.figure()
shap.plots.bar(expl, show=False)
plt.title('Mean |SHAP| — overall contribution to ensemble decision')
plt.tight_layout()
bar_out = RESULTS_DIR / 'shap_bar_global.png'
plt.savefig(bar_out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved -> {bar_out}')

# Beeswarm: distribution of SHAP values across all test videos.
plt.figure()
shap.plots.beeswarm(expl, show=False)
plt.title('SHAP beeswarm — per-video contributions on Celeb-DF test')
plt.tight_layout()
bee_out = RESULTS_DIR / 'shap_beeswarm.png'
plt.savefig(bee_out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved -> {bee_out}')
"""

S10_FORCE_PLOTS = """\
# Section 10.3 — Per-video explanations: most interesting cases

# Disagreement videos: where SBI and Xception pull the ensemble in opposite
# directions. These are the cases where the ensemble has to MAKE a decision
# rather than rubber-stamp one model.
disagreement = np.abs(shap_values[:, 0] - shap_values[:, 1])
top_disagree = np.argsort(-disagreement)[:5]

print('Top-5 disagreement videos (SBI vs Xception pull in opposite directions):')
print(f'{'idx':>4} {'true':>5} {'pred':>5} {'sbi_score':>10} {'xc_score':>10}'
      f' {'shap_sbi':>10} {'shap_xc':>10}')
for i in top_disagree:
    true = celeb_test_y[i]
    pred_prob = meta.predict_proba(test_X[i:i+1])[0, 1]
    pred = int(pred_prob >= 0.5)
    print(f'{i:>4} {true:>5} {pred:>5} '
          f'{celeb_test_sbi[i]:>10.3f} {celeb_test_xc[i]:>10.3f}'
          f' {shap_values[i,0]:>+10.3f} {shap_values[i,1]:>+10.3f}')

# Force plot for the single most-disagreement video.
worst = int(top_disagree[0])
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[worst],
    feature_names=feature_names,
    show=False,
)
plt.title(f'Per-video SHAP — Celeb-DF test idx {worst}'
          f' (true label = {"FAKE" if celeb_test_y[worst]==1 else "REAL"})')
plt.tight_layout()
waterfall_out = RESULTS_DIR / 'shap_waterfall_disagreement.png'
plt.savefig(waterfall_out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved -> {waterfall_out}')
"""

S10_DISCUSSION = """\
### Discussion — what SHAP told us

- **The bar plot (mean |SHAP|)** quantifies overall *importance*: whichever
  model has the larger bar is doing more of the heavy lifting in the ensemble.
  If they're close, both models contribute meaningfully; if one dominates, the
  ensemble is essentially that model with the other as a small correction.
- **The beeswarm** shows *direction* — high SBI/Xception scores (red) should
  produce positive SHAP values (pushing towards FAKE) and vice versa. A clean
  monotonic colour gradient confirms the meta-classifier learned a sensible
  combination.
- **The disagreement table** is the most actionable for a report: it isolates
  the videos where the two models genuinely disagree, so you can manually look
  at them and characterise the kinds of failures each model has. These rows
  are also the prediction where the *meta-classifier's choice* matters most.
"""


# ---------------------------------------------------------------------------
# Section 11 — Bootstrap CIs + paired test
# ---------------------------------------------------------------------------

S11_HEADER = """\
## Section 11 — Statistical Interpretation of the Metrics

Point estimates like "AUC = 0.95" tell you nothing about *uncertainty*. With a
test set of ~360 videos (Celeb-DF 80% slice) the difference between two models
can easily be noise. This section attaches statistical claims to the numbers.

- **Bootstrap 95% confidence intervals** for every metric: resample the test
  set with replacement 2,000 times, recompute the metric on each resample, take
  the 2.5%/97.5% percentiles. Stratified to keep both classes in every resample.
- **Paired bootstrap test** for the AUC improvement of the ensemble over the
  best single model: on each bootstrap resample we recompute both AUCs on the
  **same** resampled indices and record their difference. A two-sided p-value
  is the proportion of bootstrap samples whose sign opposes the observed
  difference (× 2).
"""

S11_CI = """\
# Section 11.1 — Bootstrap 95% CIs for every reported metric
from src.stats import bootstrap_ci, paired_bootstrap_test

# Pick the best fusion strategy by Celeb-DF AUC (from Section 9.5).
best_strategy = max(ensemble_metrics,
                    key=lambda s: ensemble_metrics[s]['celebdf']['AUC'])
print(f'Best fusion strategy on Celeb-DF: {best_strategy}')
ensemble_scores_celeb = celeb_fused[best_strategy]
ensemble_scores_ff    = ff_fused[best_strategy]

N_BOOT = 2000
configs = [
    ('FF++',     'SBI alone',          np.array(ff_y),         ff_sbi),
    ('FF++',     'Xception alone',     np.array(ff_y),         ff_xc),
    ('FF++',     f'Ensemble ({best_strategy})', np.array(ff_y), ensemble_scores_ff),
    ('Celeb-DF', 'SBI alone',          celeb_test_y,           celeb_test_sbi),
    ('Celeb-DF', 'Xception alone',     celeb_test_y,           celeb_test_xc),
    ('Celeb-DF', f'Ensemble ({best_strategy})', celeb_test_y,  ensemble_scores_celeb),
]

print()
print(f"{'Dataset':<10} {'Setting':<25} {'Metric':<10} {'Value':>9}  {'95% CI':>20}")
print('-' * 80)
ci_records = {}
for dataset, name, y, s in configs:
    for m in ('AUC', 'Accuracy', 'F1'):
        r = bootstrap_ci(y, s, metric=m, n_bootstrap=N_BOOT, seed=SEED)
        ci_records[(dataset, name, m)] = r
        print(f'{dataset:<10} {name:<25} {m:<10} {r["point"]:>9.4f}'
              f'  [{r["lo"]:.4f}, {r["hi"]:.4f}]')
    print()
"""

S11_PAIRED = """\
# Section 11.2 — Paired bootstrap test: is the ensemble's AUC gain real?

print(f'Testing: ensemble ({best_strategy})  vs  the best single model on each dataset.')
print()

def _best_single(y, s_sbi, s_xc):
    a = bootstrap_ci(y, s_sbi, 'AUC', n_bootstrap=300, seed=SEED)['point']
    b = bootstrap_ci(y, s_xc,  'AUC', n_bootstrap=300, seed=SEED)['point']
    return ('SBI', s_sbi) if a >= b else ('Xception', s_xc)

# FF++
best_name_ff, best_scores_ff = _best_single(np.array(ff_y), ff_sbi, ff_xc)
t = paired_bootstrap_test(np.array(ff_y), ensemble_scores_ff, best_scores_ff,
                          metric='AUC', n_bootstrap=N_BOOT, seed=SEED)
print(f'[FF++]     ensemble - {best_name_ff} AUC diff = {t["observed_diff"]:+.4f}'
      f'  CI [{t["ci_low"]:+.4f}, {t["ci_high"]:+.4f}]   p = {t["p_value"]:.4f}')

# Celeb-DF
best_name_c, best_scores_c = _best_single(celeb_test_y, celeb_test_sbi, celeb_test_xc)
t = paired_bootstrap_test(celeb_test_y, ensemble_scores_celeb, best_scores_c,
                          metric='AUC', n_bootstrap=N_BOOT, seed=SEED)
print(f'[Celeb-DF] ensemble - {best_name_c} AUC diff = {t["observed_diff"]:+.4f}'
      f'  CI [{t["ci_low"]:+.4f}, {t["ci_high"]:+.4f}]   p = {t["p_value"]:.4f}')

# Also test the F1 difference (more sensitive to threshold choice).
print()
t = paired_bootstrap_test(celeb_test_y, ensemble_scores_celeb, best_scores_c,
                          metric='F1', n_bootstrap=N_BOOT, seed=SEED)
print(f'[Celeb-DF] ensemble - {best_name_c} F1  diff = {t["observed_diff"]:+.4f}'
      f'  CI [{t["ci_low"]:+.4f}, {t["ci_high"]:+.4f}]   p = {t["p_value"]:.4f}')
"""

S11_PLOT = """\
# Section 11.3 — Visualize: AUC point estimates with 95% CI error bars

fig, ax = plt.subplots(figsize=(8, 4.5))

labels = ['SBI', 'Xception', f'Ensemble\n({best_strategy})']
ff_means = [ci_records[('FF++', n, 'AUC')]['point'] for n in
            ['SBI alone', 'Xception alone', f'Ensemble ({best_strategy})']]
ff_los   = [m - ci_records[('FF++', n, 'AUC')]['lo'] for n, m in zip(
            ['SBI alone', 'Xception alone', f'Ensemble ({best_strategy})'], ff_means)]
ff_his   = [ci_records[('FF++', n, 'AUC')]['hi'] - m for n, m in zip(
            ['SBI alone', 'Xception alone', f'Ensemble ({best_strategy})'], ff_means)]

c_means  = [ci_records[('Celeb-DF', n, 'AUC')]['point'] for n in
            ['SBI alone', 'Xception alone', f'Ensemble ({best_strategy})']]
c_los    = [m - ci_records[('Celeb-DF', n, 'AUC')]['lo'] for n, m in zip(
            ['SBI alone', 'Xception alone', f'Ensemble ({best_strategy})'], c_means)]
c_his    = [ci_records[('Celeb-DF', n, 'AUC')]['hi'] - m for n, m in zip(
            ['SBI alone', 'Xception alone', f'Ensemble ({best_strategy})'], c_means)]

x = np.arange(len(labels))
w = 0.36
ax.bar(x - w/2, ff_means, w, yerr=[ff_los, ff_his], capsize=4,
       color='#1f77b4', label='FF++')
ax.bar(x + w/2, c_means,  w, yerr=[c_los, c_his],   capsize=4,
       color='#ff7f0e', label='Celeb-DF v2')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.5, 1.02)
ax.set_ylabel('AUC')
ax.set_title('AUC with bootstrap 95% CIs')
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
out = RESULTS_DIR / 'auc_with_ci.png'
fig.savefig(out, dpi=150)
plt.show()
print(f'Saved -> {out}')
"""

S11_DISCUSSION = """\
### Discussion — what the statistics told us

- **Bootstrap CIs**: every metric in the report now has an honest uncertainty
  range. Narrow CIs ⇒ the estimate is stable; wide CIs ⇒ the test set is small
  enough that the headline number could easily move by several points if you
  resampled. In practice, FF++ CIs will be wider than Celeb-DF's because the
  FF++ test set is smaller (193 videos vs ~360 in the Celeb-DF 80% slice).
- **Paired bootstrap test**: the p-value answers "is the ensemble *really*
  better, or could chance explain it?". A p-value below 0.05 is the
  conventional bar for "statistically significant". For AUC, with a few
  hundred videos, even ~1 AUC-point gains can clear that bar; for F1 they
  often don't, because F1 depends on the threshold which is noisier.
- **CI plot**: visually communicates the *separation* between the bars'
  CIs — non-overlapping CIs are a strong (though informal) sign that the
  improvement is real.
"""


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Idempotency guard.
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "Section 10 — SHAP Interpretation" in src:
            print("Sections 10/11 already present; nothing to do.")
            return

    new_cells = [
        md_cell(S10_HEADER),
        code_cell(S10_FIT_AND_EXPLAIN),
        code_cell(S10_SUMMARY_AND_BEESWARM),
        code_cell(S10_FORCE_PLOTS),
        md_cell(S10_DISCUSSION),
        md_cell(S11_HEADER),
        code_cell(S11_CI),
        code_cell(S11_PAIRED),
        code_cell(S11_PLOT),
        md_cell(S11_DISCUSSION),
    ]
    nb["cells"].extend(new_cells)

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Appended {len(new_cells)} cells to {NB_PATH}")


if __name__ == "__main__":
    main()
