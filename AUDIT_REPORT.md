# Independent implementation audit — ÖTÖSLOTTO AI

Date: 2026-10-31. Scope: official-source ingestion, SQLite, every implemented
feature/model/evaluation path, statistical calculations, all eight generation
strategies plus mixed mode, all eight dashboard pages, persistence, Conda
installation and CPU performance.

## Outcome

The original passing suite was insufficient: **verified defects were found and
corrected directly in the application**. The final suite reports **141 passed,
3 skipped**, with no failures. Ruff passes. SQLite `integrity_check` returns
`ok`. All eight pages render without application exceptions.

The corrected nine-model experiment still selects **uniform**. There is no
statistically supported development-period improvement and no claim that the
application can reliably predict lottery results.

## 1. Official-source reconciliation

A fresh full response was downloaded from
<https://bet.szerencsejatek.hu/cmsfiles/otos.html>, rather than relying only on
HTTP 304 responses or a previous import.

An additional independent BeautifulSoup audit reads the actual official
16-cell row contract directly. It does **not** call the production pandas
parser. Every source drawing is reconciled against SQLite by official
year/week, all five numbers, actual date/null date, all winner counts and HUF
prize amounts.

| Check | Actual result |
|---|---:|
| Official data rows | 3,634 |
| Unique official drawings | 3,634 |
| SQLite drawings | 3,634 |
| Source drawings missing from SQLite | 0 |
| SQLite-only drawings | 0 |
| Number/date/prize field mismatches | 0 |
| Invalid official winning-number rows | 0 |
| Invalid SQLite winning-number rows | 0 |
| Duplicate official year/week or date | 0 |
| Duplicate SQLite year/week or known date | 0 |
| Internal year/week gaps | 0 |
| Incomplete interior source years | 0 |
| Malformed source prize fields | 0 |
| Officially omitted dates | 2,487 |
| Known dates | 1,147 |
| First drawing | 1957 week 10 |
| Last drawing | 2026 week 43, 2026-10-24 |

Final audited UTF-8 snapshot SHA-256 (all fields still reconcile):

```text
e0888b313a2dea5ab0386959f4474129c6f977a977b0ef704a38456c382a65da
```

Full interior years have consecutive weeks starting at 1 and ending at 52 or
53. The first and latest years are intentionally partial. This establishes
completeness **relative to the authoritative source supplied**, not a guarantee
that the publisher itself has never omitted a historical drawing.

Missing dates remain NULL and are shown as quality warnings. No dates or
winning results were fabricated. Legitimate week-52/53 January year rollovers
are retained.

Reproduce the independent offline reconciliation:

```bash
conda run -n otoslotto-ai python -m src.cli audit
```

To audit another saved official response:

```bash
conda run -n otoslotto-ai python -m src.cli audit --html /path/to/official.html
```

An audit mismatch produces a nonzero exit status.

## 2. Verified findings and implemented corrections

| Area | Finding | Correction |
|---|---|---|
| Research integrity | The same evaluation period selected models and was presented as final evidence. | Chronological development/final-test separation. Selection and tuning never consume final-test metrics. Results distinguish both segments. |
| Feature leakage | Target dates and elapsed time were read from retrospectively published drawing metadata. | Time features use only the previous known schedule. Prefix-only and target-metadata mutation tests cover every feature. |
| Probability mathematics | Clipping after normalization could leave probabilities summing to approximately **1.000088**, not 5. | Capped proportional allocation preserves exactly five units of marginal probability, including concentrated vectors. Invalid scores raise explicit errors. |
| Training controls | `training_window=0`, advertised as full history, raised `ValueError`. | Zero now means all usable preceding training rows after warmup. Verified in tests and the browser training workflow. |
| Gap convention | ML gaps were one draw larger than descriptive missed-draw gaps. | Both now count missed draws since the last appearance. |
| Baseline mathematics | Random ticket ranking jitter also perturbed marginal probabilities. | Exact 5/90 marginals are separate from seeded random rankings. Brier/log loss use the exact uniform probabilities. |
| Confidence intervals | Wilson intervals treated the five within-draw hit indicators as independent Bernoulli trials. | Hit-rate intervals use draw-level bootstrap intervals divided by five. Small-sample limitations and null prediction bands are labeled. |
| Significance controls | Critical match thresholds always used alpha 0.05, even with another configured alpha. | Thresholds use the configured alpha, with a dedicated regression test. |
| Source completeness | The scraper tolerated losing up to 5% of already saved rows while reporting synchronization. | Any absent saved year/week aborts reconciliation and preserves existing data. |
| Conditional updates | Local deletion/correction could be overlooked because the source ETag was unchanged. | Database-content fingerprints bypass stale intervals/conditional headers after a local change. Failed requests remain throttled and visibly reported. |
| Duplicate validation | Same numbers with conflicting dates/keys could be treated as an exact duplicate. | Parser, SQLite reconciliation and feature-frame validation reject conflicting identities. |
| Prize metadata | A malformed prize cell could discard an otherwise valid winning-number row. | Preserve the real draw, record the missing prize as NULL, and surface a quality warning. |
| Balanced generation | Rejection sampling could return a ticket violating the advertised balance constraints. | Validated uniform proposals recover from concentrated model weights; inability to meet constraints is an explicit failure. |
| Ranked sampling | Smoothing could introduce positive sampling mass outside the restricted ranked pool. | Pool masks remain exactly zero outside the selected pool. |
| Input validation | Out-of-range diversity was silently clamped. | Invalid diversity, non-integer seeds/counts and invalid number inputs raise useful errors. |
| Model persistence | Corrupt/incompatible artifacts and some callback warnings were silently ignored. | Loader reports errors, exposes fallback warnings, and invalidates prior feature/engine versions. Training warnings are retained; callback failures are logged. |
| Selection failure | Final-fit failure could leave the original selection explanation despite falling back to uniform. | Selection metadata and warnings explicitly describe the fallback. |
| Prediction persistence | Partial portfolios were accepted; conflicting reuse of a batch identifier was silently ignored. | Require ten unique ascending tickets. Identical saves are idempotent; changed batch contents fail transactionally. |
| UI persistence | Re-saving a portfolio after retraining could attach different backtest metadata. | Preserve the original batch's backtest metadata across save and reload. |
| UI interpretation | Worse-than-baseline detection compared a lift ratio against zero. | Comparisons use separate development evidence and final-test match estimates, with explicit provisional/insufficient labels. |
| CSV output | Candidate CSV used nested list cells rather than five numeric number columns. | Flat, round-trip-tested columns `number_1`–`number_5`, per-number scores and ticket metadata. Downloads avoid needless reruns. |
| Concurrency | Training-start checks and fixed settings temporary filenames were race-prone. | Lock the training start and use unique atomic settings temporary files. |
| Windows scripts | Piping `conda env list` into `findstr` could hide Conda enumeration failure. | Check Conda's own exit status before environment detection; quote and clean up a unique temporary listing. |
| Dependency compatibility | The Streamlit lower bound did not guarantee the newer APIs used by the dashboard. | Shared environment requires Streamlit >=1.50. Current installed version satisfies it. |
| Performance | Overview/prediction pages loaded the entire prediction archive; backtesting rendered every model's charts even in closed expanders. | SQL count/latest-batch queries, bounded log queries, bounded caches and one selected model's detailed charts. |

Older experiments and predictions were preserved for auditability. Old
feature/engine artifacts are marked outdated rather than treated as current
validated models.

## 3. Feature-by-feature leakage audit

For target index `t`, all label-dependent calculations use draws `[0,t)`.
The final feature tensor row is also reproducible from the prefix ending
immediately before the target.

| Feature group | Verified source of information |
|---|---|
| Number identity, parity and decade | Constants for the candidate number |
| All-history frequency/count/history | Cumulative labels strictly before `t` |
| Rolling 10/25/50 frequencies | Prior-only window ending at `t-1` |
| Occurrence gap | Last appearance strictly before `t` |
| EW fast/mid/slow | Sequential updates applied only after each prior draw |
| Previous and second-previous draw | Labels at `t-1` and `t-2` |
| Pair score | Prior pair counts and prior marginal counts, excluding self-pairs |
| Calendar phase | Next-week schedule assumption derived from the previous official year/week |
| Days/schedule spacing | Constant weekly assumption, not the retrospectively observed target interval |
| Log history | Number of preceding draws |

Regression tests independently reconstruct frequencies, EW recurrences, gaps
and conditional pair scores. Future-label changes leave all preceding features
unchanged. Removing target-date metadata also leaves the target feature row
unchanged.

The weekly calendar assumption is deliberately simple and may be wrong around
holiday rescheduling or legacy week conventions. It is not represented as an
actual historical date or predictive evidence.

## 4. Walk-forward validation and model selection

- Training ends strictly before each evaluation block.
- Classifier fitting is chronological and bounded to 250 prior draws by
  default; zero means full usable prior history.
- StandardScaler is fitted inside each training-only logistic pipeline.
- EW half-life and Bayesian prior strength are chosen on training rows only.
- Models refit at fixed chronological intervals. Within a block, predictions
  may use newly observed **past** draws as features; no future label is used.
- At least 30 development draws are required by default for a nonuniform
  selection.
- Candidate development p-values receive Holm correction.
- The final-test suffix is diagnostic only: it does not choose another model,
  optimize parameters or determine the development ranking.
- Subsequent fitting on all available history is for **next-draw generation**,
  not for retroactively scoring the holdout.

A regression test changes every final-test winning label while holding the
development segment fixed. Development metrics, adjusted p-values, eligibility
and selected model remain identical, while final-test metrics change.

### Actual corrected live experiment

Parameters fixed for this run: seed 42, 100 chronological evaluation draws,
50 development / 50 final-test draws, 250-draw classifier window, four refit
blocks, default model parameters, alpha 0.05, 1,000 bootstrap replicates.

- Development: **2024-W48 through 2025-W45**, indices 3534–3583.
- Final test: **2025-W46 through 2026-W43**, indices 3584–3633.

| Model | Development average matches | Development Holm p | Final-test average matches |
|---|---:|---:|---:|
| Uniform analytical baseline | 0.27778 | Not applicable | 0.27778 |
| Seeded random-ticket comparator | 0.36 | Baseline, not a candidate | 0.26 |
| Historical frequency | 0.38 | 0.69934 | 0.22 |
| Exponential frequency | 0.40 | 0.49526 | 0.26 |
| Bayesian | 0.38 | 0.69934 | 0.22 |
| Logistic regression | 0.30 | 1.00000 | 0.32 |
| Random Forest | 0.28 | 1.00000 | 0.20 |
| Extra Trees | 0.22 | 1.00000 | 0.18 |
| Gradient Boosting | 0.22 | 1.00000 | 0.28 |
| HistGradientBoosting | 0.30 | 1.00000 | 0.20 |

**Selected model: uniform.** No candidate satisfied corrected development
evidence requirements. A point estimate above 0.27778 is not by itself evidence
of improvement.

All nine required models trained successfully, alongside the random-ticket
control. The audited artifact and metrics were saved locally and reloaded with
`stale=False`.

The final historical period has been exposed during earlier development.
Splitting it correctly now does **not** make it a pristine independent
confirmatory dataset. Genuine confirmation requires new, prospectively
registered drawings.

## 5. Mathematical checks

- Ticket jackpot probability: `1 / C(90,5) = 1 / 43,949,268`.
- Expected ticket overlap: `5 × 5/90 = 25/90`.
- Match distribution: Hypergeometric `(population=90, successes=5, sample=5)`.
- Exact total-match null: convolution of independent per-draw hypergeometric
  PMFs; compared with SciPy and checked for unit mass and expected mean.
- Uniform number-level Brier score: `(5/90)(85/90)`.
- Uniform binary log loss: `-p log(p) - (1-p) log(1-p)`, `p=5/90`.
- Precision/recall at five: matches divided by five.
- Pair expectation: `n × 5×4/(90×89)`; all 4,005 pairs are included.
- Each valid draw contributes ten pair and ten triple occurrences.
- Number frequency tests: exact two-sided binomial tests, Holm-adjusted over
  90 numbers **within the chosen window**.
- Gap distributions use completed missed-draw intervals, excluding censored
  start/end intervals from completed-gap averages.
- Sums, odd/even, low/high, spreads, consecutive pairs and decade counts were
  independently reconstructed from actual number arrays.

Frequency testing does not correct a user's repeated search across many
windows. Pair/triple rankings remain descriptive, with no significance or
predictive advantage asserted. Bootstrap intervals are explicitly descriptive
and can be unstable for small or constant samples; uniform bands are null
prediction bands rather than estimated model-skill intervals.

## 6. Generation, dashboard and persistence testing

All eight named strategies were tested, plus mixed mode:

1. ranked;
2. weighted;
3. historical;
4. recent;
5. Bayesian;
6. balanced;
7. ensemble;
8. diversified.

Automated coverage includes diversity 0, 0.5 and 1, reproducible seeds, normal
and concentrated/zero-heavy score vectors, sorted distinct numbers in 1–90,
ten unique tickets and verified balance constraints. Number explanations now
include historical appearance counts and strategy-specific reasons.

In the live browser every strategy produced **10 unique valid tickets**, each
containing five numbers, with no application exceptions. Save/reload and
duplicate-safe save workflows were exercised.

All eight page callbacks also have automated Streamlit AppTest coverage using
real experiment metrics. All eight live routes were visited successfully.
History number/date filtering, manual synchronization, settings save and
background training with the full-history setting were exercised.

CSV content is round-trip tested: ten rows and five numeric number columns.
The browser download button produced no application exception, but the
available embedded browser did not expose a download event. **Actual browser
file-download delivery remains unverified in this environment.**

SQLite checks found no partial prediction batches. Corrupt artifact tests,
staleness/version checks, interrupted/failed imports, conflicting duplicates,
original batch metadata and concurrent settings writes are covered.

## 7. Installation and Windows review

Linux installer and updater were executed successfully from `/tmp`, not only
from the repository. The startup script was also executed from an unrelated
directory and binds to `127.0.0.1:8501`; the health endpoint returned `ok`.
All Linux scripts passed `bash -n` and have executable permissions.

Windows static review covered:

- `%~dp0` repository resolution and quoted paths;
- Miniconda/Anaconda activation via `call`;
- shared environment YAML and environment name;
- Conda failure handling before environment detection;
- environment creation/update, dependency verification and SQLite init;
- loopback address and port;
- error exits and terminal pauses.

Static contract tests pass. **Windows execution is not available on this Linux
machine and remains unverified.** There is no claim that Windows installation
was actually run.

Installed verification environment: Python 3.11 Conda environment
`otoslotto-ai`; Streamlit 1.66.1, scikit-learn 1.9.1, NumPy 2.4.6, pandas 2.3.3.
The shared YAML remains the only dependency definition.

## 8. Measured CPU performance

Measured on this Linux machine with all 3,634 drawings:

| Operation | Elapsed |
|---|---:|
| Build all prior-only features | 0.223 s |
| Compute complete descriptive statistics | 0.194 s |
| Nine-model + random-control 100-draw experiment, final fits and persistence | 13.457 s |
| Reload latest compatible artifact | 0.162 s |
| Generate ten combinations | 0.145 s |

Walk-forward fit costs were approximately 3.81 s for Random Forest, 3.70 s for
Gradient Boosting, 1.55 s for Extra Trees and 1.43 s for HistGradientBoosting.
These are the dominant CPU bottlenecks. Full-history classifier fitting is
supported but can be substantially slower; this report does not invent a
full-history nine-model benchmark.

Feature tensor storage is about 50 MiB for the current history. The shared
classifier sees 90 rows per training draw, so training-window limits matter.
Training remains in a background worker with progress and explicit errors.

Applied performance improvements: bounded statistics/model caches, cached
compatible artifact loading, SQL count/latest-batch/limited-log queries,
cached exact null distributions and single-model detailed chart rendering.
Artifacts, raw snapshots and experiment/prediction history still accumulate
on disk intentionally; users should manage/backup those archives.

## 9. Final automated results and limitations

```text
conda run -n otoslotto-ai python -m pytest -q
141 passed, 3 skipped

conda run -n otoslotto-ai ruff check .
All checks passed!

SQLite PRAGMA integrity_check
ok
```

The three skips are optional XGBoost, LightGBM and CatBoost training tests:
their packages are not installed. Core models all execute.

Remaining limitations:

1. No evidence of reliable prediction for independent fair lottery draws.
2. Historical final periods already inspected during development are not
   pristine confirmatory evidence; repeated runs/configuration searches are
   not globally corrected. Use prospectively registered evaluations.
3. Official older dates are unavailable and remain NULL.
4. Windows execution and embedded-browser file-download delivery are unverified.
5. Optional boosting libraries were not installed/tested here.
6. Bootstrap/calibration estimates have limited power with short evaluation
   periods; balance/diversity are heuristics, not better lottery odds.
7. Automatic checks run on browser-session startup, not while the application
   is closed. There is no permanent OS scheduler.
8. Local loopback use only; trusted local joblib artifacts only. This audit
   is not a penetration test or certification for public hosting.

The application was left running locally at <http://127.0.0.1:8501>.
