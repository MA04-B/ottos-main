# ÖTÖSLOTTO AI

**Intelligent Lottery Research & Prediction System** — a local, CPU-only
Streamlit application for researching the official Hungarian five-of-90 lottery
history. No paid services, AI API keys, cloud training or GPU are required.

## Scientific integrity

This is an analytics and experimentation tool, **not a reliable lottery
predictor**. For a fair independent draw, all five-number combinations have the
same probability:

```text
1 / C(90,5) = 1 / 43,949,268
Expected matches for one five-number ticket = 25/90 = 0.277777...
```

Historical frequency, being "hot", "cold" or "overdue", model ranking scores,
balanced sums, ensembles and reduced portfolio overlap do not establish a
greater chance of winning. Scores are number-level estimates, not ticket-winning
probabilities. Statistical anomalies in historical data are not evidence of
next-draw predictability.

Only chronological out-of-sample results may be used to assess models.
Multiple-testing corrections and uncertainty matter; winning the comparison
among several weak models is not proof of useful prediction. Small validation
windows are exploratory. Do not repeatedly tune on the final evaluation period:
that turns it into development data. A model selected using a historical
evaluation still needs independent prospective confirmation.

The audited engine splits an evaluation horizon into chronological development
and final-test segments. Only development metrics select a model, with Holm
correction and a default minimum of 30 development draws; final-test metrics
are diagnostic and never optimize the selection. Previously exposed historical
holdouts still require new prospective confirmation.

## Features

- Official HTML downloads with retries, timeouts, conditional HTTP requests,
  SHA-256 audited raw archives, validation and transactional SQLite updates.
- Complete drawing history, number/year/date filters, prize statistics and CSV.
- Frequencies, hot/cold windows (10, 25, 50, 100, 250 or full history),
  observed/expected counts, exact binomial tests with Holm correction.
- Gaps and completed-gap distributions, all 4,005 pairs (including zero counts),
  frequent triples, odd/even, low/high, sums, spreads, consecutive pairs, decades
  and historical rolling frequencies.
- CPU model laboratory with uniform, historical frequency, exponential
  frequency, Bayesian smoothing, logistic regression, Random Forest, Extra
  Trees, Gradient Boosting and HistGradientBoosting. Optional XGBoost,
  LightGBM and CatBoost are discovered only if installed.
- Prior-only per-number features and chronological walk-forward testing,
  baseline comparisons, match distributions, ranking/precision/recall,
  log loss, Brier score, calibration, period results and confidence intervals.
- Background training with progress, persistent experiments, local joblib
  artifacts and data-version checks.
- Exactly ten distinct, sorted five-number candidates; reproducible seeds,
  ranked/weighted/historical/recent/Bayesian/balanced/ensemble/diversified
  generation, number explanations and configurable portfolio diversity.
- Prediction persistence, duplicate-safe saving, CSV export and clearly labeled
  in-sample historical comparisons.
- Eight dark-dashboard pages and persistent local settings.

## Official data and missing dates

Authoritative source:
<https://bet.szerencsejatek.hu/cmsfiles/otos.html>

The official table omits drawing dates for many older records. Those records
are retained with **NULL dates and their official year/week**, not guessed
calendar dates. Their winning numbers are real official results. The database
enforces unique year/week and unique non-null dates; the data-quality report
shows all missing dates. Date-range filtering explicitly excludes undated rows.
Models and charts use chronological year/week order, including undated draws.
Year-end week 52/53 draws may have an official calendar date in early January
of the following year; this legitimate rollover is preserved.

Invalid number sets are quarantined in the quality report, not fabricated.
Conflicting duplicates, unrecognizable tables and substantially truncated
responses abort ingestion. Network failure never deletes saved history.
"Rebuild / reconcile" reimports and validates the full source without deleting
valid saved records. Unknown future drawing dates remain unset in predictions.

## Requirements and shared environment

- Miniconda, Anaconda or a compatible Conda installation.
- Python **3.11**, provided by the dedicated **`otoslotto-ai`** environment.
- Normal Windows or Linux desktop CPU; allow several GB of free disk space.
- Internet for initial installation and synchronization; cached history and
  existing artifacts work offline.

`environment.yml` is the **single dependency definition** for both operating
systems. The scripts do not use global Python, `venv` or `virtualenv`.
Optional boosting packages are not required and are not installed automatically.
Install them explicitly into this Conda environment only if needed.

### Linux

1. Install Miniconda from
   <https://docs.conda.io/projects/miniconda/>. Follow its Linux installer
   instructions, run `conda init bash` and restart the terminal.
2. From the project directory:

   ```bash
   ./install_linux.sh
   ./start_linux.sh
   ```

3. Open <http://127.0.0.1:8501>. Initial session startup checks the official
   source if automatic synchronization is enabled.
4. Update packages and synchronize manually:

   ```bash
   ./update_linux.sh
   ```

5. Run tests and lint:

   ```bash
   conda run -n otoslotto-ai pytest -q
   conda run -n otoslotto-ai ruff check .
   ```

The scripts locate their own directory, so they can also be run using absolute
paths from another working directory. Installation updates an existing
environment from the YAML, including changed requirements.

### Windows

1. Install Miniconda from
   <https://docs.conda.io/projects/miniconda/>. Anaconda is also supported.
2. Open **Miniconda Prompt** or **Anaconda Prompt**, change to this directory,
   and run:

   ```bat
   install_windows.bat
   start_windows.bat
   ```

   Double-clicking the scripts is supported for common installation locations.
   For a custom Conda installation, use its initialized prompt.

3. Open <http://127.0.0.1:8501>. Keep the startup terminal open.
4. To update:

   ```bat
   update_windows.bat
   ```

5. Tests and lint:

   ```bat
   conda run -n otoslotto-ai pytest -q
   conda run -n otoslotto-ai ruff check .
   ```

All Windows scripts use their own directory, quoted paths and `call` for Conda
batch activation. They pause after completion/errors so diagnostics remain
visible. Windows execution must be checked on an actual Windows machine.

### Equivalent manual Conda commands

```bash
conda env create -f environment.yml
conda activate otoslotto-ai
python -m src.cli init
python -m src.cli verify
python -m src.cli sync
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

For an existing environment:

```bash
conda env update -n otoslotto-ai -f environment.yml --prune
```

## Using the application

1. **Data Management:** synchronize the official source and inspect quality
   warnings. Review raw archives and logs if parsing fails.
2. **Overview / History:** inspect the latest draw, history and source prizes.
3. **Statistical Analysis:** select a window and inspect descriptive charts.
4. **AI Model Laboratory:** select models, a validation horizon and classifier
   training window. Click training/backtesting. A background worker keeps
   navigation responsive. Smaller windows provide a CPU-friendly starting
   point; `0` training-window means full classifier history.
5. **Backtesting:** inspect metrics, match histograms, calibration, periods and
   the uniform baseline. The best-supported selection remains uniform when no
   alternative is sufficiently supported. Persisted history is exportable.
6. **Prediction Center:** choose a strategy, seed and diversity, then generate
   exactly ten candidates. Expand explanations to inspect number scores,
   historical counts and combination characteristics. Generation saves the
   batch automatically; the separate Save button is idempotent.
   "Generate again" advances the displayed seed deterministically.
   **🪄 Magic Generate**, next to Generate Again, runs a cancellable background
   search using the same generator and current strategy/model settings. An
   attempt generates ten tickets with a new seed; the default limit is 1,000
   attempts and the configurable maximum is 10,000. Live progress shows attempts
   and tickets checked. **Stop search** cancels after the current generator call
   finishes. Other pages and CSV downloads remain usable.

   An exact five-of-five historical repeat stops the search and displays the
   matching ticket within its ten-ticket portfolio, the attempt count and the
   official historical draw date/numbers. Undated official records show their
   year/week without inventing dates. The successful portfolio is saved; failure,
   cancellation or exhaustion preserves the previously displayed portfolio.
   Conflicting generation buttons are disabled during a search. A search is
   isolated to its browser session and remains bounded if you navigate away.

   **This intentionally searches known history and is not a prediction,
   backtest or guarantee of future wins.** Exact repeats can be rare, and some
   constrained strategies may never reach any historical ticket. The limit
   controls attempts, not wall-clock time: a large search can take minutes
   because each call uses the existing engine, including its historical
   analysis. Closing a browser does not immediately cancel its worker; use Stop
   search first, or the worker will end at its limit.
7. **Settings:** persist defaults, update interval and model preferences.
   Automatic selection or uniform always works; an explicitly chosen learned
   model needs an appropriate saved artifact.

Historical candidate comparisons use the same data that informed generation
and are explicitly **not** a leakage-free backtest. Never use them as evidence
of predictive performance.

### Initial Linux delivery (superseded by the audit below)

The live official source imported **3,634 draws**, from 1957 week 10 through
2026 week 43 (latest dated draw: 2026-10-24). All winning-number rows validated;
2,487 older rows have officially missing dates, reported as warnings.

A saved CPU experiment evaluated the nine required models plus a reproducible
random-ticket comparator on the last 20 draws. The logistic model averaged
0.50 matches, but its Holm-corrected p-value was approximately 0.379; it did
**not** establish significant improvement. Selection correctly remained
**uniform** (expected matches 0.27778). This short exploratory experiment is
not evidence of reliable prediction.

XGBoost, LightGBM and CatBoost are optional and not installed in the default
environment. Windows scripts are complete but have not been executed on Windows.

Final verification: **87 automated tests passed**, three optional-library tests
were skipped, and Ruff passed. Linux installation/update/startup scripts ran
successfully. All eight pages were exercised in the local browser; background
training completed, ten cards containing fifty valid numbers were generated,
and the saved portfolio reloaded successfully. The loopback health endpoint
returned `ok`.

### Independent audit and corrections

See [AUDIT_REPORT.md](./AUDIT_REPORT.md) for the detailed findings, corrections,
source reconciliation, model results and limitations.

Final audit: **141 tests passed**, three optional-library tests skipped, and
Ruff passed. An independent direct-HTML audit reconciled all **3,634** official
rows with SQLite, including prize fields; no missing source rows, duplicates,
invalid numbers or field mismatches were found.

The corrected 100-draw walk-forward experiment used **50 development and 50
final-test draws**, trained all nine required models, and selected **uniform**.
No model met corrected development-evidence requirements.

The audit also fixed full-history training (`0`), probability mass constraints,
target-date leakage, confidence-interval interpretation, source completeness
checks, corrupt-model reporting, balanced generation, batch persistence,
Windows Conda error handling and dashboard bottlenecks.

Re-run the independent source check:

```bash
conda run -n otoslotto-ai python -m src.cli audit
```

Candidate CSVs now have five numeric winning-number columns. Their contents
are round-trip tested; actual file delivery could not be observed through the
available embedded browser. Windows scripts were statically reviewed and
tested for their contract but still require execution on Windows.

## Architecture and persistence

```text
app.py                   Eight-page Streamlit dashboard (st.navigation)
config/settings.py       Project paths and authoritative URL
src/data/                Scraper, schema-aware parser, validation, SQLite
src/analysis/            Vectorized descriptive statistics
src/features/            Chronological prior-only number features
src/ml/                  Registry, estimators, backtesting, selection, artifacts
src/prediction/          Scored generation and portfolio diversification
src/jobs.py              Single CPU training worker, progress, error logging
src/preferences.py       Atomic local JSON preferences
src/cli.py               Init / dependency verification / synchronization
tests/                   Mocked HTTP, parsing, database, models and UI tests
environment.yml          Shared Conda dependency definition
```

SQLite uses versioned initialization (`PRAGMA user_version=1`), WAL mode,
constraints, chronological indexes and transactions. Tables contain draws
(including original prizes), experiment metrics/artifact locations, prediction
batches with number scores/method/seed/backtest metadata, quality reports,
synchronization state and system events.

Local runtime files are excluded from Git:

- `data/lottery.db` and SQLite sidecars.
- `data/raw/` audited official HTML.
- `data/settings.json` local defaults.
- `data/exports/` local exports.
- `models/trained/` trusted local joblib artifacts and metadata.
- `logs/application.log`.

Do not load joblib artifacts from untrusted sources: pickle-based model files
can execute code. This application is bound to loopback for local use and has
no public hosting/authentication layer. Do not expose it to the internet.
Back up the database and model directories before changing machines.

`OTOSLOTTO_DATA_DIR` optionally changes the data directory. Set it in your
shell before launching. `.env.example` documents the variable; the application
does not automatically read `.env`.

## Automatic updates and offline operation

Each new browser session checks the stored update interval. A recent attempt
suppresses repeated downloads, including repeated failed requests; manual
update bypasses the interval. HTTP ETag/Last-Modified headers allow 304
unchanged responses. Complete HTML is downloaded only when required by the
server. Automatic checks happen on application session startup, not via a
permanent OS scheduler while the application is closed.

New data invalidate dataset-specific model metadata. Retrain deliberately;
the app does not train automatically or treat stale scores as current evidence.
An offline database remains readable; uniform/statistical generation remains
available even without a learned artifact.

## Testing and troubleshooting

- **Conda missing:** install Miniconda, restart the terminal or use its prompt.
- **Dependencies missing:** rerun installation; all dependencies must be inside
  `otoslotto-ai`, not your global interpreter or another editor environment.
- **Official website unavailable:** existing records are preserved. Inspect the
  error banner/system events and retry with Data Management or the update
  script. Unit tests mock HTTP and do not need the website.
- **Website changed:** parsing refuses an unrecognized schema. Inspect the
  archived HTML and adjust parser recognition; do not guess winning columns.
- **Undated historical rows:** these are official omissions, not an ingestion
  failure. Use year/week filters.
- **Training slow:** start with frequency baselines, 20 evaluation draws and
  a 250-draw classifier window. Large forests/boosting and full-history
  walk-forward fits can take much longer on CPU.
- **No model beats random:** this is an expected and scientifically valid
  result, not an application failure.
- **Invalid model/settings file:** review the explicit error, repair/remove
  the affected local file, then retrain/save settings. Preserve the database.
- **Port 8501 occupied:** stop the specific process using it or choose a
  different port manually; do not kill unrelated processes.
- **Stopping:** press Ctrl+C in the startup terminal.

The automated suite includes deterministic synthetic **test fixtures only**;
production history is populated exclusively from the official source.
Optional dependencies and Windows execution are reported separately from
verified core Linux workflows.
