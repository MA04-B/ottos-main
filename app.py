import html
import json
import logging
import math
import sqlite3
import uuid
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from config.settings import MODELS, ROOT, SOURCE, initialize_directories
from src.analysis.statistics import NUMBER_COLUMNS, analyze
from src.data.database import Database
from src.data.scraper import sync_database
from src.jobs import TrainingJob
from src.ml import available_models
from src.ml.training import load_latest
from src.prediction.generator import combinations_frame, generate_combinations
from src.preferences import read_preferences, save_preferences

st.set_page_config(page_title="ÖTÖSLOTTO AI", page_icon="🎱", layout="wide")
initialize_directories()
logging.basicConfig(filename=ROOT / "logs" / "application.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
st.markdown("""
<style>
.block-container {padding-top:2rem; max-width:1500px}
h1,h2,h3 {letter-spacing:-.04em}
.hero {background:linear-gradient(120deg,#15324d,#12283a,#12372f);padding:28px;
 border:1px solid #294559;border-radius:20px;margin-bottom:20px}
.eyebrow {color:#38d9a9;font-size:12px;letter-spacing:.2em}
.ball {display:inline-flex;align-items:center;justify-content:center;width:52px;height:52px;
 border-radius:50%;margin:5px;background:linear-gradient(145deg,#43e5b4,#16957d);
 color:#071c19;font-size:22px;font-weight:800;box-shadow:0 4px 18px #38d9a922}
.card {padding:20px;background:#111d30;border:1px solid #294155;border-radius:16px;margin:10px 0}
div[data-testid="stMetric"] {background:#111d30;padding:18px;border-radius:14px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def database():
    return Database()


@st.cache_resource
def training_job():
    return TrainingJob()


@st.cache_data(show_spinner=False, max_entries=6)
def statistics(frame):
    return analyze(frame)


@st.cache_resource(max_entries=4)
def cached_artifact(frame, signature):
    return load_latest(frame)


db = database()
try:
    preferences = read_preferences()
except (OSError, ValueError) as exc:
    st.error(f"Settings file is unreadable: {exc}. Repair or remove the settings file.")
    st.stop()

if "startup_sync" not in st.session_state:
    st.session_state.startup_sync = True
    if preferences["auto_update"]:
        try:
            with st.spinner("Checking official historical database..."):
                sync_database(db, interval_hours=preferences["interval_hours"])
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(f"Synchronization failed; existing data preserved: {exc}")
        except requests.RequestException as exc:
            st.error(f"Official source unavailable; existing data preserved: {exc}")

frame = db.draws()
state = db.state()
if state.get("last_error"):
    st.warning(f"Last synchronization error: {state['last_error']}")


def require_data():
    if frame.empty:
        st.info("No historical data yet. Open Data Management and synchronize the official database.")
        st.stop()


def artifact():
    try:
        signature = tuple((p.name, p.stat().st_mtime_ns) for p in sorted(MODELS.glob("experiment_*.json")))
        current = cached_artifact(frame, signature) if not frame.empty else None
        if current:
            for warning in current.get("load_warnings", []):
                st.warning(warning)
        if current and current.get("stale"):
            st.warning(f"Saved models are outdated: {current.get('stale_reason')}. "
                       "Retrain to use current scores. Generation uses uniform until retraining.")
        return current
    except (OSError, ValueError, EOFError) as exc:
        st.error(f"Saved model cannot be loaded: {exc}. Retrain in the laboratory.")
        return None


def balls(numbers):
    return "".join(f'<span class="ball">{int(n):02d}</span>' for n in numbers)


def download(data, name):
    st.download_button("Download CSV", data.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv", on_click="ignore")


def disclaimer():
    st.caption(f"Research, not a reliable lottery predictor. Every fair five-number combination has "
               f"probability 1 / {math.comb(90, 5):,}. Historical frequencies, scores and portfolio "
               "diversity do not increase an individual combination's winning probability.")


def overview():
    st.markdown('<div class="hero"><div class="eyebrow">LOCAL • CPU • SCIENTIFICALLY HONEST</div>'
                '<h1>ÖTÖSLOTTO AI</h1><p>Intelligent Lottery Research & Prediction System</p></div>',
                unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("Historical drawings", f"{len(frame):,}")
    columns[1].metric("Latest known date", frame.draw_date.dropna().max() if not frame.empty else "—")
    columns[2].metric("Saved combinations", db.prediction_count())
    columns[3].metric("Training status", training_job().message)
    st.caption(f"Last successful update: {state.get('last_success', 'Never')}")
    require_data()
    latest = frame.iloc[-1]
    st.subheader(f"Latest official draw · {latest.year} / week {latest.week}")
    st.markdown(balls(latest[NUMBER_COLUMNS]), unsafe_allow_html=True)
    current = artifact()
    st.info(f"Model selection (provisional, not a lottery predictor): "
            f"{current['selected_model'] if current and not current.get('stale') else 'Uniform baseline'}")
    if current:
        st.caption(current.get("selection_reason", ""))
    stats = statistics(frame)
    st.plotly_chart(px.bar(stats["frequency"], x="number", y="count",
                           title="Complete-history number frequency", color="difference",
                           color_continuous_scale="Teal"), width="stretch")
    st.caption(f"{frame.draw_date.isna().sum():,} official records omit drawing dates. "
               "They are retained in year/week order; no dates have been invented.")


def history():
    st.title("Historical Results")
    require_data()
    years = st.multiselect("Years", sorted(frame.year.unique(), reverse=True))
    filtered = frame[frame.year.isin(years)] if years else frame
    number = st.number_input("Contains number (0 = all)", 0, 90, 0)
    if number:
        filtered = filtered[filtered[NUMBER_COLUMNS].eq(number).any(axis=1)]
    use_dates = st.checkbox("Filter by known drawing date (excludes undated official records)")
    if use_dates:
        dates = st.date_input("Date range", (date(2005, 1, 1), date.today()))
        if isinstance(dates, tuple) and len(dates) == 2:
            filtered = filtered[filtered.draw_date.fillna("").between(str(dates[0]), str(dates[1]))]
    st.caption(f"{len(filtered):,} results · prizes stored as official HUF amounts and winner counts")
    st.dataframe(filtered.drop(columns=["id"]), hide_index=True, width="stretch")
    download(filtered, "lottery_history.csv")


def analysis_page():
    st.title("Statistical Analysis")
    require_data()
    windows = [0, 10, 25, 50, 100, 250]
    window = st.selectbox("Historical window", windows, index=windows.index(preferences["analysis_window"]),
                          format_func=lambda x: "Entire history" if x == 0 else f"Last {x} draws")
    stats = statistics(frame.tail(window) if window else frame)
    frequency, gap, pair, distributions, trends = st.tabs(
        ["Frequency / hot & cold", "Occurrence gaps", "Pairs & triples", "Distributions", "Trends"])
    with frequency:
        st.plotly_chart(px.bar(stats["frequency"], x="number", y="count", color="difference",
                              title="Observed versus uniform expected frequency"), width="stretch")
        a, b = st.columns(2)
        a.subheader("Historically hot")
        a.dataframe(stats["frequency"].nlargest(10, "count"), hide_index=True)
        b.subheader("Historically cold")
        b.dataframe(stats["frequency"].nsmallest(10, "count"), hide_index=True)
        st.caption("Exact two-sided binomial tests use Holm correction across 90 numbers. "
                   "This controls only this selected window, not repeated searches across windows. "
                   "Significant historical nonuniformity is not proof of next-draw predictability.")
        st.dataframe(stats["frequency"], hide_index=True)
    with gap:
        st.caption("Gaps count missed draws between appearances. Completed intervals exclude censored "
                   "start/end intervals. Being overdue does not make a number due.")
        st.dataframe(stats["gaps"], hide_index=True, width="stretch")
        selected = st.number_input("Gap distribution for number", 1, 90, 1)
        intervals = stats["gaps"].iloc[selected - 1].completed_gap_distribution
        if intervals:
            st.plotly_chart(px.histogram(x=intervals, labels={"x": "Missed draws"}), width="stretch")
    with pair:
        a, b = st.columns(2)
        a.subheader("Most common pairs")
        a.dataframe(stats["pairs"].nlargest(25, "count"), hide_index=True)
        b.subheader("Least common pairs (including zero)")
        b.dataframe(stats["pairs"].nsmallest(25, "count"), hide_index=True)
        st.subheader("Most common triples")
        st.dataframe(stats["triples"].head(40), hide_index=True)
        st.caption("Pair/triple rankings are descriptive. No significance or predictive advantage is claimed.")
    with distributions:
        for key in ("odd", "low_1_45", "sum", "spread", "consecutive_pairs"):
            st.plotly_chart(px.histogram(stats["distribution"], x=key, title=key.replace("_", " ").title()),
                            width="stretch")
        st.plotly_chart(px.bar(stats["decades"], x="band", y="count", title="Number bands"), width="stretch")
    with trends:
        selected = st.multiselect("Numbers", range(1, 91), default=[1, 17, 45, 90])
        if selected:
            st.plotly_chart(px.line(stats["trends"], x="draw_index", y=selected,
                                   title="100-draw rolling occurrence rates"), width="stretch")
        st.caption("Random fluctuations are expected. These curves are not predictive evidence.")


@st.fragment(run_every=2)
def job_status():
    job = training_job()
    st.progress(job.progress, text=job.message)
    if job.future and job.future.done():
        try:
            result = job.future.result()
            st.success(f"Completed. Best-supported model: {result['selected_model']}")
            st.dataframe(pd.DataFrame(result["results"]), hide_index=True)
        except Exception as exc:
            st.error(f"Training failed: {exc}. Details are recorded in system logs.")
    elif job.future:
        st.info("CPU training runs in a background worker. Other pages remain usable.")


def laboratory():
    st.title("AI Model Laboratory")
    require_data()
    registry = available_models()
    st.dataframe(pd.DataFrame([{"model": k, "availability": str(v)} for k, v in registry.items()]),
                 hide_index=True, width="stretch")
    names = st.multiselect("Models to evaluate", list(registry),
                          default=[n for n in ("uniform", "frequency", "ew_frequency", "bayesian") if n in registry])
    evaluation = st.number_input("Walk-forward evaluation draws", 10, 250,
                                 int(preferences["validation_draws"]), step=10)
    training_window = st.number_input("Maximum classifier training draws (0 = full history)", 0, 3500,
                                      int(preferences["training_window"]), step=50)
    st.caption("Models are refitted chronologically; features contain only prior draws. Small evaluations "
               "are exploratory. The first half is used for model selection; the final half is diagnostic "
               "only and never selects or tunes a model. At least 30 development draws are required "
               "for nonuniform selection. Reusing exposed historical holdouts is not prospective validation.")
    if st.button("Train and backtest selected models", type="primary"):
        if not names:
            st.error("Select at least one model")
        else:
            try:
                training_job().start(frame, names, evaluation, preferences["seed"], training_window)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    job_status()
    st.subheader("Persisted experiment history")
    st.dataframe(db.table("model_experiments"), hide_index=True, width="stretch")


def prediction_center():
    st.title("Prediction Center")
    require_data()
    current = artifact()
    strategies = ["mixed", "ranked", "weighted", "historical", "recent", "bayesian",
                  "balanced", "ensemble", "diversified"]
    strategy = st.selectbox("Generation strategy", strategies, index=strategies.index(preferences["strategy"]))
    diversity = st.slider("Portfolio diversity", 0., 1., float(preferences["diversity"]))
    seed = st.number_input("Random seed", 0, 2**31 - 1, int(preferences["seed"]))
    st.caption("Ranking scores are not winning probabilities. Ensemble uses independent statistical "
               "heuristics unless trained model outputs are available. Diversity only reduces portfolio overlap.")
    generate, regenerate = st.columns(2)
    first = generate.button("Generate 10 combinations", type="primary")
    again = regenerate.button("Generate again")
    if first or again:
        effective_seed = int(seed) + (st.session_state.get("regeneration", 0) + 1 if again else 0)
        if again:
            st.session_state.regeneration = st.session_state.get("regeneration", 0) + 1
        valid_model = current is not None and not current.get("stale", False)
        scores = current["scores"] if valid_model else np.full(90, 5 / 90)
        model_name = current["selected_model"] if valid_model else "uniform"
        override = preferences["selected_model"]
        if override != "automatic":
            if valid_model and override in current.get("model_scores", {}):
                scores, model_name = current["model_scores"][override], override
                st.warning("Manual model selection is exploratory; it does not establish predictive improvement.")
            elif override == "uniform":
                scores, model_name = np.full(90, 5 / 90), "uniform"
            else:
                st.error("The selected model has no loaded artifact. Train/select automatic or uniform.")
                st.stop()
        try:
            ensemble_sources = ({name: values for name, values in current.get("model_scores", {}).items()
                                 if name not in ("uniform", "random")} if valid_model else None)
            predictions = generate_combinations(
                frame, scores=scores, seed=effective_seed, diversity=diversity,
                strategy=strategy, model_name=model_name, ensemble_scores=ensemble_sources)
            batch = str(uuid.uuid4())
            db.save_predictions(predictions, batch, backtest=current.get("results") if current else None)
            st.session_state.predictions = predictions
            st.session_state.batch = batch
            st.session_state.predictions_backtest = current.get("results") if current else None
        except (ValueError, RuntimeError, sqlite3.Error) as exc:
            st.error(f"Generation failed: {exc}")
    predictions = st.session_state.get("predictions", [])
    saved = db.latest_predictions()
    if not predictions and not saved.empty:
        latest_batch = saved.iloc[0].batch_id
        latest_rows = saved[saved.batch_id == latest_batch].sort_values("combination_number")
        try:
            predictions = [json.loads(value) for value in latest_rows.metadata]
            st.session_state.predictions_backtest = predictions[0].get("backtesting")
            predictions = [{k: v for k, v in p.items() if k != "backtesting"} for p in predictions]
            st.session_state.predictions = predictions
            st.session_state.batch = latest_batch
            st.caption("Showing the latest saved portfolio")
        except (ValueError, TypeError) as exc:
            st.error(f"Saved predictions cannot be decoded: {exc}")
    if not predictions:
        st.info("Generate a reproducible portfolio. Every generated batch is saved automatically.")
        return
    columns = st.columns(2)
    for i, prediction in enumerate(predictions):
        with columns[i % 2]:
            strategy_name = html.escape(prediction["strategy"])
            st.markdown(f'<div class="card"><div class="eyebrow">COMBINATION {i+1:02d} · {strategy_name}</div>'
                        f'{balls(prediction["numbers"])}</div>', unsafe_allow_html=True)
            st.caption(f"Seed {prediction['seed']} · Sum {prediction['sum']} · Odd/even {prediction['odd']}/{5-prediction['odd']} · "
                       f"Spread {prediction['spread']} · Mean score {np.mean(prediction['scores']):.4f}")
            with st.expander(f"Explanation & number scores #{i+1}"):
                st.write(prediction["explanation"])
                st.dataframe(pd.DataFrame(prediction["number_details"]), hide_index=True)
                st.write("Number bands:", [f"{((n-1)//10)*10+1}-{((n-1)//10)*10+10}"
                                          for n in prediction["numbers"]])
    if st.button("Save predictions"):
        try:
            db.save_predictions(predictions, st.session_state.batch,
                                backtest=st.session_state.get("predictions_backtest"))
            st.success("Saved (duplicate-safe)")
        except (ValueError, sqlite3.Error) as exc:
            st.error(f"Could not save predictions: {exc}")
    download(combinations_frame(predictions), "candidate_combinations.csv")
    if st.checkbox("Compare with historical results (descriptive, not a backtest)"):
        rows = frame[NUMBER_COLUMNS].to_numpy()
        comparison = []
        for i, prediction in enumerate(predictions, 1):
            matches = np.isin(rows, prediction["numbers"]).sum(axis=1)
            comparison.append({"combination": i, "historical_max_matches": int(matches.max()),
                               "historical_mean_matches": float(matches.mean()),
                               **{f"{j}_matches": int((matches == j).sum()) for j in range(6)}})
        st.dataframe(pd.DataFrame(comparison), hide_index=True)
        st.warning("These combinations were constructed using this history. This comparison is in-sample "
                   "and cannot establish predictive performance.")


def backtesting():
    st.title("Backtesting")
    require_data()
    current = artifact()
    st.info("Uniform expected overlap = 25/90 = 0.27778 numbers per five-number ticket. "
            "A highest score is not proof of improvement; corrected significance and confidence intervals matter.")
    if not current:
        st.info("Run a chronological evaluation in the AI Model Laboratory.")
        return
    results = pd.DataFrame([{
        "model": r["model"], "status": r["status"],
        "development_matches": (r.get("selection_metrics") or {}).get("mean_matches"),
        "development_p_holm": r.get("p_value_holm"),
        "final_test_matches": (r.get("final_test_metrics") or {}).get("mean_matches"),
        "baseline_comparison": (
            "Insufficient development evidence" if r.get("status") != "ok" or current.get("selection_draws", 0) < 30
            else "Provisional development improvement" if r.get("eligible")
            else "Worse final point estimate (not a significance claim)"
            if (r.get("final_test_metrics") or {}).get("mean_matches", 25 / 90) < 25 / 90
            else "No corrected development improvement"),
    } for r in current["results"]])
    st.dataframe(results, hide_index=True, width="stretch")
    successful = results[results.status == "ok"]
    if not successful.empty:
        figure = px.bar(successful, x="model", y="final_test_matches",
                        title="Final-test matches (never used for model selection)")
        figure.add_hline(y=25 / 90, line_dash="dash", annotation_text="Uniform expectation")
        st.plotly_chart(figure, width="stretch")
    st.caption(current.get("selection_reason", ""))
    st.caption("Final-test per-model p-values are unadjusted diagnostics. Inspecting many models or reusing "
               "this historical period does not provide pristine confirmatory evidence.")
    st.json({k: v for k, v in current.items()
             if k not in ("scores", "model_scores", "ensemble_scores", "results", "models", "fitted_model")},
            expanded=False)
    model_to_inspect = st.selectbox("Inspect model details", [r["model"] for r in current["results"]])
    for result in (r for r in current["results"] if r["model"] == model_to_inspect):
        with st.expander(str(result.get("model", "Model"))):
            metrics = result.get("final_test_metrics") or {}
            periods = metrics.get("periods") or []
            if periods:
                st.plotly_chart(px.line(pd.DataFrame(periods), x="first_label", y="mean_matches",
                                        markers=True, title="Matches across evaluation periods"),
                                width="stretch", key=f"periods_{result['model']}")
            bins = (metrics.get("calibration") or {}).get("bins") or []
            if bins:
                calibration = px.scatter(pd.DataFrame(bins), x="mean_predicted", y="observed_rate",
                                         size="count", title="Number-level calibration (not ticket probability)")
                st.plotly_chart(calibration, width="stretch", key=f"calibration_{result['model']}")
            histogram = metrics.get("match_histogram")
            if isinstance(histogram, dict):
                st.plotly_chart(px.bar(x=list(histogram.keys()), y=list(histogram.values()),
                                       labels={"x": "Matches", "y": "Draws"},
                                       title="Match distribution"), width="stretch",
                                key=f"histogram_{result['model']}")
            elif isinstance(histogram, list):
                st.plotly_chart(px.bar(x=list(range(len(histogram))), y=histogram,
                                       labels={"x": "Matches", "y": "Draws"},
                                       title="Match distribution"), width="stretch",
                                key=f"histogram_{result['model']}")
            st.json(result)
    experiments = db.table("model_experiments")
    download(experiments, "backtesting_experiments.csv")


def data_management():
    st.title("Data Management")
    st.link_button("Official historical source", SOURCE)
    st.write(f"Stored drawings: {len(frame):,}")
    st.write("Last synchronization:", state.get("last_success", "Never"))
    st.caption("Rebuild reconciles the complete official source transactionally; existing valid records "
               "are never deleted. Conflicts abort the import. Conditional requests reduce unnecessary downloads.")
    update, rebuild = st.columns(2)
    force = update.button("Update database", type="primary")
    reconcile = rebuild.button("Rebuild / reconcile database")
    if force or reconcile:
        try:
            with st.spinner("Downloading and validating official source..."):
                result = sync_database(db, force=True, rebuild=reconcile)
            st.success(str(result))
            st.cache_data.clear()
            st.rerun()
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(f"Import failed, saved data preserved: {exc}")
        except requests.RequestException as exc:
            st.error(f"Download failed, saved data preserved: {exc}")
    st.subheader("Data-quality report")
    quality = db.table("data_quality")
    st.dataframe(quality, hide_index=True, width="stretch")
    st.subheader("System events")
    st.dataframe(db.table("system_logs", limit=200), hide_index=True, width="stretch")
    if not frame.empty:
        download(frame, "official_history.csv")


def settings_page():
    st.title("Settings")
    with st.form("preferences"):
        values = preferences.copy()
        values["auto_update"] = st.checkbox("Automatic update on application session startup", values["auto_update"])
        values["interval_hours"] = st.number_input("Minimum update interval (hours)", 1, 168, int(values["interval_hours"]))
        values["seed"] = st.number_input("Default seed", 0, 2**31 - 1, int(values["seed"]))
        values["diversity"] = st.slider("Default diversity", 0., 1., float(values["diversity"]))
        strategies = ["mixed", "ranked", "weighted", "historical", "recent", "bayesian", "balanced", "ensemble", "diversified"]
        values["strategy"] = st.selectbox("Default strategy", strategies, index=strategies.index(values["strategy"]))
        models = ["automatic", *available_models()]
        if values["selected_model"] not in models:
            st.warning("The saved model choice is no longer available. Select a replacement and save.")
        values["selected_model"] = st.selectbox(
            "Model selection", models, index=models.index(values["selected_model"])
            if values["selected_model"] in models else 0)
        windows = [0, 10, 25, 50, 100, 250]
        values["analysis_window"] = st.selectbox("Analysis window", windows,
                                                index=windows.index(values["analysis_window"]))
        values["validation_draws"] = st.number_input("Default evaluation draws", 10, 250, int(values["validation_draws"]))
        values["training_window"] = st.number_input("Default classifier training window", 0, 3500,
                                                  int(values["training_window"]))
        if st.form_submit_button("Save settings"):
            try:
                save_preferences(values)
                st.success("Settings saved")
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(f"Could not save settings: {exc}")


navigation = st.navigation([
    st.Page(overview, title="Overview Dashboard", icon="🏠"),
    st.Page(history, title="Historical Results", icon="📚"),
    st.Page(analysis_page, title="Statistical Analysis", icon="📊"),
    st.Page(laboratory, title="AI Model Laboratory", icon="🧪"),
    st.Page(prediction_center, title="Prediction Center", icon="🎱"),
    st.Page(backtesting, title="Backtesting", icon="📈"),
    st.Page(data_management, title="Data Management", icon="🗃️"),
    st.Page(settings_page, title="Settings", icon="⚙️"),
])
with st.sidebar:
    st.caption("LOCAL RESEARCH WORKSPACE")
    disclaimer()
navigation.run()
disclaimer()
