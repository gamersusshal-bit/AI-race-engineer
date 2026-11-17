"""Engineer dashboard: ``streamlit run race_engineer/dashboard/app.py``."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from race_engineer import pipeline
from race_engineer.config import CIRCUITS, COMPOUND_NAMES, COMPOUNDS, DRIVERS
from race_engineer.data.telemetry_synth import lap_seed, synth_lap_telemetry
from race_engineer.features.telemetry import aggregate_lap
from race_engineer.models.registry import load_model
from race_engineer.sim.race import RaceState, simulate
from race_engineer.strategy.baselines import BASELINES, start_state
from race_engineer.strategy.explain import explain
from race_engineer.strategy.optimizer import optimize, pit_now_or_stay_out

ART = Path("artifacts")
COMPOUND_COLOR = {"SOFT": "#e34948", "MEDIUM": "#eda100", "HARD": "#2a78d6"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"

st.set_page_config(page_title="AI Race Engineer", layout="wide")


def style(fig: go.Figure, title: str, x: str, y: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12), hovermode="x unified",
        legend=dict(orientation="h", y=-0.28, yanchor="top", x=0, xanchor="left", title=None),
    )
    fig.update_xaxes(title=x, showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title=y, gridcolor=GRID, zeroline=False)
    return fig


@st.cache_resource(show_spinner="Building data and training models (first run only)…")
def load():
    if not (ART / "metrics.json").exists():
        pipeline.run(ART)
    return {
        "pace": pipeline.load_pace(ART),
        "deg": load_model("degradation", ART / "models")[0],
        "metrics": json.loads((ART / "metrics.json").read_text()),
        "laps": pd.read_parquet(ART / "laps.parquet"),
        "preds": pd.read_parquet(ART / "laptime_test_predictions.parquet"),
        "strategy_eval": pd.read_csv(ART / "strategy_eval.csv") if (ART / "strategy_eval.csv").exists() else None,
    }


@st.cache_data(show_spinner="Searching strategies…")
def run_optimizer(circuit: str, driver: str, temp: float, modes: bool):
    pace = load()["pace"]
    res = optimize(RaceState(circuit, driver, track_temp=temp), pace, optimize_modes=modes)
    base = []
    for name, make in BASELINES.items():
        plan = make(circuit)
        summary, laps = simulate([plan], start_state(circuit, driver, plan, temp), pace)
        base.append({"name": name, "plan": plan, "total": summary["total_time_s"].iloc[0], "laps": laps})
    return res, base, explain(res, pace)


def fmt_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}"


A = load()
pace, metrics = A["pace"], A["metrics"]

st.sidebar.title("AI Race Engineer")
circuit = st.sidebar.selectbox("Circuit", list(CIRCUITS))
driver = st.sidebar.selectbox("Driver", list(DRIVERS), index=2)
temp = st.sidebar.slider("Track temperature (°C)", 26, 46, 35)
modes = st.sidebar.toggle("Choose push / conserve per stint", value=False)
st.sidebar.caption(
    f"Data source: **{metrics['data']['source']}** (simulator-generated; circuit and driver parameters are "
    "simulation assumptions, not measured values). Lap-time model: "
    f"**{metrics['laptime']['selected']}**.")

tab_strategy, tab_call, tab_tyres, tab_models, tab_tel = st.tabs(
    ["Race strategy", "Pit call", "Tyres & fuel", "Model quality", "Telemetry"])

# ------------------------------------------------------------------ strategy
with tab_strategy:
    res, baselines, reasons = run_optimizer(circuit, driver, float(temp), modes)
    best_total = res.summary["total_time_s"].iloc[0]
    best_base = min(baselines, key=lambda b: b["total"])
    st.caption("Recommended strategy")
    st.subheader(res.summary["strategy"].iloc[0], anchor=False)
    c2, c3, _ = st.columns([1, 1, 2])
    c2.metric("Predicted race time", fmt_time(best_total))
    c3.metric("vs best rule-based baseline", f"{best_total - best_base['total']:+.1f} s",
              help=f"Best baseline: {best_base['name']}")

    rows = [(f"#{i + 1}", p, res.summary["total_time_s"].iloc[i]) for i, p in enumerate(res.plans[:5])]
    rows += [(b["name"], b["plan"], b["total"]) for b in baselines]
    fig = go.Figure()
    shown = set()
    for label, plan, total in rows[::-1]:
        lap0 = 0
        for s in plan.stints:
            fig.add_bar(y=[label], x=[s.laps - 0.3], base=[lap0 + 0.15], orientation="h", width=0.55,
                        marker_color=COMPOUND_COLOR[s.compound], name=s.compound.title(), legendgroup=s.compound,
                        showlegend=s.compound not in shown, text=f"{s.laps}", textposition="inside",
                        insidetextanchor="middle", textangle=0, textfont=dict(color="#ffffff", size=12),
                        hovertemplate=f"{s.compound.title()} · laps {lap0 + 1}–{lap0 + s.laps} · {s.mode}<extra></extra>")
            shown.add(s.compound)
            lap0 += s.laps
        fig.add_annotation(x=lap0 + 0.6, y=label, text=f"+{total - best_total:.1f} s" if total > best_total else "best",
                           showarrow=False, xanchor="left", font=dict(color=INK, size=12))
    style(fig, "Stints by strategy — top 5 found and the rule-based baselines (gap to best)", "Lap", "", 380)
    fig.update_layout(barmode="overlay", hovermode="closest", xaxis_range=[0, CIRCUITS[circuit].total_laps * 1.12])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        rec = res.laps[res.laps["plan_rank"] == 1]
        fig = go.Figure()
        fig.add_scatter(x=rec["lap_number"], y=rec["clean_lap_s"], name="Recommended", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=best_base["laps"]["lap_number"], y=best_base["laps"]["clean_lap_s"],
                        name=best_base["name"], line=dict(color=SERIES[1], width=2))
        for lap in rec.loc[rec["pit_in"], "lap_number"]:
            fig.add_vline(x=lap + 0.5, line=dict(color=MUTED, width=1, dash="dot"))
            fig.add_annotation(x=lap + 0.5, y=1, yref="paper", text=f"pit L{lap}", showarrow=False,
                               yanchor="bottom", font=dict(color=MUTED, size=11))
        style(fig, "Predicted lap time, pit-stop time excluded", "Lap", "Lap time (s)")
        fig.update_traces(hovertemplate="%{y:.2f} s")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Why this strategy")
        st.markdown("\n".join(f"- {r}" for r in reasons))
    with st.expander("All ranked strategies (table)"):
        st.dataframe(res.summary.head(50).round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ pit call
with tab_call:
    total_laps = CIRCUITS[circuit].total_laps
    a, b, c, d = st.columns(4)
    lap = a.number_input("Current lap", 2, total_laps - 1, min(20, total_laps - 1))
    comp = b.selectbox("Fitted compound", COMPOUND_NAMES, index=1)
    age = c.number_input("Tyre age (laps)", 0, COMPOUNDS[comp].max_stint_laps - 1, min(18, lap - 1))
    used = d.multiselect("Compounds already used", COMPOUND_NAMES, default=[comp])
    state = RaceState(circuit, driver, int(lap), comp, int(age), compounds_used=tuple({*used, comp}), track_temp=float(temp))
    try:
        call = pit_now_or_stay_out(state, pace)
    except ValueError as e:
        st.error(str(e))
    else:
        st.metric("Call", call["call"],
                  None if call["delta_s"] is None else f"{abs(call['delta_s']):.1f} s better than the alternative",
                  delta_color="off")
        options = pd.DataFrame([
            {"option": label, "best plan from here": call[k]["strategy"],
             "time to flag (s)": round(call[k]["total_time_s"], 1)}
            for k, label in (("pit_now", f"Pit at the end of lap {lap}"), ("stay_out", "Stay out")) if call[k]])
        st.dataframe(options, use_container_width=True, hide_index=True)
        first = pd.Series([p.stints[0].laps if p.n_stops else None for p in call["result"].plans])
        window = call["result"].summary.assign(pit_lap=first + lap - 1).dropna(subset=["pit_lap"])
        window = window.groupby("pit_lap", as_index=False)["total_time_s"].min()
        window["loss_s"] = window["total_time_s"] - window["total_time_s"].min()
        fig = go.Figure(go.Scatter(x=window["pit_lap"], y=window["loss_s"], mode="lines+markers",
                                   line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                                   hovertemplate="Pit on lap %{x}: +%{y:.2f} s<extra></extra>"))
        style(fig, "Cost of pitting on each lap (best plan for that pit lap, vs the optimum)", "Next pit stop on lap", "Time lost (s)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ tyres & fuel
with tab_tyres:
    deg = A["deg"]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for comp_name in COMPOUND_NAMES:
            if comp_name in deg.offsets.get(circuit, {}):
                ages = list(range(0, int(deg.max_age[circuit][comp_name]) + 1))
                fig.add_scatter(x=ages, y=deg.curve(circuit, comp_name, ages), name=comp_name.title(),
                                line=dict(color=COMPOUND_COLOR[comp_name], width=2), hovertemplate="%{y:+.2f} s")
        style(fig, f"Tyre degradation at {circuit}: lap-time delta vs a fresh Medium", "Tyre age (laps)", "Delta (s)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Estimated from race laps with driver/session effects and race progression removed. "
                   "Fuel burn-off and track evolution are not separated from each other: both are absorbed by a "
                   f"single progression term (averaging {deg.progression_s_per_lap[circuit]:+.3f} s per lap here).")
    with right:
        rec = res.laps[res.laps["plan_rank"] == 1]
        fig = go.Figure(go.Scatter(x=rec["lap_number"], y=rec["fuel_kg"], line=dict(color=SERIES[0], width=2),
                                   hovertemplate="%{y:.1f} kg<extra></extra>"))
        style(fig, "Projected fuel load for the recommended strategy", "Lap", "Fuel at lap start (kg)")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        burn = pace.fuel_table(circuit)
        st.caption("Expected burn per lap: " + " · ".join(f"{m} {kg:.2f} kg" for m, kg in burn.items())
                   + ". Fuel quantities are simulator values; public timing data does not contain them.")
    st.dataframe(deg.table().query("circuit == @circuit").drop(columns="circuit").round(3),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ model quality
with tab_models:
    lt = metrics["laptime"]
    table = pd.DataFrame({k: {"CV MAE (s)": v.get("cv_mae"), "Test MAE (s)": v["test"]["mae"],
                              "Test RMSE (s)": v["test"]["rmse"], "Test R²": v["test"]["r2"]}
                          for k, v in lt["models"].items()}).T
    st.subheader("Lap-time prediction on unseen sessions")
    st.caption(f"Held-out season, whole sessions unseen in training. Selected **{lt['selected']}** by "
               f"{lt['selection_rule']}. R² is on the delta to each circuit's reference lap.")
    st.dataframe(table.round(3), use_container_width=True)
    left, right = st.columns(2)
    with left:
        p = A["preds"][A["preds"]["circuit"] == circuit]
        lo, hi = p["lap_time_s"].min(), p["lap_time_s"].max()
        fig = go.Figure()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, width=1, dash="dot"),
                        name="Perfect prediction", hoverinfo="skip")
        fig.add_scatter(x=p["lap_time_s"], y=p["predicted_s"], mode="markers", name="Test laps",
                        marker=dict(color=SERIES[0], size=5, opacity=0.45),
                        hovertemplate="actual %{x:.2f} s · predicted %{y:.2f} s<extra></extra>")
        style(fig, f"Predicted vs actual lap time at {circuit} (test laps)", "Actual (s)", "Predicted (s)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        err = pd.DataFrame(lt["error_breakdown"]["circuit"]).sort_values("mae")
        fig = go.Figure(go.Bar(x=err["mae"], y=err["circuit"], orientation="h", marker_color=SERIES[0], width=0.55,
                               text=[f"{v:.2f}" for v in err["mae"]], textposition="outside", textfont=dict(color=INK),
                               hovertemplate="%{y}: MAE %{x:.3f} s<extra></extra>"))
        style(fig, "Test MAE by circuit", "Mean absolute error (s)", "")
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
        st.plotly_chart(fig, use_container_width=True)
    if metrics["fuel"]:
        st.subheader("Fuel burned per lap (kg)")
        st.dataframe(pd.DataFrame({k: v["test"] for k, v in metrics["fuel"].items()}).T.round(4), use_container_width=True)
        st.caption("telemetry_linear sees only telemetry and circuit; baseline_circuit_mode_mean is given the "
                   "drive-mode label and is the table the simulator uses to project fuel.")
    if A["strategy_eval"] is not None:
        st.subheader("Do the models lead to good decisions?")
        st.caption("Strategies chosen with the learned models, re-scored on the simulator's reference physics. "
                   "Regret is time lost to the true optimum; gains are time saved vs each rule-based baseline.")
        ev = A["strategy_eval"]
        st.dataframe(ev[ev["deg_scale"] == 1.0].drop(columns=["deg_scale", "predicted_time_s"]).round(2),
                     use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ telemetry
with tab_tel:
    laps = A["laps"]
    a, b = st.columns(2)
    session = a.selectbox("Session", sorted(laps.loc[laps["circuit"] == circuit, "session_id"].unique(), reverse=True))
    mine = laps[(laps["session_id"] == session) & (laps["driver"] == driver)]
    lap_no = b.select_slider("Lap", list(mine["lap_number"]), value=int(mine["lap_number"].iloc[len(mine) // 2]))
    row = mine[mine["lap_number"] == lap_no].iloc[0]
    tel = synth_lap_telemetry(circuit, row["lap_time_s"], row["drive_mode"], lap_seed(session, driver, int(lap_no)))
    feats = aggregate_lap(tel)
    cols = st.columns(5)
    cols[0].metric("Lap time", f"{row['lap_time_s']:.2f} s")
    cols[1].metric("Tyre (laps old)", f"{row['compound'].title()} ({int(row['tyre_age'])})")
    cols[2].metric("Avg speed", f"{feats['tel_avg_speed']:.0f} km/h")
    cols[3].metric("Full throttle", f"{feats['tel_full_throttle_frac']:.0%}")
    cols[4].metric("Braking", f"{feats['tel_brake_time_s']:.1f} s")
    for col, title, unit in (("speed", "Speed", "km/h"), ("throttle", "Throttle", "%"), ("gear", "Gear", "")):
        fig = go.Figure(go.Scatter(x=tel["distance_m"], y=tel[col], line=dict(color=SERIES[0], width=2,
                                   shape="hv" if col != "speed" else "linear"),
                                   hovertemplate=f"%{{y:.0f}} {unit}<extra></extra>"))
        style(fig, title, "Distance (m)" if col == "gear" else "", unit, 220 if col == "speed" else 160)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Telemetry traces are synthesised with irregular sampling; lap features are time-weighted. "
               f"Samples this lap: {len(tel)}, drive mode: {row['drive_mode']}.")
