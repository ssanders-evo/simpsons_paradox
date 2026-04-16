import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Call Center Learning Simulation",
    layout="wide",
)


# -----------------------------
# Simulation
# -----------------------------
def simulate_call_center(
    n_experts=2500,
    min_engagements=2,
    max_engagements=55,
    mean_cst=78.0,
    proficiency_sd=11.0,
    noise_sd=9.0,
    small_learning=-0.12,
    complexity_drift_mean=0.03,
    complexity_drift_sd=0.015,
    volume_link_strength=0.95,
    volume_noise_sd=0.45,
    seed=7,
):
    """
    Simulate expert-level engagement data where:
      - experts vary in baseline proficiency
      - more proficient experts tend to accumulate more engagements
      - within-expert learning is weak or absent
      - total engagement counts are right-skewed
    """
    rng = np.random.default_rng(seed)

    proficiency = rng.normal(loc=0.0, scale=1.0, size=n_experts)

    raw_volume = np.exp(
        volume_link_strength * proficiency + rng.normal(0, volume_noise_sd, n_experts)
    )
    raw_volume = (raw_volume - raw_volume.min()) / (raw_volume.max() - raw_volume.min())
    total_engagements = np.floor(
        min_engagements + raw_volume * (max_engagements - min_engagements)
    ).astype(int)
    total_engagements = np.clip(total_engagements, min_engagements, max_engagements)

    expert_baseline = mean_cst - proficiency_sd * proficiency

    expert_df = pd.DataFrame(
        {
            "expert_id": np.arange(n_experts),
            "proficiency": proficiency,
            "baseline_cst": expert_baseline,
            "total_engagements": total_engagements,
        }
    )

    rows = []
    for row in expert_df.itertuples(index=False):
        mild_complexity_drift = rng.normal(
            loc=complexity_drift_mean, scale=complexity_drift_sd
        )

        for nth in range(1, row.total_engagements + 1):
            time_effect = small_learning * (nth - 1)
            complexity_effect = mild_complexity_drift * (nth - 1)
            eps = rng.normal(0, noise_sd)

            cst = row.baseline_cst + time_effect + complexity_effect + eps
            cst = max(5, cst)

            rows.append(
                (row.expert_id, nth, row.total_engagements, row.baseline_cst, cst)
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "expert_id",
            "nth_engagement",
            "total_engagements",
            "baseline_cst",
            "cst_minutes",
        ],
    )
    return df, expert_df


# -----------------------------
# Aggregations
# -----------------------------
def make_aggregate_curve(df):
    return (
        df.groupby("nth_engagement", as_index=False)
        .agg(avg_cst=("cst_minutes", "mean"), n_experts=("expert_id", "nunique"))
    )



def make_segmented_curves(df, cutoffs=(5, 10, 15, 20)):
    parts = []
    for cutoff in cutoffs:
        sub = df[(df["total_engagements"] >= cutoff) & (df["nth_engagement"] <= cutoff)].copy()
        curve = (
            sub.groupby("nth_engagement", as_index=False)["cst_minutes"]
            .mean()
            .rename(columns={"cst_minutes": "avg_cst"})
        )
        curve["target_n"] = f"{cutoff}+"
        curve["cutoff"] = cutoff
        curve["n_experts_in_group"] = sub["expert_id"].nunique()
        parts.append(curve)
    return pd.concat(parts, ignore_index=True)



def expert_level_summary(df, expert_df):
    tmp = (
        df.groupby("expert_id", as_index=False)
        .agg(
            observed_mean_cst=("cst_minutes", "mean"),
            total_engagements=("total_engagements", "max"),
        )
        .merge(expert_df[["expert_id", "baseline_cst"]], on="expert_id", how="left")
    )

    corr_baseline_volume = np.corrcoef(tmp["baseline_cst"], tmp["total_engagements"])[0, 1]
    corr_observed_volume = np.corrcoef(tmp["observed_mean_cst"], tmp["total_engagements"])[0, 1]

    bins = [0, 5, 10, 15, 20, 30, 10**9]
    labels = ["<=5", "6-10", "11-15", "16-20", "21-30", "31+"]
    tmp["engagement_bin"] = pd.cut(
        tmp["total_engagements"], bins=bins, labels=labels, include_lowest=True
    )

    summary = (
        tmp.groupby("engagement_bin", observed=False)
        .agg(
            n_experts=("expert_id", "count"),
            mean_baseline_cst=("baseline_cst", "mean"),
            mean_observed_cst=("observed_mean_cst", "mean"),
            mean_total_engagements=("total_engagements", "mean"),
        )
        .reset_index()
    )

    return tmp, summary, corr_baseline_volume, corr_observed_volume


# -----------------------------
# Chart helpers
# -----------------------------
def prep_aggregate_chart(aggregate):
    chart_df = aggregate[["nth_engagement", "avg_cst"]].copy()
    chart_df = chart_df.set_index("nth_engagement")
    return chart_df



def prep_segmented_chart(segmented, cutoffs):
    wide = (
        segmented[segmented["cutoff"].isin(cutoffs)]
        .pivot(index="nth_engagement", columns="target_n", values="avg_cst")
        .sort_index()
    )
    ordered_cols = [f"{c}+" for c in cutoffs if f"{c}+" in wide.columns]
    if ordered_cols:
        wide = wide[ordered_cols]
    return wide



def prep_tail_sample_chart(aggregate):
    chart_df = aggregate[["nth_engagement", "n_experts"]].copy()
    chart_df = chart_df.set_index("nth_engagement")
    return chart_df



def prep_volume_histogram(expert_df):
    counts = (
        expert_df["total_engagements"]
        .value_counts()
        .sort_index()
        .rename_axis("total_engagements")
        .reset_index(name="n_experts")
        .set_index("total_engagements")
    )
    return counts


# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    "Composition artifact": {
        "n_experts": 2500,
        "max_engagements": 55,
        "mean_cst": 78.0,
        "proficiency_sd": 11.0,
        "volume_link_strength": 0.95,
        "small_learning": -0.12,
        "noise_sd": 9.0,
        "complexity_drift_mean": 0.03,
        "volume_noise_sd": 0.45,
        "cutoffs": [5, 10, 15, 20],
    },
    "True learning only": {
        "n_experts": 2500,
        "max_engagements": 55,
        "mean_cst": 78.0,
        "proficiency_sd": 2.0,
        "volume_link_strength": 0.05,
        "small_learning": -0.45,
        "noise_sd": 9.0,
        "complexity_drift_mean": 0.00,
        "volume_noise_sd": 0.65,
        "cutoffs": [5, 10, 15, 20],
    },
    "Learning plus selection": {
        "n_experts": 2500,
        "max_engagements": 55,
        "mean_cst": 78.0,
        "proficiency_sd": 9.0,
        "volume_link_strength": 0.75,
        "small_learning": -0.30,
        "noise_sd": 9.0,
        "complexity_drift_mean": 0.02,
        "volume_noise_sd": 0.45,
        "cutoffs": [5, 10, 15, 20],
    },
    "Tail instability stress test": {
        "n_experts": 600,
        "max_engagements": 70,
        "mean_cst": 78.0,
        "proficiency_sd": 11.0,
        "volume_link_strength": 1.10,
        "small_learning": -0.10,
        "noise_sd": 14.0,
        "complexity_drift_mean": 0.03,
        "volume_noise_sd": 0.35,
        "cutoffs": [5, 10, 15, 20],
    },
}


# -----------------------------
# UI
# -----------------------------
st.title("Call Center Learning Simulation")
st.caption(
    "Pedagogical app for showing how aggregate learning curves can be driven by selection, heterogeneity, and tail instability."
)

with st.sidebar:
    st.header("Scenario")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    st.header("Simulation controls")
    n_experts = st.slider("Number of experts", 200, 10000, preset["n_experts"], 100)
    max_engagements = st.slider("Maximum engagements", 20, 100, preset["max_engagements"], 1)
    mean_cst = st.slider("Overall mean CST", 40.0, 120.0, preset["mean_cst"], 1.0)

    st.subheader("Key teaching knobs")
    proficiency_sd = st.slider(
        "Between-expert heterogeneity",
        0.0,
        20.0,
        preset["proficiency_sd"],
        0.5,
        help="Higher values create larger baseline CST differences across experts. Lesson: stronger heterogeneity creates more vertical separation across segmented curves.",
    )
    volume_link_strength = st.slider(
        "Selection strength: proficiency -> volume",
        0.0,
        2.0,
        preset["volume_link_strength"],
        0.05,
        help="Higher values make low-CST experts more likely to reach later engagements. Lesson: stronger selection makes the aggregate curve look more like learning even when within-expert learning is weak.",
    )
    learning_improvement = st.slider(
        "Within-expert efficiency gain per engagement",
        0.0,
        1.0,
        max(0.0, -preset["small_learning"]),
        0.01,
        help="Higher values mean experts get faster over successive engagements, reducing CST.",
    )
    noise_sd = st.slider(
        "Engagement-level noise",
        0.0,
        25.0,
        preset["noise_sd"],
        0.5,
        help="Higher values increase jaggedness. Lesson: random noise matters much more when only a few experts remain in the tail.",
    )

    st.subheader("Optional realism knobs")
    complexity_drift_mean = st.slider(
        "Case complexity drift per engagement",
        -0.20,
        0.20,
        preset["complexity_drift_mean"],
        0.005,
        help="Positive values make later engagements slightly harder. Lesson: even a small worsening case mix can flatten or offset real learning.",
    )
    volume_noise_sd = st.slider(
        "Randomness in engagement count generation",
        0.0,
        1.5,
        preset["volume_noise_sd"],
        0.05,
        help="Higher values weaken the clean link between proficiency and total engagements. Lesson: noisier assignment reduces the composition effect.",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1)

    st.subheader("Segmented curve settings")
    selected_cutoffs = st.multiselect(
        "Inclusive cohort cutoffs",
        options=[3, 5, 8, 10, 12, 15, 20, 25, 30],
        default=preset["cutoffs"],
        help="Experts with total engagements >= cutoff are included in that cohort, and the curve is censored at the cutoff.",
    )
    selected_cutoffs = sorted(selected_cutoffs) if selected_cutoffs else [5, 10, 15, 20]


@st.cache_data(show_spinner=False)
def run_simulation_cached(
    n_experts,
    max_engagements,
    mean_cst,
    proficiency_sd,
    volume_link_strength,
    learning_improvement,
    noise_sd,
    complexity_drift_mean,
    volume_noise_sd,
    seed,
    cutoffs,
):
    df, expert_df = simulate_call_center(
        n_experts=n_experts,
        min_engagements=2,
        max_engagements=max_engagements,
        mean_cst=mean_cst,
        proficiency_sd=proficiency_sd,
        noise_sd=noise_sd,
        small_learning=-learning_improvement,
        complexity_drift_mean=complexity_drift_mean,
        complexity_drift_sd=0.015,
        volume_link_strength=volume_link_strength,
        volume_noise_sd=volume_noise_sd,
        seed=seed,
    )
    aggregate = make_aggregate_curve(df)
    segmented = make_segmented_curves(df, cutoffs=tuple(cutoffs))
    expert_level, summary, corr_baseline_volume, corr_observed_volume = expert_level_summary(df, expert_df)
    return (
        df,
        expert_df,
        aggregate,
        segmented,
        expert_level,
        summary,
        corr_baseline_volume,
        corr_observed_volume,
    )


(
    df,
    expert_df,
    aggregate,
    segmented,
    expert_level,
    summary,
    corr_baseline_volume,
    corr_observed_volume,
) = run_simulation_cached(
    n_experts,
    max_engagements,
    mean_cst,
    proficiency_sd,
    volume_link_strength,
    learning_improvement,
    noise_sd,
    complexity_drift_mean,
    volume_noise_sd,
    int(seed),
    tuple(selected_cutoffs),
)

aggregate_chart = prep_aggregate_chart(aggregate)
segmented_chart = prep_segmented_chart(segmented, selected_cutoffs)
tail_chart = prep_tail_sample_chart(aggregate)
volume_hist = prep_volume_histogram(expert_df)




tab1, tab2, tab3 = st.tabs([
    "Aggregate view",
    "Segmented view",
    "Diagnostics",
])

with tab1:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Aggregate CST by Nth Engagement")
        st.line_chart(aggregate_chart, height=360)
    with right:
        st.metric("Experts at engagement 1", int(aggregate["n_experts"].iloc[0]))
        st.metric("Experts at final engagement", int(aggregate["n_experts"].iloc[-1]))
        st.metric("Aggregate CST at engagement 1", f"{aggregate['avg_cst'].iloc[0]:.1f}")
        st.metric("Aggregate CST at final engagement", f"{aggregate['avg_cst'].iloc[-1]:.1f}")

    with st.expander("Show segmented curves here too"):
        st.line_chart(segmented_chart, height=360)

with tab2:
    st.subheader("Segmented CST Curves")
    st.line_chart(segmented_chart, height=420)
    segmented_table = segmented.copy()
    segmented_table["avg_cst"] = segmented_table["avg_cst"].round(2)
    st.dataframe(segmented_table, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Experts Contributing to Each Nth Engagement")
        st.line_chart(tail_chart, height=320)
    with c2:
        st.subheader("Distribution of Total Engagements per Expert")
        st.bar_chart(volume_hist, height=320)

    st.write("Selection diagnostics")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Corr(baseline CST, total engagements)", f"{corr_baseline_volume:.3f}")
    with m2:
        st.metric("Corr(observed mean CST, total engagements)", f"{corr_observed_volume:.3f}")

    st.dataframe(summary.round(2), use_container_width=True)

    with st.expander("Raw aggregate curve data"):
        st.dataframe(aggregate.round(2), use_container_width=True)

    with st.expander("Raw expert-level data sample"):
        st.dataframe(df.head(500).round(2), use_container_width=True)

st.download_button(
    label="Download simulated data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="call_center_simulation.csv",
    mime="text/csv",
)


