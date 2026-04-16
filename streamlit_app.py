import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # Higher proficiency => lower baseline CST and more total engagements
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
        curve["target_n"] = cutoff
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
# Plotting
# -----------------------------
def plot_aggregate(aggregate):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(aggregate["nth_engagement"], aggregate["avg_cst"], marker="o")
    ax.set_title("Aggregate CST by Nth Engagement")
    ax.set_xlabel("Nth Engagement")
    ax.set_ylabel("Average CST (minutes)")
    ax.grid(alpha=0.3)
    return fig



def plot_segmented(segmented, cutoffs):
    fig, ax = plt.subplots(figsize=(11, 5))
    for cutoff in cutoffs:
        tmp = segmented[segmented["target_n"] == cutoff]
        ax.plot(tmp["nth_engagement"], tmp["avg_cst"], marker="o", label=f"{cutoff}+")
    ax.set_title("Segmented CST Curves (Inclusive Threshold Cohorts)")
    ax.set_xlabel("Nth Engagement")
    ax.set_ylabel("Average CST (minutes)")
    ax.legend(title="Target n")
    ax.grid(alpha=0.3)
    return fig



def plot_tail_sample_size(aggregate):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(aggregate["nth_engagement"], aggregate["n_experts"], marker="o")
    ax.set_title("Experts Contributing to Each Nth Engagement")
    ax.set_xlabel("Nth Engagement")
    ax.set_ylabel("Unique Experts")
    ax.grid(alpha=0.3)
    return fig



def plot_volume_histogram(expert_df):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(expert_df["total_engagements"], bins=30)
    ax.set_title("Distribution of Total Engagements per Expert")
    ax.set_xlabel("Total Engagements")
    ax.set_ylabel("Number of Experts")
    ax.grid(alpha=0.2)
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Call Center Learning Simulation")
st.caption(
    "Pedagogical app for showing how aggregate learning curves can be driven by selection, heterogeneity, and tail instability."
)

with st.sidebar:
    st.header("Simulation controls")

    n_experts = st.slider("Number of experts", 200, 10000, 2500, 100)
    max_engagements = st.slider("Maximum engagements", 20, 100, 55, 1)
    mean_cst = st.slider("Overall mean CST", 40.0, 120.0, 78.0, 1.0)

    st.subheader("Key teaching knobs")
    proficiency_sd = st.slider(
        "Between-expert heterogeneity",
        0.0,
        20.0,
        11.0,
        0.5,
        help="Higher values create larger baseline CST differences across experts. Lesson: stronger heterogeneity creates more vertical separation across segmented curves.",
    )
    volume_link_strength = st.slider(
        "Selection strength: proficiency -> volume",
        0.0,
        2.0,
        0.95,
        0.05,
        help="Higher values make low-CST experts more likely to reach later engagements. Lesson: stronger selection makes the aggregate curve look more like learning even when within-expert learning is weak.",
    )
    small_learning = st.slider(
        "True within-expert learning per engagement",
        -1.0,
        1.0,
        -0.12,
        0.01,
        help="Negative values mean experts genuinely get faster. Lesson: compare real within-expert learning against the apparent learning in the aggregate curve.",
    )
    noise_sd = st.slider(
        "Engagement-level noise",
        0.0,
        25.0,
        9.0,
        0.5,
        help="Higher values increase jaggedness. Lesson: random noise matters much more when only a few experts remain in the tail.",
    )

    st.subheader("Optional realism knobs")
    complexity_drift_mean = st.slider(
        "Case complexity drift per engagement",
        -0.20,
        0.20,
        0.03,
        0.005,
        help="Positive values make later engagements slightly harder. Lesson: even a small worsening case mix can flatten or reverse true learning.",
    )
    volume_noise_sd = st.slider(
        "Randomness in engagement count generation",
        0.0,
        1.5,
        0.45,
        0.05,
        help="Higher values weaken the clean link between proficiency and total engagements. Lesson: noisier assignment reduces the composition effect.",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1)

    st.subheader("Segmented curve settings")
    selected_cutoffs = st.multiselect(
        "Inclusive cohort cutoffs",
        options=[3, 5, 8, 10, 12, 15, 20, 25, 30],
        default=[5, 10, 15, 20],
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
    small_learning,
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
        small_learning=small_learning,
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
    small_learning,
    noise_sd,
    complexity_drift_mean,
    volume_noise_sd,
    int(seed),
    tuple(selected_cutoffs),
)


# -----------------------------
# Main layout
# -----------------------------
lesson = """
Default reading of the aggregate curve is dangerous. A declining pooled CST curve can be produced by changing sample composition rather than real learning. In this simulation, later nth engagements are disproportionately contributed by experts who were already better to begin with.
"""
st.info(lesson)


tab1, tab2, tab3, tab4 = st.tabs([
    "Aggregate view",
    "Segmented view",
    "Diagnostics",
    "Teaching notes",
])

with tab1:
    left, right = st.columns([2, 1])
    with left:
        st.pyplot(plot_aggregate(aggregate), clear_figure=True)
    with right:
        st.metric("Experts at engagement 1", int(aggregate["n_experts"].iloc[0]))
        st.metric("Experts at final engagement", int(aggregate["n_experts"].iloc[-1]))
        st.metric("Aggregate CST at engagement 1", f"{aggregate['avg_cst'].iloc[0]:.1f}")
        st.metric("Aggregate CST at final engagement", f"{aggregate['avg_cst'].iloc[-1]:.1f}")

        st.write("What to notice")
        st.write(
            "If the aggregate curve slopes down, that does not by itself imply within-expert learning. Check whether later positions are supported by a different subset of experts."
        )

    with st.expander("Show segmented curves here too"):
        st.pyplot(plot_segmented(segmented, selected_cutoffs), clear_figure=True)

with tab2:
    st.pyplot(plot_segmented(segmented, selected_cutoffs), clear_figure=True)
    st.write("Interpretation")
    st.write(
        "Vertical separation across curves indicates baseline differences across experts who survive to higher engagement counts. Flat within-cohort lines imply little true learning. Strong downward slopes within each cohort would be stronger evidence of real within-expert improvement."
    )

    segmented_table = segmented.copy()
    segmented_table["avg_cst"] = segmented_table["avg_cst"].round(2)
    st.dataframe(segmented_table, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_tail_sample_size(aggregate), clear_figure=True)
    with c2:
        st.pyplot(plot_volume_histogram(expert_df), clear_figure=True)

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

with tab4:
    st.subheader("How to teach with the controls")

    st.markdown(
        """
1. **Start from the default aggregate view.**
   Ask: does this look like learning? Most people will say yes.

2. **Then open the segmented view.**
   Show that much of the pattern is actually vertical separation, not within-group decline.

3. **Increase selection strength.**
   The aggregate decline gets steeper, even if true learning stays close to zero. This teaches how survivorship or selective continuation can create misleading pooled trends.

4. **Increase between-expert heterogeneity.**
   The segmented curves separate more strongly. This teaches how baseline differences drive composition artifacts.

5. **Set true learning close to zero.**
   If the aggregate still declines, that demonstrates the core point: pooled decline is not sufficient evidence for learning.

6. **Increase engagement-level noise or reduce number of experts.**
   The tail gets spiky. This teaches why late-sequence averages become unstable when few experts remain in the denominator.

7. **Add positive case complexity drift.**
   This can flatten or offset real learning. This teaches that outcome trends reflect both proficiency change and task mix.
        """
    )

    st.subheader("Interpretive lesson")
    st.write(
        "When the composition of who remains in the sample changes over sequence position, aggregate trajectories confound within-expert change with between-expert selection. The right analysis question is usually not just 'what is the average outcome at nth engagement?' but 'who is still represented at nth engagement, and how do they differ from those who are no longer represented?'"
    )

    st.subheader("Suggested classroom demos")
    st.write(
        "A strong sequence is: first show a misleading aggregate decline, then reveal segmented curves, then use the selection-strength knob to make the artifact stronger or weaker in real time."
    )


st.download_button(
    label="Download simulated data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="call_center_simulation.csv",
    mime="text/csv",
)
