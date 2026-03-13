"""Interactive Streamlit dashboard for BDS Medtech Survival Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import bds_client
import survival

st.set_page_config(page_title="Medtech Firm Survival", layout="wide")


@st.cache_data(ttl=3600)
def load_timeseries(naics: str) -> pd.DataFrame:
    return bds_client.get_timeseries(naics=naics)


@st.cache_data(ttl=3600)
def load_firm_age(naics: str) -> pd.DataFrame:
    return bds_client.get_firm_age_timeseries(naics=naics)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
naics = st.sidebar.text_input("NAICS Code", value="3391")
naics_label = st.sidebar.text_input("Industry Label", value="Medical Equipment & Supplies")

ts = load_timeseries(naics)
age_df = load_firm_age(naics)

metrics = survival.compute_annual_metrics(ts)
year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(metrics["YEAR"].min()),
    max_value=int(metrics["YEAR"].max()),
    value=(int(metrics["YEAR"].min()), int(metrics["YEAR"].max())),
)
metrics_filtered = metrics[(metrics["YEAR"] >= year_range[0]) & (metrics["YEAR"] <= year_range[1])]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Data: [Census Bureau BDS](https://www.census.gov/programs-surveys/bds.html)  \n"
    "NAICS codes exclude M&A and reclassification from firm deaths."
)

# ---------------------------------------------------------------------------
# Key Insight Metrics (top of page)
# ---------------------------------------------------------------------------
st.title(f"Medtech Firm Survival — NAICS {naics}")

cumul = survival.compute_cumulative_survival_by_year(age_df)
profile = survival.compute_survival_profile(age_df, ts)

# Recent averages for headline stats — use last 5 cohort years available
# (cumul ends ~5 years before the latest data year since it needs 5 forward years)
if not cumul.empty:
    cumul_max = int(cumul["YEAR"].max())
    recent_cumul = cumul[cumul["YEAR"] >= cumul_max - 4]
else:
    recent_cumul = cumul

yr1_death = (100 - recent_cumul["SURV_YEAR_1"].mean()) if (not recent_cumul.empty and "SURV_YEAR_1" in recent_cumul.columns) else 0
yr5_death = (100 - recent_cumul["SURV_YEAR_5"].mean()) if (not recent_cumul.empty and "SURV_YEAR_5" in recent_cumul.columns) else 0
cond_1_to_5 = recent_cumul["COND_1_TO_5"].mean() if (not recent_cumul.empty and "COND_1_TO_5" in recent_cumul.columns) else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Die in Year 1", f"{yr1_death:.0f}%", help="% of new firms that don't survive their first year")
col2.metric("Die within 5 Years", f"{yr5_death:.0f}%", help="% of new firms that don't reach age 5")
col3.metric(
    "Year 1 Survivors → Year 5",
    f"{cond_1_to_5:.0f}%",
    help="If a firm survives year 1, probability of reaching year 5",
)
col4.metric(
    "Avg Dying Firm Size",
    f"{profile['avg_dying_firm_size']:.0f} emp",
    help="Average employees at firms that die",
)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_survival, tab_industry, tab_age, tab_deaths, tab_data = st.tabs(
    ["Survival Rates", "Industry Overview", "Age Distribution", "Firm Deaths", "Raw Data"]
)

# ======================== TAB 1: SURVIVAL RATES ============================
with tab_survival:
    st.header("5-Year Survival Rate Over Time")

    # Main chart: cumulative 5-year survival by cohort year
    if "SURV_YEAR_5" in cumul.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumul["YEAR"], y=cumul["SURV_YEAR_1"],
            mode="lines", name="1-Year Survival",
            line=dict(width=2, color="#2196F3"),
        ))
        fig.add_trace(go.Scatter(
            x=cumul["YEAR"], y=cumul["SURV_YEAR_3"],
            mode="lines", name="3-Year Survival",
            line=dict(width=2, color="#FF9800"),
        ))
        fig.add_trace(go.Scatter(
            x=cumul["YEAR"], y=cumul["SURV_YEAR_5"],
            mode="lines+markers", name="5-Year Survival",
            line=dict(width=3, color="#F44336"),
            marker=dict(size=5),
        ))

        # Average line
        avg_5yr = cumul["SURV_YEAR_5"].mean()
        fig.add_hline(
            y=avg_5yr, line_dash="dash", line_color="gray",
            annotation_text=f"Avg 5yr: {avg_5yr:.0f}%",
        )
        fig.update_layout(
            yaxis_title="Cumulative Survival (%)",
            xaxis_title="Cohort Year (year firm was founded)",
            yaxis_range=[0, 100],
            height=500,
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data to compute 5-year survival rates.")

    # Conditional survival chart
    st.subheader("Conditional Survival: If You Survive Year 1...")
    if "COND_1_TO_5" in cumul.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cumul["YEAR"], y=cumul["COND_1_TO_5"],
            mode="lines+markers", name="P(reach Year 5 | survived Year 1)",
            line=dict(width=2, color="#4CAF50"),
            marker=dict(size=5),
        ))
        avg_cond = cumul["COND_1_TO_5"].mean()
        fig2.add_hline(y=avg_cond, line_dash="dash", line_color="gray",
                       annotation_text=f"Avg: {avg_cond:.0f}%")
        fig2.update_layout(
            yaxis_title="Conditional Survival (%)",
            xaxis_title="Cohort Year",
            yaxis_range=[40, 100],
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Cumulative survival waterfall for most recent period
    st.subheader("Survival Funnel (Recent 5-Year Average)")
    surv_labels = [label for label, _ in profile["cumulative_survival"]]
    surv_values = [pct for _, pct in profile["cumulative_survival"]]

    fig3 = go.Figure(go.Funnel(
        y=surv_labels,
        x=surv_values,
        textinfo="value+percent initial",
        marker=dict(color=["#4CAF50", "#8BC34A", "#CDDC39", "#FFC107", "#FF9800", "#F44336"][:len(surv_values)]),
    ))
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ======================== TAB 2: INDUSTRY OVERVIEW =========================
with tab_industry:
    st.header("Industry Overview")

    col_left, col_right = st.columns(2)

    with col_left:
        # Firms vs Employment
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_filtered["YEAR"], y=metrics_filtered["FIRM"],
            name="Firms", yaxis="y1",
            line=dict(width=2, color="#2196F3"),
        ))
        fig.add_trace(go.Scatter(
            x=metrics_filtered["YEAR"], y=metrics_filtered["EMP"],
            name="Employment", yaxis="y2",
            line=dict(width=2, color="#F44336"),
        ))
        fig.update_layout(
            title="Firms vs. Employment",
            yaxis=dict(title=dict(text="Firms", font=dict(color="#2196F3")), tickfont=dict(color="#2196F3")),
            yaxis2=dict(title=dict(text="Employment", font=dict(color="#F44336")), tickfont=dict(color="#F44336"),
                        overlaying="y", side="right"),
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Average firm size
        fig = px.line(
            metrics_filtered, x="YEAR", y="AVG_FIRM_SIZE",
            title="Average Firm Size (Consolidation)",
            labels={"AVG_FIRM_SIZE": "Employees per Firm", "YEAR": "Year"},
        )
        fig.update_traces(line=dict(width=2, color="#4CAF50"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Job dynamics
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_filtered["YEAR"], y=metrics_filtered["JOB_CREATION"],
        name="Job Creation", marker_color="#4CAF50", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=metrics_filtered["YEAR"], y=-metrics_filtered["JOB_DESTRUCTION"],
        name="Job Destruction", marker_color="#F44336", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=metrics_filtered["YEAR"], y=metrics_filtered["NET_JOB_CREATION"],
        name="Net", mode="lines+markers", line=dict(color="black", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        title="Job Creation vs. Destruction",
        barmode="relative", height=400,
        yaxis_title="Jobs",
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================== TAB 3: AGE DISTRIBUTION ==========================
with tab_age:
    st.header("Firm Age Distribution")

    dist = survival.compute_age_distribution(age_df)
    dist_reset = dist.reset_index()

    # Stacked area chart
    dist_melted = dist_reset.melt(id_vars="YEAR", var_name="Age Bucket", value_name="Percentage")
    fig = px.area(
        dist_melted, x="YEAR", y="Percentage", color="Age Bucket",
        title="Firm Age Distribution Over Time",
        labels={"YEAR": "Year", "Percentage": "% of Firms"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Death rate by age (bar chart)
    st.subheader("Annual Death Rate by Firm Age")
    death_rates = survival.compute_death_rate_by_age(age_df)

    dr_year_range = st.slider(
        "Death Rate Year Range",
        min_value=int(death_rates["YEAR"].min()),
        max_value=int(death_rates["YEAR"].max()),
        value=(int(death_rates["YEAR"].max()) - 4, int(death_rates["YEAR"].max())),
        key="dr_slider",
    )
    dr_filtered = death_rates[
        (death_rates["YEAR"] >= dr_year_range[0]) & (death_rates["YEAR"] <= dr_year_range[1])
    ]
    avg_dr = dr_filtered.groupby("FAGE_LABEL")["DEATH_RATE"].mean().reset_index()
    # Reorder by age code
    age_order = [l for l in survival.INDIVIDUAL_AGE_LABELS if l in avg_dr["FAGE_LABEL"].values]
    avg_dr["FAGE_LABEL"] = pd.Categorical(avg_dr["FAGE_LABEL"], categories=age_order, ordered=True)
    avg_dr = avg_dr.sort_values("FAGE_LABEL")

    fig = px.bar(
        avg_dr, x="FAGE_LABEL", y="DEATH_RATE",
        title=f"Average Annual Death Rate by Age ({dr_year_range[0]}-{dr_year_range[1]})",
        labels={"FAGE_LABEL": "Firm Age", "DEATH_RATE": "Death Rate (%)"},
        color="DEATH_RATE",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ======================== TAB 4: FIRM DEATHS ===============================
with tab_deaths:
    st.header("Firm Deaths: Who Dies?")

    death_comp = survival.compute_estab_vs_firm_deaths(ts)
    dc_filtered = death_comp[(death_comp["YEAR"] >= year_range[0]) & (death_comp["YEAR"] <= year_range[1])]

    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dc_filtered["YEAR"], y=dc_filtered["AVG_DYING_FIRM_SIZE"],
            name="Dying Firms", line=dict(color="#F44336", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dc_filtered["YEAR"], y=dc_filtered["AVG_SURVIVING_FIRM_SIZE"],
            name="Surviving Firms", line=dict(color="#4CAF50", width=2),
        ))
        fig.update_layout(
            title="Avg Size: Dying vs. Surviving Firms",
            yaxis_title="Employees per Firm",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dc_filtered["YEAR"], y=dc_filtered["FIRM_DEATH_SHARE_OF_FIRMS"],
            name="% of Firms", line=dict(color="#FF9800", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dc_filtered["YEAR"], y=dc_filtered["FIRMDEATH_EMP_SHARE"],
            name="% of Employment", line=dict(color="#9C27B0", width=2),
        ))
        fig.update_layout(
            title="Deaths: Share of Firms vs. Share of Employment",
            yaxis_title="Percentage (%)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        **Key insight**: While **~{dc_filtered['FIRM_DEATH_SHARE_OF_FIRMS'].mean():.0f}%** of firms die each year,
        they account for only **~{dc_filtered['FIRMDEATH_EMP_SHARE'].mean():.1f}%** of employment.
        Dying firms average **{dc_filtered['AVG_DYING_FIRM_SIZE'].mean():.0f} employees**
        vs. **{dc_filtered['AVG_SURVIVING_FIRM_SIZE'].mean():.0f}** for survivors.
        Firm deaths are overwhelmingly a small-firm phenomenon.
        """
    )

# ======================== TAB 5: RAW DATA ==================================
with tab_data:
    st.header("Raw Data")

    data_choice = st.selectbox("Dataset", ["Time Series", "Firm Age", "Survival by Cohort", "Death Rates by Age"])

    if data_choice == "Time Series":
        st.dataframe(metrics_filtered, use_container_width=True, height=500)
    elif data_choice == "Firm Age":
        st.dataframe(age_df, use_container_width=True, height=500)
    elif data_choice == "Survival by Cohort":
        st.dataframe(cumul, use_container_width=True, height=500)
    elif data_choice == "Death Rates by Age":
        st.dataframe(survival.compute_death_rate_by_age(age_df), use_container_width=True, height=500)

    st.download_button(
        "Download as CSV",
        data=metrics.to_csv(index=False),
        file_name=f"bds_naics_{naics}_timeseries.csv",
        mime="text/csv",
    )
