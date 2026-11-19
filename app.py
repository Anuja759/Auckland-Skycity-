"""
Order Channel Performance & Market Share Analytics
SkyCity Auckland Restaurants & Bars  |  Streamlit dashboard

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SkyCity Auckland | Channel Analytics",
    page_icon="🍽️",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "SkyCity_Auckland_Restaurants___Bars.csv"

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
ORD = {
    "In-Store": "InStoreOrders",
    "Uber Eats": "UberEatsOrders",
    "DoorDash": "DoorDashOrders",
    "Self-Delivery": "SelfDeliveryOrders",
}
REV = {
    "In-Store": "InStoreRevenue",
    "Uber Eats": "UberEatsRevenue",
    "DoorDash": "DoorDashRevenue",
    "Self-Delivery": "SelfDeliveryRevenue",
}
COLORS = {
    "In-Store": "#2E7D32",
    "Uber Eats": "#06C167",
    "DoorDash": "#EB1700",
    "Self-Delivery": "#1F77B4",
    "Delivery": "#7B61FF",
}
# Uber Eats green is too close to In-Store green, so tune In-Store darker/teal
COLORS["In-Store"] = "#0B3C5D"
RISK_COLORS = {"Balanced": "#2E9E6B", "Watch": "#F2A93B", "High risk": "#D64545"}
MODES = ["All channels", "In-Store vs Delivery", "Delivery channels only"]


# ----------------------------------------------------------------------------
# Data loading & feature engineering
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    o = df[list(ORD.values())]
    total = df["MonthlyOrders"]

    df["DeliveryOrders"] = df["UberEatsOrders"] + df["DoorDashOrders"] + df["SelfDeliveryOrders"]
    df["AggregatorOrders"] = df["UberEatsOrders"] + df["DoorDashOrders"]
    df["TotalRevenue"] = df[list(REV.values())].sum(axis=1)

    # KPIs (all computed from order counts so they reconcile to MonthlyOrders)
    df["InStoreReliance"] = df["InStoreOrders"] / total
    df["AggDependence"] = df["AggregatorOrders"] / total
    df["MaxAggShare"] = df[["UberEatsOrders", "DoorDashOrders"]].max(axis=1) / total
    df["TopAggregator"] = np.where(df["UberEatsOrders"] >= df["DoorDashOrders"], "Uber Eats", "DoorDash")

    p = o.div(o.sum(axis=1), axis=0)
    entropy = -(p * np.log(p.where(p > 0, 1))).sum(axis=1)
    df["DiversificationScore"] = entropy / np.log(4) * 100  # 0-100, 100 = perfectly even
    return df


def load_data() -> pd.DataFrame | None:
    up = st.sidebar.file_uploader("Upload a different CSV (optional)", type="csv")
    if up is not None:
        return pd.read_csv(up)
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    st.error(f"Data file not found: {DATA_FILE.name}. Place it next to app.py or upload it in the sidebar.")
    return None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def mode_map(mode: str, base: dict) -> dict:
    if mode == "All channels":
        return {k: [v] for k, v in base.items()}
    if mode == "Delivery channels only":
        return {k: [v] for k, v in base.items() if k != "In-Store"}
    return {
        "In-Store": [base["In-Store"]],
        "Delivery": [base["Uber Eats"], base["DoorDash"], base["Self-Delivery"]],
    }


def channel_frame(d: pd.DataFrame, mode: str, base: dict = ORD) -> pd.DataFrame:
    return pd.DataFrame({k: d[v].sum(axis=1) for k, v in mode_map(mode, base).items()})


def grouped(d: pd.DataFrame, by: str, mode: str, base: dict = ORD) -> pd.DataFrame:
    return channel_frame(d, mode, base).groupby(d[by]).sum()


def pct(t: pd.DataFrame) -> pd.DataFrame:
    return t.div(t.sum(axis=1), axis=0) * 100


def heatmap(share: pd.DataFrame, mode_view: str, title: str, market: pd.Series):
    if mode_view.startswith("Index"):
        z = share.div(market, axis=1) * 100
        fig = px.imshow(
            z.round(0), text_auto=".0f", aspect="auto",
            color_continuous_scale="RdBu_r", color_continuous_midpoint=100,
            labels=dict(color="Index"), title=title + " (100 = market average)",
        )
    else:
        fig = px.imshow(
            share.round(1), text_auto=".1f", aspect="auto",
            color_continuous_scale="Blues", labels=dict(color="% of orders"),
            title=title + " (% of orders)",
        )
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(margin=dict(t=80, b=10), height=max(260, 70 * len(share) + 110))
    return fig


def classify(d: pd.DataFrame, single_thr: float, comb_thr: float, watch_thr: float) -> pd.Series:
    high = (d["MaxAggShare"] >= single_thr) | (d["AggDependence"] >= comb_thr)
    watch = ~high & (d["AggDependence"] >= watch_thr)
    return pd.Series(np.where(high, "High risk", np.where(watch, "Watch", "Balanced")), index=d.index)


def fmt_int(x):
    return f"{int(round(x)):,}"


# ----------------------------------------------------------------------------
# Load + sidebar filters
# ----------------------------------------------------------------------------
st.title("🍽️ Order Channel Performance & Market Share")
st.caption("SkyCity Auckland Restaurants & Bars · channel mix, geography, cuisine and dependency risk")

raw = load_data()
if raw is None:
    st.stop()
full = prepare(raw)

st.sidebar.header("Filters")
sub_sel = st.sidebar.multiselect("Subregion", sorted(full["Subregion"].unique()), default=sorted(full["Subregion"].unique()))
cui_sel = st.sidebar.multiselect("Cuisine type", sorted(full["CuisineType"].unique()), default=sorted(full["CuisineType"].unique()))
seg_sel = st.sidebar.multiselect("Segment", sorted(full["Segment"].unique()), default=sorted(full["Segment"].unique()))
mode = st.sidebar.radio(
    "Channel view", MODES,
    help="All channels: 4 channels. In-Store vs Delivery: the three delivery channels are combined. "
         "Delivery channels only: shares are calculated within delivery orders.",
)

with st.sidebar.expander("Dependency risk thresholds", expanded=False):
    single_thr = st.slider("Single-aggregator reliance ≥ (% of orders)", 30, 90, 70, 1) / 100
    comb_thr = st.slider("Combined aggregator (UE + DD) ≥ → High risk", 40, 90, 70, 1) / 100
    watch_thr = st.slider("Combined aggregator ≥ → Watch", 30, 85, 60, 1) / 100
    if watch_thr > comb_thr:
        st.warning("Watch threshold is above the High-risk threshold; Watch will be empty.")

d = full[
    full["Subregion"].isin(sub_sel) & full["CuisineType"].isin(cui_sel) & full["Segment"].isin(seg_sel)
].copy()
if d.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()
d["RiskProfile"] = classify(d, single_thr, comb_thr, watch_thr)

st.sidebar.markdown("---")
st.sidebar.metric("Branches in selection", f"{len(d):,} / {len(full):,}")

tab_over, tab_geo, tab_cui, tab_risk, tab_dq = st.tabs(
    ["📊 Channel Mix Overview", "🗺️ Geography", "🍜 Cuisine & Segment", "⚠️ Dependency Risk", "✅ Data Quality"]
)

# ----------------------------------------------------------------------------
# TAB 1: Overview
# ----------------------------------------------------------------------------
with tab_over:
    tot_orders = d["MonthlyOrders"].sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Monthly orders", fmt_int(tot_orders))
    k2.metric("In-Store reliance ratio", f"{d['InStoreOrders'].sum() / tot_orders:.1%}")
    k3.metric("Aggregator dependence index", f"{d['AggregatorOrders'].sum() / tot_orders:.1%}",
              help="(Uber Eats + DoorDash orders) / all orders")
    k4.metric("Avg diversification score", f"{d['DiversificationScore'].mean():.1f} / 100",
              help="Normalised Shannon entropy of each branch's 4-channel order mix. 100 = perfectly even.")
    k5.metric("Total revenue", f"${d['TotalRevenue'].sum() / 1e6:,.2f}M")

    orders = channel_frame(d, mode).sum().rename("Orders")
    revenue = channel_frame(d, mode, REV).sum().rename("Revenue")
    rank = pd.concat([orders, revenue], axis=1)
    rank["Order share %"] = rank["Orders"] / rank["Orders"].sum() * 100
    rank["Revenue share %"] = rank["Revenue"] / rank["Revenue"].sum() * 100
    rank = rank.sort_values("Orders", ascending=False)
    rank.insert(0, "Rank", range(1, len(rank) + 1))

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(
            rank.reset_index(names="Channel"), x="Channel", y="Orders", color="Channel",
            color_discrete_map=COLORS, text="Orders", title="Order volume by channel",
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, template="plotly_white", yaxis_title="Monthly orders", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.pie(
            rank.reset_index(names="Channel"), names="Channel", values="Orders", hole=0.55,
            color="Channel", color_discrete_map=COLORS, title="Channel order share",
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Channel ranking")
    st.dataframe(
        rank.style.format({"Orders": "{:,.0f}", "Revenue": "${:,.0f}", "Order share %": "{:.1f}", "Revenue share %": "{:.1f}"}),
        width="stretch",
    )

    st.subheader("How much does channel share vary across branches?")
    ch = channel_frame(d, mode)
    sh = pct(ch).melt(var_name="Channel", value_name="Share of orders (%)")
    fig = px.box(sh, x="Channel", y="Share of orders (%)", color="Channel", color_discrete_map=COLORS, points="outliers")
    fig.update_layout(showlegend=False, template="plotly_white", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# TAB 2: Geography
# ----------------------------------------------------------------------------
with tab_geo:
    st.subheader("Subregion channel heatmap")
    view = st.radio("Heatmap values", ["% share of orders", "Index vs market average"], horizontal=True, key="geo_view")
    g_share = pct(grouped(d, "Subregion", mode))
    market = pct(channel_frame(d, mode).sum().to_frame().T).iloc[0]
    st.plotly_chart(heatmap(g_share, view, "Channel mix by subregion", market), width="stretch")

    # Dominance table
    dom = pd.DataFrame({
        "Dominant channel": g_share.idxmax(axis=1),
        "Dominant share %": g_share.max(axis=1),
        "Runner-up": g_share.apply(lambda r: r.nlargest(2).index[-1], axis=1),
        "Runner-up share %": g_share.apply(lambda r: r.nlargest(2).iloc[-1], axis=1),
        "Orders": grouped(d, "Subregion", mode).sum(axis=1),
    })
    dom["Lead (pp)"] = dom["Dominant share %"] - dom["Runner-up share %"]
    st.markdown("**Subregion channel dominance**")
    st.dataframe(
        dom.style.format({"Dominant share %": "{:.1f}", "Runner-up share %": "{:.1f}", "Lead (pp)": "{:.1f}", "Orders": "{:,.0f}"}),
        width="stretch",
    )

    if len(g_share) > 1:
        spread = (g_share.max() - g_share.min()).rename("Spread across subregions (pp)")
        st.info(
            "**Spread between highest and lowest subregion (percentage points):** "
            + " · ".join(f"{k}: {v:.1f}" for k, v in spread.items())
            + ". Small spreads mean the channel mix is similar everywhere; use the index view to see subtle differences."
        )

    cA, cB = st.columns(2)
    with cA:
        long = grouped(d, "Subregion", mode).reset_index().melt(id_vars="Subregion", var_name="Channel", value_name="Orders")
        fig = px.bar(long, x="Subregion", y="Orders", color="Channel", barmode="group",
                     color_discrete_map=COLORS, title="Order volume by subregion & channel")
        fig.update_layout(template="plotly_white", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with cB:
        long = g_share.reset_index().melt(id_vars="Subregion", var_name="Channel", value_name="Share %")
        fig = px.bar(long, x="Share %", y="Subregion", color="Channel", orientation="h",
                     color_discrete_map=COLORS, title="Channel mix by subregion (100% stacked)")
        fig.update_layout(template="plotly_white", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Urban vs suburban ordering behaviour")
    st.caption("Assumption: **CBD = urban**; North Shore, West Auckland and South Auckland = suburban. "
               "(The dataset has no 'Central Auckland' value, so CBD is used as the central/urban area.)")
    d["Area type"] = np.where(d["Subregion"] == "CBD", "Urban (CBD)", "Suburban")
    if d["Area type"].nunique() == 2:
        u = pct(grouped(d, "Area type", mode))
        long = u.reset_index().melt(id_vars="Area type", var_name="Channel", value_name="Share %")
        fig = px.bar(long, x="Channel", y="Share %", color="Area type", barmode="group",
                     color_discrete_sequence=["#0B3C5D", "#F2A93B"], text_auto=".1f")
        fig.update_layout(template="plotly_white", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe((u.loc["Urban (CBD)"] - u.loc["Suburban"]).rename("Urban minus suburban (pp)").to_frame().style.format("{:+.1f}"))
    else:
        st.info("Select CBD together with at least one suburban subregion to compare urban vs suburban.")

# ----------------------------------------------------------------------------
# TAB 3: Cuisine & Segment
# ----------------------------------------------------------------------------
with tab_cui:
    view_c = st.radio("Heatmap values", ["% share of orders", "Index vs market average"], horizontal=True, key="cui_view")
    market = pct(channel_frame(d, mode).sum().to_frame().T).iloc[0]

    st.subheader("Cuisine vs channel")
    c_share = pct(grouped(d, "CuisineType", mode))
    st.plotly_chart(heatmap(c_share, view_c, "Channel mix by cuisine", market), width="stretch")

    long = c_share.reset_index().melt(id_vars="CuisineType", var_name="Channel", value_name="Share %")
    fig = px.bar(long, x="Share %", y="CuisineType", color="Channel", orientation="h",
                 color_discrete_map=COLORS, title="Channel mix by cuisine (100% stacked)")
    fig.update_layout(template="plotly_white", yaxis_title=None, height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Aggregator-heavy cuisines")
    agg = d.groupby("CuisineType").agg(Agg=("AggregatorOrders", "sum"), Tot=("MonthlyOrders", "sum"),
                                       UE=("UberEatsOrders", "sum"), DD=("DoorDashOrders", "sum"))
    agg["Aggregator dependence %"] = agg["Agg"] / agg["Tot"] * 100
    agg["Uber Eats %"] = agg["UE"] / agg["Tot"] * 100
    agg["DoorDash %"] = agg["DD"] / agg["Tot"] * 100
    agg = agg.sort_values("Aggregator dependence %", ascending=True)
    overall_agg = d["AggregatorOrders"].sum() / d["MonthlyOrders"].sum() * 100
    fig = px.bar(agg.reset_index(), x="Aggregator dependence %", y="CuisineType", orientation="h",
                 text_auto=".1f", color_discrete_sequence=["#7B61FF"])
    fig.add_vline(x=overall_agg, line_dash="dash", annotation_text=f"Market avg {overall_agg:.1f}%")
    fig.update_layout(template="plotly_white", yaxis_title=None, height=380)
    st.plotly_chart(fig, width="stretch")
    top = agg["Aggregator dependence %"].idxmax()
    st.caption(f"Most aggregator-heavy cuisine in the current selection: **{top}** "
               f"({agg.loc[top, 'Aggregator dependence %']:.1f}% of orders via Uber Eats + DoorDash).")

    st.subheader("Segment vs channel")
    s_share = pct(grouped(d, "Segment", mode))
    cL, cR = st.columns([3, 2])
    with cL:
        st.plotly_chart(heatmap(s_share, view_c, "Channel mix by segment", market), width="stretch")
    with cR:
        s_rel = d.groupby("Segment").agg(
            InStore=("InStoreOrders", "sum"), Agg=("AggregatorOrders", "sum"), Tot=("MonthlyOrders", "sum"),
            Div=("DiversificationScore", "mean"),
        )
        s_tbl = pd.DataFrame({
            "In-Store reliance %": s_rel["InStore"] / s_rel["Tot"] * 100,
            "Aggregator dependence %": s_rel["Agg"] / s_rel["Tot"] * 100,
            "Avg diversification": s_rel["Div"],
        }).sort_values("In-Store reliance %", ascending=False)
        st.markdown("**Segment reliance**")
        st.dataframe(s_tbl.style.format("{:.1f}"), width="stretch")

    st.subheader("Aggregator dependence: segment × cuisine")
    pv = d.pivot_table(index="Segment", columns="CuisineType", values=["AggregatorOrders", "MonthlyOrders"], aggfunc="sum")
    mat = (pv["AggregatorOrders"] / pv["MonthlyOrders"] * 100)
    fig = px.imshow(mat.round(1), text_auto=".1f", aspect="auto", color_continuous_scale="Purples",
                    labels=dict(color="% via aggregators"))
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(height=320, margin=dict(t=70))
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# TAB 4: Dependency risk
# ----------------------------------------------------------------------------
with tab_risk:
    st.markdown(
        f"**Risk rules (adjust in the sidebar):** *High risk* = one aggregator ≥ **{single_thr:.0%}** of a branch's orders "
        f"**or** Uber Eats + DoorDash combined ≥ **{comb_thr:.0%}**. *Watch* = combined ≥ **{watch_thr:.0%}**. "
        "*Balanced* = everything else."
    )

    counts = d["RiskProfile"].value_counts().reindex(["Balanced", "Watch", "High risk"]).fillna(0).astype(int)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🟢 Balanced", f"{counts['Balanced']:,}", f"{counts['Balanced'] / len(d):.0%} of branches", delta_color="off")
    p2.metric("🟠 Watch", f"{counts['Watch']:,}", f"{counts['Watch'] / len(d):.0%} of branches", delta_color="off")
    p3.metric("🔴 High risk", f"{counts['High risk']:,}", f"{counts['High risk'] / len(d):.0%} of branches", delta_color="off")
    single_n = int((d["MaxAggShare"] >= single_thr).sum())
    p4.metric(f"Single aggregator ≥ {single_thr:.0%}", f"{single_n:,}",
              help="Branches where Uber Eats or DoorDash alone reaches the single-aggregator threshold.")

    if single_n == 0:
        st.warning(
            f"No branch relies on a **single** aggregator for ≥ {single_thr:.0%} of orders "
            f"(highest observed: {d['MaxAggShare'].max():.1%}, {d.loc[d['MaxAggShare'].idxmax(), 'TopAggregator']}). "
            "Dependency risk in this dataset comes from the **combined** weight of the two aggregators. "
            "Lower the single-aggregator slider in the sidebar to explore a stricter benchmark."
        )

    cA, cB = st.columns(2)
    with cA:
        fig = px.scatter(
            d, x="AggDependence", y="DiversificationScore", color="RiskProfile", size="MonthlyOrders",
            color_discrete_map=RISK_COLORS, category_orders={"RiskProfile": ["Balanced", "Watch", "High risk"]},
            hover_data=["RestaurantName", "Subregion", "CuisineType", "Segment"], opacity=0.6,
            labels={"AggDependence": "Aggregator dependence (share of orders)", "DiversificationScore": "Diversification score"},
            title="Dependence vs diversification",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    with cB:
        fig = px.histogram(d, x="AggDependence", nbins=40, color_discrete_sequence=["#7B61FF"],
                           title="Distribution of aggregator dependence", labels={"AggDependence": "Aggregator dependence"})
        fig.add_vline(x=watch_thr, line_dash="dot", line_color=RISK_COLORS["Watch"], annotation_text="Watch")
        fig.add_vline(x=comb_thr, line_dash="dash", line_color=RISK_COLORS["High risk"], annotation_text="High risk")
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(template="plotly_white", yaxis_title="Branches")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Where is the risk concentrated?")
    dim = st.radio("Break down by", ["Subregion", "CuisineType", "Segment"], horizontal=True, key="risk_dim")
    ct = pd.crosstab(d[dim], d["RiskProfile"], normalize="index").reindex(columns=["Balanced", "Watch", "High risk"], fill_value=0) * 100
    ct = ct.sort_values("High risk", ascending=True)
    long = ct.reset_index().melt(id_vars=dim, var_name="Profile", value_name="% of branches")
    fig = px.bar(long, x="% of branches", y=dim, color="Profile", orientation="h",
                 color_discrete_map=RISK_COLORS, category_orders={"Profile": ["Balanced", "Watch", "High risk"]})
    fig.update_layout(template="plotly_white", yaxis_title=None, height=max(280, 50 * len(ct) + 120))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Channel diversification benchmarks")
    q = d["DiversificationScore"].quantile([0.25, 0.5, 0.75])
    bm = pd.DataFrame({
        "Benchmark": ["Aggregator dependence (combined)", "Single-aggregator reliance", "In-Store reliance",
                      "Diversification score"],
        "Target": [f"< {watch_thr:.0%}", f"< {max(single_thr - 0.2, 0.3):.0%}", "≥ 20%", f"≥ {q[0.75]:.0f} (top quartile)"],
        "Selection median": [f"{d['AggDependence'].median():.1%}", f"{d['MaxAggShare'].median():.1%}",
                             f"{d['InStoreReliance'].median():.1%}", f"{q[0.5]:.1f}"],
        "Branches meeting target": [
            f"{(d['AggDependence'] < watch_thr).mean():.0%}",
            f"{(d['MaxAggShare'] < max(single_thr - 0.2, 0.3)).mean():.0%}",
            f"{(d['InStoreReliance'] >= 0.20).mean():.0%}",
            f"{(d['DiversificationScore'] >= q[0.75]).mean():.0%}",
        ],
    })
    st.dataframe(bm, hide_index=True, width="stretch")
    st.caption("Targets are illustrative planning benchmarks derived from the thresholds and distribution in the current selection; "
               "adjust them to suit your strategy.")

    st.subheader("Branch watch-list")
    lvl = st.multiselect("Show profiles", ["High risk", "Watch", "Balanced"], default=["High risk"])
    cols = ["RestaurantID", "RestaurantName", "Subregion", "CuisineType", "Segment", "MonthlyOrders",
            "InStoreReliance", "AggDependence", "MaxAggShare", "TopAggregator", "DiversificationScore", "RiskProfile"]
    wl = d[d["RiskProfile"].isin(lvl)][cols].sort_values("AggDependence", ascending=False)
    st.dataframe(
        wl.style.format({"InStoreReliance": "{:.1%}", "AggDependence": "{:.1%}", "MaxAggShare": "{:.1%}",
                         "DiversificationScore": "{:.1f}", "MonthlyOrders": "{:,}"}),
        hide_index=True, width="stretch", height=380,
    )
    st.download_button("⬇️ Download watch-list (CSV)", wl.to_csv(index=False).encode(), "channel_risk_watchlist.csv", "text/csv")

# ----------------------------------------------------------------------------
# TAB 5: Data quality  (always on the full dataset)
# ----------------------------------------------------------------------------
with tab_dq:
    st.subheader("Validation checks (full dataset)")
    f = full
    ord_sum = f[list(ORD.values())].sum(axis=1)
    deliv_share_sum = f["UE_share"] + f["DD_share"] + f["SD_share"]
    calc_instore = f["InStoreOrders"] / f["MonthlyOrders"]
    gap = (calc_instore - f["InStoreShare"]).abs()
    rev_ok = pd.concat(
        [((f[REV[k]] / f[ORD[k]]) / f["AOV"] - 1).abs() < 0.01 for k in ORD], axis=1
    ).all(axis=1)

    checks = pd.DataFrame([
        ("Channel orders sum to MonthlyOrders", int((ord_sum != f["MonthlyOrders"]).sum()), "✅ Pass" if (ord_sum == f["MonthlyOrders"]).all() else "❌ Fail"),
        ("Delivery shares (UE + DD + SD) sum to ~100%", int(((deliv_share_sum - 1).abs() > 0.01).sum()), "✅ Pass" if ((deliv_share_sum - 1).abs() <= 0.01).all() else "❌ Fail"),
        ("Revenue ≈ orders × AOV for every channel", int((~rev_ok).sum()), "✅ Pass" if rev_ok.all() else "❌ Fail"),
        ("Missing values", int(f.isna().sum().sum()), "✅ Pass" if f.isna().sum().sum() == 0 else "❌ Fail"),
        ("Duplicate RestaurantIDs", int(f["RestaurantID"].duplicated().sum()), "✅ Pass" if not f["RestaurantID"].duplicated().any() else "❌ Fail"),
        ("Stored InStoreShare within 2pp of computed share", int((gap > 0.02).sum()), "⚠️ Review" if (gap > 0.02).any() else "✅ Pass"),
    ], columns=["Check", "Rows failing", "Result"])
    st.dataframe(checks, hide_index=True, width="stretch")

    st.info(
        f"**Notes on the data.** The stored `InStoreShare` column differs from the share calculated from order counts "
        f"(mean gap {gap.mean() * 100:.1f}pp, max {gap.max() * 100:.1f}pp), so this dashboard computes every share from the order counts. "
        "`UE_share`, `DD_share` and `SD_share` are shares of *delivery* orders only (they sum to 100% without in-store). "
        "Subregions in the file are CBD, North Shore, South Auckland and West Auckland. "
        f"{f['RestaurantName'].nunique()} distinct restaurant names map to {f['RestaurantID'].nunique()} branch IDs, so each row is treated as one branch."
    )

    st.subheader("Outlier scan (IQR rule, 1.5×)")
    num_cols = ["MonthlyOrders", "AOV", "InStoreOrders", "UberEatsOrders", "DoorDashOrders", "SelfDeliveryOrders",
                "AggDependence", "DiversificationScore"]
    rows = []
    for c in num_cols:
        q1, q3 = f[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n = int(((f[c] < lo) | (f[c] > hi)).sum())
        rows.append((c, round(lo, 3), round(hi, 3), n, f"{n / len(f):.1%}"))
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Lower fence", "Upper fence", "Outliers", "% of rows"]),
                 hide_index=True, width="stretch")

    with st.expander("Preview filtered data"):
        st.dataframe(d.drop(columns=["RiskProfile"]).head(200), width="stretch")
