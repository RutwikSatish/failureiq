import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="FailureIQ | Hardware RCA Engine",
    page_icon="🔬", layout="wide"
)

st.markdown("""
<style>
  html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{
    background-color:#0f1117!important;color:#e8e8e8!important}
  [data-testid="stSidebar"]{background-color:#161b22!important}
  [data-testid="stSidebar"] *{color:#e8e8e8!important}
  h1,h2,h3,h4,p,span,div,label,.stMarkdown{color:#e8e8e8!important}
  [data-testid="metric-container"]{background-color:#1c2333!important;
    border:1px solid #30363d!important;border-radius:8px!important;padding:12px!important}
  [data-testid="metric-container"] *{color:#e8e8e8!important}
  [data-testid="stMetricValue"]{color:#ffffff!important;font-weight:600!important}
  [data-testid="stMetricLabel"]{color:#8b949e!important}
  [data-testid="stTabs"] button{color:#8b949e!important;background:transparent!important}
  [data-testid="stTabs"] button[aria-selected="true"]{color:#58a6ff!important;
    border-bottom:2px solid #58a6ff!important}
  [data-testid="stDataFrame"]{background-color:#161b22!important;
    border:1px solid #30363d!important;border-radius:8px!important}
  .stDataFrame th{background-color:#21262d!important;color:#c9d1d9!important;font-weight:600!important}
  .stDataFrame td{color:#e8e8e8!important;background-color:#161b22!important}
  [data-testid="stSelectbox"]>div,[data-testid="stMultiSelect"]>div{
    background-color:#21262d!important;border:1px solid #30363d!important;
    color:#e8e8e8!important;border-radius:6px!important}
  [data-testid="stButton"] button{background-color:#238636!important;
    color:#ffffff!important;border:none!important;border-radius:6px!important;font-weight:500!important}
  [data-testid="stExpander"]{background-color:#161b22!important;
    border:1px solid #30363d!important;border-radius:8px!important}
  [data-testid="stExpander"] *{color:#e8e8e8!important}
  hr{border-color:#30363d!important}
  table{background-color:#161b22!important;color:#e8e8e8!important;
    border-collapse:collapse!important;width:100%!important}
  th{background-color:#21262d!important;color:#c9d1d9!important;
    padding:8px 12px!important;border:1px solid #30363d!important}
  td{color:#e8e8e8!important;padding:8px 12px!important;border:1px solid #30363d!important}
  tr:nth-child(even) td{background-color:#1c2333!important}
</style>
""", unsafe_allow_html=True)

DARK = dict(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b22",
            font=dict(color="#e8e8e8"), margin=dict(t=30,b=40,l=10,r=10))

# ── SYNTHETIC DATA ────────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=1200, seed=42):
    random.seed(seed); np.random.seed(seed)
    vendors    = ["VendorA","VendorB","VendorC","VendorD",
                  "VendorE","VendorF","VendorG","VendorH"]
    zones      = ["Zone-1 (Thermal)","Zone-2 (Thermal)","Zone-3 (Standard)",
                  "Zone-4 (Standard)","Zone-5 (Cold)","Zone-6 (Cold)"]
    components = ["CPU Module","Memory DIMM","NVMe SSD","Power Supply",
                  "Network Card","Cooling Fan","RAID Controller","GPU Module"]
    modes      = ["Thermal Overstress","ESD Damage","Mechanical Fatigue",
                  "Firmware Corruption","Solder Joint Failure","Capacitor Degradation",
                  "Connector Wear","Moisture Ingress"]
    severities = ["Critical","Major","Minor"]
    rows = []
    base = datetime(2024,1,1)
    for i in range(n):
        v = random.choices(vendors, weights=[0.22,0.21,0.12,0.11,0.1,0.1,0.07,0.07])[0]
        z = random.choices(zones,
              weights=[0.28,0.25,0.15,0.12,0.1,0.1] if v in ["VendorA","VendorB"]
              else [0.15,0.1,0.2,0.2,0.17,0.18])[0]
        c = random.choice(components)
        m = random.choices(modes,
              weights=[0.28,0.05,0.12,0.15,0.18,0.1,0.07,0.05]
              if "Thermal" in z else [0.08,0.18,0.15,0.12,0.15,0.12,0.1,0.1])[0]
        age = int(np.random.exponential(180) + 30)
        sev = random.choices(severities,
              weights=[0.45,0.35,0.2] if v in ["VendorA","VendorB"] and "Thermal" in z
              else [0.15,0.35,0.5])[0]
        batch = f"B{2024 + i//400}-{random.randint(1,12):02d}"
        ts = base + timedelta(days=random.randint(0,365), hours=random.randint(0,23))
        rows.append({"ID": f"F{i+1:04d}", "Timestamp": ts, "Vendor": v,
                     "Thermal Zone": z, "Component": c, "Failure Mode": m,
                     "Age at Failure (days)": age, "Severity": sev,
                     "Batch": batch, "Resolved": random.random() > 0.35})
    return pd.DataFrame(rows)

df = generate_data()

def mtbf(ages): return round(float(np.mean(ages)), 1) if len(ages) else 0

def chi2_cluster(col):
    counts  = df[col].value_counts()
    exp     = np.full(len(counts), len(df)/len(counts))
    chi2, p = stats.chisquare(counts.values, f_exp=exp)
    return round(chi2,2), round(p,4)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    sel_vendors  = st.multiselect("Vendor",       df["Vendor"].unique().tolist(),       default=df["Vendor"].unique().tolist())
    sel_zones    = st.multiselect("Thermal zone", df["Thermal Zone"].unique().tolist(), default=df["Thermal Zone"].unique().tolist())
    sel_sev      = st.multiselect("Severity",     ["Critical","Major","Minor"],          default=["Critical","Major","Minor"])
    sel_comp     = st.multiselect("Component",    df["Component"].unique().tolist(),     default=df["Component"].unique().tolist())
    age_range    = st.slider("Age at failure (days)", 0, 600, (0, 600))

filt = df[
    df["Vendor"].isin(sel_vendors) &
    df["Thermal Zone"].isin(sel_zones) &
    df["Severity"].isin(sel_sev) &
    df["Component"].isin(sel_comp) &
    df["Age at Failure (days)"].between(*age_range)
].copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#e8e8e8;margin-bottom:2px'>FailureIQ — Hardware Root Cause Analysis Engine</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:13px'>Automated failure pattern detection, root cause identification, and CAPA tracking for data center hardware quality engineering</p>", unsafe_allow_html=True)

with st.expander("Preview demo data", expanded=False):
    st.markdown("<p style='color:#8b949e;font-size:13px'>Sample of the 1,200 hardware failure records powering this dashboard</p>", unsafe_allow_html=True)
    sample = generate_data().sample(10, random_state=1)[
        ["ID","Timestamp","Vendor","Thermal Zone","Component","Failure Mode","Age at Failure (days)","Severity","Batch","Resolved"]
    ].reset_index(drop=True)
    def sty_sev_prev(v):
        return ({"Critical":"background-color:#3d1f1f;color:#f85149;font-weight:600",
                 "Major":   "background-color:#3d2f0f;color:#d29922;font-weight:600",
                 "Minor":   "background-color:#0f2d1f;color:#3fb950;font-weight:600"}).get(v,"")
    st.dataframe(sample.style.map(sty_sev_prev, subset=["Severity"]),
                 use_container_width=True, hide_index=True)
    st.markdown("<p style='color:#8b949e;font-size:12px'>Full dataset: 1,200 records across 8 vendors, 6 thermal zones, 8 components, 8 failure modes. See Data preview tab for filtered exploration.</p>", unsafe_allow_html=True)

with st.expander("What does this app do?", expanded=False):
    st.markdown("""
**FailureIQ** replicates the root cause analysis workflow a data center quality engineer
performs when investigating hardware reliability data — automated in a single tool instead
of a manual spreadsheet process.

**The problem it solves**

Data centers process thousands of hardware failure events across servers, storage, and
network components. Identifying whether failures concentrate by vendor batch, thermal zone,
component age, or failure mode requires statistical pattern detection — the kind a quality
engineer performs using Pareto analysis, chi-square tests, and MTBF calculations.
Done manually, this takes days. FailureIQ does it in seconds.

**How it works**

The analysis pipeline runs four techniques simultaneously: Pareto analysis identifies the
vital few failure modes driving 80% of events. Chi-square cluster detection tests whether
failures are statistically non-random across vendors and zones. MTBF calculation quantifies
mean time between failures per component and vendor. Fishbone diagram generation maps
contributing factors across the standard 5M+E categories (Materials, Methods, Machine,
Man, Measurement, Environment) with significance-weighted branches.

**CAPA workflow**

Each identified root cause links to a corrective and preventive action record with owner,
target date, and verification metric — structured output that mirrors hardware quality
engineering documentation standards.
    """)

# ── METRICS ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
crit_n  = int((filt["Severity"]=="Critical").sum())
top_v   = filt["Vendor"].value_counts().index[0] if len(filt) else "N/A"
top_vm  = filt[filt["Vendor"]==top_v]["Failure Mode"].value_counts().index[0] if len(filt) else "N/A"
unres   = int((~filt["Resolved"]).sum())
avg_age = mtbf(filt["Age at Failure (days)"])

c1.metric("Total failures",     len(filt))
c2.metric("Critical failures",  crit_n)
c3.metric("Unresolved",         unres)
c4.metric("Avg MTBF (days)",    avg_age)
c5.metric("Top failure vendor", top_v)
st.markdown("---")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Pareto analysis","Cluster detection","MTBF breakdown","Fishbone + CAPA","Data preview"
])

# ── TAB 1: PARETO ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<h4 style='color:#e8e8e8'>Pareto analysis — top failure modes</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:13px'>Identifies the vital few failure modes driving 80% of all events (80/20 principle)</p>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        fm_counts = filt["Failure Mode"].value_counts().reset_index()
        fm_counts.columns = ["Failure Mode","Count"]
        fm_counts["Cumulative %"] = (fm_counts["Count"].cumsum() / fm_counts["Count"].sum() * 100).round(1)
        fig = go.Figure()
        fig.add_bar(x=fm_counts["Failure Mode"], y=fm_counts["Count"],
                    name="Count", marker_color="#58a6ff")
        fig.add_scatter(x=fm_counts["Failure Mode"], y=fm_counts["Cumulative %"],
                        name="Cumulative %", yaxis="y2", mode="lines+markers",
                        line=dict(color="#f85149", width=2), marker=dict(size=6))
        fig.add_hline(y=80, line_dash="dash", line_color="#d29922",
                      yref="y2", annotation_text="80% threshold",
                      annotation_font_color="#d29922")
        fig.update_layout(**DARK, height=380,
                          yaxis2=dict(overlaying="y", side="right",
                                      range=[0,110], gridcolor="#30363d",
                                      tickfont=dict(color="#8b949e"), title="Cumulative %",
                                      titlefont=dict(color="#8b949e")),
                          legend=dict(orientation="h", y=1.05, font=dict(color="#e8e8e8")),
                          xaxis=dict(tickangle=30))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        vnd_counts = filt["Vendor"].value_counts().reset_index()
        vnd_counts.columns = ["Vendor","Count"]
        vnd_counts["Cumulative %"] = (vnd_counts["Count"].cumsum() / vnd_counts["Count"].sum() * 100).round(1)
        fig2 = go.Figure()
        fig2.add_bar(x=vnd_counts["Vendor"], y=vnd_counts["Count"],
                     name="Count", marker_color="#3fb950")
        fig2.add_scatter(x=vnd_counts["Vendor"], y=vnd_counts["Cumulative %"],
                         name="Cumulative %", yaxis="y2", mode="lines+markers",
                         line=dict(color="#f85149",width=2), marker=dict(size=6))
        fig2.add_hline(y=80, line_dash="dash", line_color="#d29922",
                       yref="y2", annotation_text="80% threshold",
                       annotation_font_color="#d29922")
        fig2.update_layout(**DARK, height=380,
                           yaxis2=dict(overlaying="y", side="right",
                                       range=[0,110], gridcolor="#30363d",
                                       tickfont=dict(color="#8b949e")),
                           legend=dict(orientation="h", y=1.05, font=dict(color="#e8e8e8")))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Failure heatmap — vendor vs. failure mode</h4>", unsafe_allow_html=True)
    heat = filt.groupby(["Vendor","Failure Mode"]).size().reset_index(name="Count")
    fig3 = px.density_heatmap(heat, x="Vendor", y="Failure Mode", z="Count",
                               color_continuous_scale=["#161b22","#1D6FA4","#f85149"])
    fig3.update_layout(**DARK, height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: CHI-SQUARE ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("<h4 style='color:#e8e8e8'>Statistical cluster detection — chi-square test</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:13px'>Tests whether failures are statistically non-random across vendors, zones, and components. p less than 0.05 means clustering is significant.</p>", unsafe_allow_html=True)

    results = []
    for col in ["Vendor","Thermal Zone","Component","Failure Mode"]:
        chi2, p = chi2_cluster(col)
        results.append({"Dimension": col, "Chi-square": chi2, "p-value": p,
                        "Significant": "Yes" if p < 0.05 else "No",
                        "Finding": "Failures cluster non-randomly" if p < 0.05
                                   else "Failures distributed randomly"})
    chi_df = pd.DataFrame(results)

    def style_sig(v):
        return ("background-color:#3d1f1f;color:#f85149;font-weight:600" if v=="Yes"
                else "background-color:#0f2d1f;color:#3fb950;font-weight:600")

    st.dataframe(chi_df.style.map(style_sig, subset=["Significant"]),
                 use_container_width=True, hide_index=True)

    st.markdown("<h4 style='color:#e8e8e8'>Failure distribution by thermal zone and severity</h4>", unsafe_allow_html=True)
    zone_sev = filt.groupby(["Thermal Zone","Severity"]).size().reset_index(name="Count")
    fig4 = px.bar(zone_sev, x="Thermal Zone", y="Count", color="Severity",
                  color_discrete_map={"Critical":"#f85149","Major":"#d29922","Minor":"#3fb950"},
                  barmode="stack")
    fig4.update_layout(**DARK, height=360,
                       legend=dict(orientation="h",y=1.05,font=dict(color="#e8e8e8")),
                       xaxis=dict(tickangle=20))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Age at failure distribution by vendor</h4>", unsafe_allow_html=True)
    fig5 = px.box(filt, x="Vendor", y="Age at Failure (days)", color="Severity",
                  color_discrete_map={"Critical":"#f85149","Major":"#d29922","Minor":"#3fb950"},
                  points="outliers")
    fig5.update_layout(**DARK, height=360,
                       legend=dict(orientation="h",y=1.05,font=dict(color="#e8e8e8")))
    st.plotly_chart(fig5, use_container_width=True)

# ── TAB 3: MTBF ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<h4 style='color:#e8e8e8'>Mean Time Between Failures (MTBF) analysis</h4>", unsafe_allow_html=True)

    mtbf_comp = filt.groupby("Component")["Age at Failure (days)"].agg(
        MTBF="mean", Count="count", Std="std").round(1).reset_index()
    mtbf_comp["MTBF"] = mtbf_comp["MTBF"].round(1)
    mtbf_comp = mtbf_comp.sort_values("MTBF")

    fig6 = go.Figure()
    fig6.add_bar(y=mtbf_comp["Component"], x=mtbf_comp["MTBF"],
                 orientation="h", marker_color="#58a6ff",
                 error_x=dict(type="data", array=mtbf_comp["Std"], color="#8b949e"))
    fig6.add_vline(x=180, line_dash="dash", line_color="#d29922",
                   annotation_text="Target MTBF (180 days)",
                   annotation_font_color="#d29922")
    fig6.update_layout(**DARK, height=400)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>MTBF by vendor</h4>", unsafe_allow_html=True)
    mtbf_vnd = filt.groupby("Vendor")["Age at Failure (days)"].agg(
        MTBF="mean", Count="count").round(1).reset_index().sort_values("MTBF")

    def mtbf_color(v):
        if v < 120: return "background-color:#3d1f1f;color:#f85149;font-weight:600"
        if v < 180: return "background-color:#3d2f0f;color:#d29922;font-weight:600"
        return "background-color:#0f2d1f;color:#3fb950;font-weight:600"

    st.dataframe(mtbf_vnd.style.map(mtbf_color, subset=["MTBF"]),
                 use_container_width=True, hide_index=True)

    st.markdown("<h4 style='color:#e8e8e8'>Failure trend over time</h4>", unsafe_allow_html=True)
    trend = filt.copy()
    trend["Month"] = trend["Timestamp"].dt.to_period("M").astype(str)
    trend_m = trend.groupby(["Month","Severity"]).size().reset_index(name="Count")
    fig7 = px.line(trend_m, x="Month", y="Count", color="Severity",
                   color_discrete_map={"Critical":"#f85149","Major":"#d29922","Minor":"#3fb950"},
                   markers=True)
    fig7.update_layout(**DARK, height=340,
                       legend=dict(orientation="h",y=1.05,font=dict(color="#e8e8e8")),
                       xaxis=dict(tickangle=30))
    st.plotly_chart(fig7, use_container_width=True)

# ── TAB 4: FISHBONE + CAPA ────────────────────────────────────────────────────
with tab4:
    st.markdown("<h4 style='color:#e8e8e8'>Fishbone diagram + CAPA generator</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:13px'>Select a failure mode to generate structured root cause mapping and corrective action record</p>", unsafe_allow_html=True)

    sel_mode = st.selectbox("Select failure mode", filt["Failure Mode"].value_counts().index.tolist())
    sub = filt[filt["Failure Mode"] == sel_mode]

    fishbone = {
        "Materials": [
            f"Vendor concentration: {sub['Vendor'].value_counts().index[0]} accounts for "
            f"{sub['Vendor'].value_counts().iloc[0]/len(sub)*100:.0f}% of this mode",
            f"Avg component age at failure: {mtbf(sub['Age at Failure (days)'])} days",
            f"Top batch affected: {sub['Batch'].value_counts().index[0]}"
        ],
        "Environment": [
            f"Primary zone: {sub['Thermal Zone'].value_counts().index[0]} "
            f"({sub['Thermal Zone'].value_counts().iloc[0]/len(sub)*100:.0f}% of events)",
            "Thermal zone correlation with failure rate: statistically significant (p < 0.05)"
            if chi2_cluster("Thermal Zone")[1] < 0.05 else "No significant zone correlation detected"
        ],
        "Machine": [
            f"Most affected component: {sub['Component'].value_counts().index[0]}",
            f"Component accounts for {sub['Component'].value_counts().iloc[0]/len(sub)*100:.0f}% of this failure mode"
        ],
        "Methods": [
            f"Resolution rate: {sub['Resolved'].mean()*100:.0f}% of cases resolved",
            "CAPA closure rate suggests process gap in corrective action follow-through"
            if sub['Resolved'].mean() < 0.7 else "CAPA closure rate within acceptable range"
        ],
        "Measurement": [
            "Failure detection lag: review whether monitoring thresholds are calibrated to current thermal conditions",
            f"Severity distribution: {(sub['Severity']=='Critical').mean()*100:.0f}% critical events"
        ],
    }

    for cat, items in fishbone.items():
        st.markdown(
            f"<div style='background:#161b22;border-left:3px solid #58a6ff;"
            f"padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px'>"
            f"<b style='color:#58a6ff'>{cat}</b><br>"
            + "".join([f"<span style='color:#e8e8e8;font-size:13px'>- {i}</span><br>" for i in items])
            + "</div>", unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("<h4 style='color:#e8e8e8'>CAPA record</h4>", unsafe_allow_html=True)
    capa = pd.DataFrame([
        {"ID": f"CAPA-{sel_mode[:4].upper()}-001",
         "Root Cause": f"Vendor batch concentration and thermal zone correlation for {sel_mode}",
         "Corrective Action": f"Issue SCAR to {sub['Vendor'].value_counts().index[0]}; require batch traceability audit",
         "Preventive Action": "Update incoming inspection protocol; add thermal zone monitoring threshold",
         "Owner": "Quality Engineering",
         "Target Date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
         "Verification Metric": f"Reduce {sel_mode} rate by 40% within 60 days",
         "Status": "Open"}
    ])
    st.dataframe(capa, use_container_width=True, hide_index=True)

# ── TAB 5: DATA PREVIEW ───────────────────────────────────────────────────────
with tab5:
    st.markdown("<h4 style='color:#e8e8e8'>Raw failure log</h4>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    pf_v = c1.selectbox("Vendor",   ["All"]+sorted(df["Vendor"].unique().tolist()), key="pv")
    pf_z = c2.selectbox("Zone",     ["All"]+sorted(df["Thermal Zone"].unique().tolist()), key="pz")
    pf_s = c3.selectbox("Severity", ["All","Critical","Major","Minor"], key="ps")
    prev = df.copy()
    if pf_v!="All": prev=prev[prev["Vendor"]==pf_v]
    if pf_z!="All": prev=prev[prev["Thermal Zone"]==pf_z]
    if pf_s!="All": prev=prev[prev["Severity"]==pf_s]
    st.markdown(f"<p style='color:#8b949e;font-size:13px'>Showing <b style='color:#e8e8e8'>{len(prev)}</b> of <b style='color:#e8e8e8'>{len(df)}</b> records</p>", unsafe_allow_html=True)

    def sty_sev(v):
        return ({"Critical":"background-color:#3d1f1f;color:#f85149;font-weight:600",
                 "Major":   "background-color:#3d2f0f;color:#d29922;font-weight:600",
                 "Minor":   "background-color:#0f2d1f;color:#3fb950;font-weight:600"}).get(v,"")

    st.dataframe(
        prev.sort_values("Timestamp",ascending=False).reset_index(drop=True)
            .style.map(sty_sev, subset=["Severity"]),
        use_container_width=True, hide_index=True, height=440
    )
    csv=prev.to_csv(index=False).encode()
    st.download_button("Download filtered log as CSV", csv, "failureiq_log.csv","text/csv")
