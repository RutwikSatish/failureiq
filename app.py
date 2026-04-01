import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="FailureIQ | Hardware RCA Engine",
                   page_icon="🔬", layout="wide")

st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],
[data-testid="block-container"]{background-color:#0f1117!important;color:#e8e8e8!important}
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
</style>""", unsafe_allow_html=True)

def apply_dark(fig, height=380, xangle=0, show_legend=True):
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e8e8e8"),
        height=height,
        margin=dict(t=30, b=50, l=10, r=10),
        showlegend=show_legend,
    )
    if show_legend:
        fig.update_layout(legend=dict(orientation="h", y=1.05,
                                      font=dict(color="#e8e8e8")))
    fig.update_xaxes(gridcolor="#30363d", linecolor="#30363d",
                     tickfont=dict(color="#8b949e"), tickangle=xangle)
    fig.update_yaxes(gridcolor="#30363d", linecolor="#30363d",
                     tickfont=dict(color="#8b949e"))
    return fig

@st.cache_data
def generate_data(n=1200, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    vendors    = ["VendorA","VendorB","VendorC","VendorD",
                  "VendorE","VendorF","VendorG","VendorH"]
    zones      = ["Zone-1 (Thermal)","Zone-2 (Thermal)","Zone-3 (Standard)",
                  "Zone-4 (Standard)","Zone-5 (Cold)","Zone-6 (Cold)"]
    components = ["CPU Module","Memory DIMM","NVMe SSD","Power Supply",
                  "Network Card","Cooling Fan","RAID Controller","GPU Module"]
    modes      = ["Thermal Overstress","ESD Damage","Mechanical Fatigue",
                  "Firmware Corruption","Solder Joint Failure","Capacitor Degradation",
                  "Connector Wear","Moisture Ingress"]
    rows = []
    base = datetime(2024, 1, 1)
    for i in range(n):
        v = random.choices(vendors, weights=[0.22,0.21,0.12,0.11,0.1,0.1,0.07,0.07])[0]
        z = random.choices(zones,
              weights=[0.28,0.25,0.15,0.12,0.1,0.1]
              if v in ["VendorA","VendorB"] else [0.15,0.1,0.2,0.2,0.17,0.18])[0]
        c = random.choice(components)
        m = random.choices(modes,
              weights=[0.28,0.05,0.12,0.15,0.18,0.1,0.07,0.05]
              if "Thermal" in z else [0.08,0.18,0.15,0.12,0.15,0.12,0.1,0.1])[0]
        age  = int(np.random.exponential(180) + 30)
        sev  = random.choices(["Critical","Major","Minor"],
               weights=[0.45,0.35,0.2]
               if v in ["VendorA","VendorB"] and "Thermal" in z
               else [0.15,0.35,0.5])[0]
        batch = f"B{2024 + i//400}-{random.randint(1,12):02d}"
        ts    = base + timedelta(days=random.randint(0,365),
                                  hours=random.randint(0,23))
        rows.append({"ID": f"F{i+1:04d}", "Timestamp": ts, "Vendor": v,
                     "Thermal Zone": z, "Component": c, "Failure Mode": m,
                     "Age at Failure (days)": age, "Severity": sev,
                     "Batch": batch, "Resolved": random.random() > 0.35})
    return pd.DataFrame(rows)

df = generate_data()

SEV_STYLE = {
    "Critical": "background-color:#3d1f1f;color:#f85149;font-weight:600",
    "Major":    "background-color:#3d2f0f;color:#d29922;font-weight:600",
    "Minor":    "background-color:#0f2d1f;color:#3fb950;font-weight:600",
}
def sty_sev(v): return SEV_STYLE.get(v, "")

def avg_age(s): return round(float(np.mean(s)), 1) if len(s) else 0

with st.sidebar:
    st.markdown("### Filters")
    sel_v  = st.multiselect("Vendor",        df["Vendor"].unique().tolist(),
                             default=df["Vendor"].unique().tolist())
    sel_z  = st.multiselect("Thermal zone",  df["Thermal Zone"].unique().tolist(),
                             default=df["Thermal Zone"].unique().tolist())
    sel_s  = st.multiselect("Severity",      ["Critical","Major","Minor"],
                             default=["Critical","Major","Minor"])
    sel_c  = st.multiselect("Component",     df["Component"].unique().tolist(),
                             default=df["Component"].unique().tolist())
    age_r  = st.slider("Age at failure (days)", 0, 600, (0, 600))

filt = df[
    df["Vendor"].isin(sel_v) &
    df["Thermal Zone"].isin(sel_z) &
    df["Severity"].isin(sel_s) &
    df["Component"].isin(sel_c) &
    df["Age at Failure (days)"].between(*age_r)
].copy()

st.markdown("<h2 style='color:#e8e8e8;margin-bottom:2px'>"
            "FailureIQ — Hardware Root Cause Analysis Engine</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:13px'>"
            "Automated failure pattern detection, root cause identification, "
            "and CAPA tracking for data center hardware quality engineering</p>",
            unsafe_allow_html=True)

with st.expander("Preview demo data", expanded=False):
    st.markdown("<p style='color:#8b949e;font-size:13px'>"
                "Sample of 10 records from the 1,200-event dataset</p>",
                unsafe_allow_html=True)
    s = df.sample(10, random_state=1)[
        ["ID","Timestamp","Vendor","Thermal Zone","Component",
         "Failure Mode","Age at Failure (days)","Severity","Resolved"]
    ].reset_index(drop=True)
    st.dataframe(s.style.map(sty_sev, subset=["Severity"]),
                 use_container_width=True, hide_index=True)

with st.expander("What does this app do?", expanded=False):
    st.markdown("""
**FailureIQ** automates the root cause analysis workflow for data center hardware failures.

- **Pareto analysis** identifies which failure modes and vendors drive 80% of events
- **Chi-square cluster detection** tests whether failures are statistically non-random
- **MTBF analysis** quantifies mean time between failures per component and vendor
- **Fishbone + CAPA** maps root causes and generates a corrective action record
    """)

c1,c2,c3,c4,c5 = st.columns(5)
top_v = filt["Vendor"].value_counts().index[0] if len(filt) else "N/A"
c1.metric("Total failures",     len(filt))
c2.metric("Critical",           int((filt["Severity"]=="Critical").sum()))
c3.metric("Unresolved",         int((~filt["Resolved"]).sum()))
c4.metric("Avg age (days)",     avg_age(filt["Age at Failure (days)"]))
c5.metric("Top failure vendor", top_v)
st.markdown("---")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Pareto analysis","Cluster detection",
    "MTBF breakdown","Fishbone + CAPA","Data preview"
])

with tab1:
    st.markdown("<h4 style='color:#e8e8e8'>Failure mode Pareto</h4>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fm = filt["Failure Mode"].value_counts().reset_index()
        fm.columns = ["Failure Mode","Count"]
        fm["Cum%"] = (fm["Count"].cumsum()/fm["Count"].sum()*100).round(1).astype(str)+"%"
        fig1 = go.Figure()
        fig1.add_bar(x=fm["Failure Mode"], y=fm["Count"],
                     marker_color="#58a6ff",
                     text=fm["Cum%"], textposition="outside",
                     textfont=dict(color="#8b949e", size=10))
        apply_dark(fig1, height=360, xangle=30, show_legend=False)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        vn = filt["Vendor"].value_counts().reset_index()
        vn.columns = ["Vendor","Count"]
        vn["Cum%"] = (vn["Count"].cumsum()/vn["Count"].sum()*100).round(1).astype(str)+"%"
        fig2 = go.Figure()
        fig2.add_bar(x=vn["Vendor"], y=vn["Count"],
                     marker_color="#3fb950",
                     text=vn["Cum%"], textposition="outside",
                     textfont=dict(color="#8b949e", size=10))
        apply_dark(fig2, height=360, show_legend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Vendor x failure mode heatmap</h4>",
                unsafe_allow_html=True)
    heat = filt.groupby(["Vendor","Failure Mode"]).size().reset_index(name="Count")
    fig3 = px.density_heatmap(heat, x="Vendor", y="Failure Mode", z="Count",
                               color_continuous_scale=["#161b22","#1D6FA4","#f85149"])
    apply_dark(fig3, height=320, show_legend=False)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("<h4 style='color:#e8e8e8'>Chi-square cluster detection</h4>",
                unsafe_allow_html=True)
    rows = []
    for col in ["Vendor","Thermal Zone","Component","Failure Mode"]:
        counts = filt[col].value_counts()
        exp    = np.full(len(counts), len(filt)/len(counts))
        chi2, p = stats.chisquare(counts.values, f_exp=exp)
        rows.append({"Dimension": col,
                     "Chi-square": round(chi2,2),
                     "p-value": round(p,4),
                     "Significant": "Yes" if p<0.05 else "No",
                     "Finding": "Non-random clustering" if p<0.05
                                else "Random distribution"})
    def sty_sig(v):
        return ("background-color:#3d1f1f;color:#f85149;font-weight:600"
                if v=="Yes" else "background-color:#0f2d1f;color:#3fb950;font-weight:600")
    st.dataframe(pd.DataFrame(rows).style.map(sty_sig, subset=["Significant"]),
                 use_container_width=True, hide_index=True)

    fig4 = px.bar(
        filt.groupby(["Thermal Zone","Severity"]).size().reset_index(name="Count"),
        x="Thermal Zone", y="Count", color="Severity", barmode="stack",
        color_discrete_map={"Critical":"#f85149","Major":"#d29922","Minor":"#3fb950"})
    apply_dark(fig4, height=340, xangle=20)
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.box(filt, x="Vendor", y="Age at Failure (days)", color="Severity",
                  color_discrete_map={"Critical":"#f85149","Major":"#d29922","Minor":"#3fb950"},
                  points="outliers")
    apply_dark(fig5, height=340)
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.markdown("<h4 style='color:#e8e8e8'>MTBF by component</h4>",
                unsafe_allow_html=True)
    mc = (filt.groupby("Component")["Age at Failure (days)"]
              .agg(MTBF="mean", Count="count", Std="std")
              .round(1).reset_index().sort_values("MTBF"))
    fig6 = go.Figure()
    fig6.add_bar(y=mc["Component"], x=mc["MTBF"], orientation="h",
                 marker_color="#58a6ff",
                 error_x=dict(type="data",
                              array=mc["Std"].fillna(0).tolist(),
                              color="#8b949e"))
    fig6.add_vline(x=180, line_dash="dash", line_color="#d29922",
                   annotation_text="Target 180 days",
                   annotation_font_color="#d29922")
    apply_dark(fig6, height=380, show_legend=False)
    st.plotly_chart(fig6, use_container_width=True)

    mv = (filt.groupby("Vendor")["Age at Failure (days)"]
              .agg(MTBF="mean", Count="count")
              .round(1).reset_index().sort_values("MTBF"))
    def mc_color(v):
        if v<120: return "background-color:#3d1f1f;color:#f85149;font-weight:600"
        if v<180: return "background-color:#3d2f0f;color:#d29922;font-weight:600"
        return "background-color:#0f2d1f;color:#3fb950;font-weight:600"
    st.dataframe(mv.style.map(mc_color, subset=["MTBF"]),
                 use_container_width=True, hide_index=True)

    tr = filt.copy()
    tr["Month"] = tr["Timestamp"].dt.to_period("M").astype(str)
    fig7 = px.line(tr.groupby(["Month","Severity"]).size().reset_index(name="Count"),
                   x="Month", y="Count", color="Severity", markers=True,
                   color_discrete_map={"Critical":"#f85149","Major":"#d29922","Minor":"#3fb950"})
    apply_dark(fig7, height=300, xangle=30)
    st.plotly_chart(fig7, use_container_width=True)

with tab4:
    st.markdown("<h4 style='color:#e8e8e8'>Fishbone + CAPA generator</h4>",
                unsafe_allow_html=True)
    sel_mode = st.selectbox("Select failure mode",
                            filt["Failure Mode"].value_counts().index.tolist())
    sub = filt[filt["Failure Mode"]==sel_mode]
    tv  = sub["Vendor"].value_counts().index[0]        if len(sub) else "N/A"
    tz  = sub["Thermal Zone"].value_counts().index[0]  if len(sub) else "N/A"
    tc  = sub["Component"].value_counts().index[0]     if len(sub) else "N/A"
    fishbone = {
        "Materials":   [f"Top vendor: {tv} ({sub['Vendor'].value_counts().iloc[0]/max(len(sub),1)*100:.0f}% of mode)",
                        f"Avg age: {avg_age(sub['Age at Failure (days)'])} days"],
        "Environment": [f"Primary zone: {tz} ({sub['Thermal Zone'].value_counts().iloc[0]/max(len(sub),1)*100:.0f}% of events)"],
        "Machine":     [f"Most affected component: {tc}"],
        "Methods":     [f"Resolution rate: {sub['Resolved'].mean()*100:.0f}%"],
        "Measurement": [f"Critical severity rate: {(sub['Severity']=='Critical').mean()*100:.0f}%"],
    }
    for cat, items in fishbone.items():
        st.markdown(
            f"<div style='background:#161b22;border-left:3px solid #58a6ff;"
            f"padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px'>"
            f"<b style='color:#58a6ff'>{cat}</b><br>"
            + "".join([f"<span style='color:#e8e8e8;font-size:13px'>- {i}</span><br>"
                       for i in items]) + "</div>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h4 style='color:#e8e8e8'>CAPA record</h4>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{
        "ID": f"CAPA-{sel_mode[:4].upper()}-001",
        "Root Cause": f"Vendor batch concentration for {sel_mode}",
        "Corrective Action": f"Issue SCAR to {tv}; require batch traceability audit",
        "Preventive Action": "Update incoming inspection; add thermal zone monitoring",
        "Owner": "Quality Engineering",
        "Target Date": (datetime.now()+timedelta(days=30)).strftime("%Y-%m-%d"),
        "Verification Metric": f"Reduce {sel_mode} rate by 40% within 60 days",
        "Status": "Open",
    }]), use_container_width=True, hide_index=True)

with tab5:
    st.markdown("<h4 style='color:#e8e8e8'>Raw failure log</h4>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    pv = c1.selectbox("Vendor",   ["All"]+sorted(df["Vendor"].unique()), key="pv")
    pz = c2.selectbox("Zone",     ["All"]+sorted(df["Thermal Zone"].unique()), key="pz")
    ps = c3.selectbox("Severity", ["All","Critical","Major","Minor"], key="ps")
    prev = df.copy()
    if pv!="All": prev=prev[prev["Vendor"]==pv]
    if pz!="All": prev=prev[prev["Thermal Zone"]==pz]
    if ps!="All": prev=prev[prev["Severity"]==ps]
    st.markdown(f"<p style='color:#8b949e;font-size:13px'>Showing "
                f"<b style='color:#e8e8e8'>{len(prev)}</b> / "
                f"<b style='color:#e8e8e8'>{len(df)}</b></p>",
                unsafe_allow_html=True)
    st.dataframe(
        prev.sort_values("Timestamp", ascending=False)
            .reset_index(drop=True)
            .style.map(sty_sev, subset=["Severity"]),
        use_container_width=True, hide_index=True, height=440)
    st.download_button("Download as CSV",
                       prev.to_csv(index=False).encode(),
                       "failureiq_log.csv", "text/csv")
