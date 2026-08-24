import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

# -----------------------------------------------------------------------------
# 1. Page Configuration & Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PGA Tour Championship 30-Player GPP Optimizer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⛳ PGA Tour Championship GPP Optimizer (30-Player No-Cut)")
st.markdown("""
Custom **Mixed Integer Linear Programming (MILP)** optimization engine engineered specifically for 
**East Lake Golf Club** and the **30-player FedEx Cup finale**. Features **Roster Architecture Presets** 
(Balanced 6-Pack, Single Anchor, Stars & Scrubs), **Unspent Salary Uniqueness Arbitrage**, and **Strokes Gained Course-Fit Weighting**.
""")

# -----------------------------------------------------------------------------
# 2. Built-In 30-Player Tour Championship Slate
# -----------------------------------------------------------------------------
def get_tour_championship_slate():
    """Returns the full 30-player field with starting strokes, SG metrics, and base projections."""
    data = [
        {"Name": "Scottie Scheffler", "Salary": 12500, "StartingStrokes": -10, "BaseProjection": 122.5, "Ownership": 44.5, "SG_OTT": 1.15, "SG_APP": 1.42, "SG_P": 0.25, "BoB_Pct": 26.8, "ID": "300001"},
        {"Name": "Xander Schauffele", "Salary": 11600, "StartingStrokes": -8, "BaseProjection": 114.0, "Ownership": 38.0, "SG_OTT": 0.85, "SG_APP": 1.05, "SG_P": 0.65, "BoB_Pct": 25.4, "ID": "300002"},
        {"Name": "Rory McIlroy", "Salary": 10800, "StartingStrokes": -4, "BaseProjection": 102.5, "Ownership": 28.5, "SG_OTT": 1.25, "SG_APP": 0.75, "SG_P": 0.35, "BoB_Pct": 25.1, "ID": "300003"},
        {"Name": "Collin Morikawa", "Salary": 10200, "StartingStrokes": -4, "BaseProjection": 98.0, "Ownership": 26.0, "SG_OTT": 0.55, "SG_APP": 1.20, "SG_P": 0.15, "BoB_Pct": 23.9, "ID": "300004"},
        {"Name": "Ludvig Aberg", "Salary": 9800, "StartingStrokes": -5, "BaseProjection": 95.5, "Ownership": 24.5, "SG_OTT": 0.95, "SG_APP": 0.80, "SG_P": 0.20, "BoB_Pct": 24.2, "ID": "300005"},
        {"Name": "Hideki Matsuyama", "Salary": 9600, "StartingStrokes": -7, "BaseProjection": 96.0, "Ownership": 27.0, "SG_OTT": 0.45, "SG_APP": 1.10, "SG_P": -0.10, "BoB_Pct": 23.5, "ID": "300006"},
        {"Name": "Wyndham Clark", "Salary": 9200, "StartingStrokes": -4, "BaseProjection": 89.5, "Ownership": 18.5, "SG_OTT": 0.70, "SG_APP": 0.40, "SG_P": 0.55, "BoB_Pct": 23.8, "ID": "300007"},
        {"Name": "Patrick Cantlay", "Salary": 9000, "StartingStrokes": -3, "BaseProjection": 88.0, "Ownership": 21.0, "SG_OTT": 0.60, "SG_APP": 0.65, "SG_P": 0.45, "BoB_Pct": 22.9, "ID": "300008"},
        {"Name": "Viktor Hovland", "Salary": 8800, "StartingStrokes": -2, "BaseProjection": 85.5, "Ownership": 19.5, "SG_OTT": 0.75, "SG_APP": 0.85, "SG_P": -0.05, "BoB_Pct": 23.1, "ID": "300009"},
        {"Name": "Sam Burns", "Salary": 8600, "StartingStrokes": -4, "BaseProjection": 86.0, "Ownership": 20.5, "SG_OTT": 0.35, "SG_APP": 0.55, "SG_P": 0.75, "BoB_Pct": 24.0, "ID": "300010"},
        {"Name": "Sahith Theegala", "Salary": 8400, "StartingStrokes": -3, "BaseProjection": 83.0, "Ownership": 17.5, "SG_OTT": 0.20, "SG_APP": 0.60, "SG_P": 0.40, "BoB_Pct": 23.4, "ID": "300011"},
        {"Name": "Sungjae Im", "Salary": 8200, "StartingStrokes": -3, "BaseProjection": 82.5, "Ownership": 19.0, "SG_OTT": 0.50, "SG_APP": 0.45, "SG_P": 0.30, "BoB_Pct": 22.6, "ID": "300012"},
        {"Name": "Tony Finau", "Salary": 8000, "StartingStrokes": -3, "BaseProjection": 81.0, "Ownership": 16.5, "SG_OTT": 0.55, "SG_APP": 0.90, "SG_P": -0.30, "BoB_Pct": 22.8, "ID": "300013"},
        {"Name": "Shane Lowry", "Salary": 7800, "StartingStrokes": -3, "BaseProjection": 79.5, "Ownership": 15.0, "SG_OTT": 0.40, "SG_APP": 0.85, "SG_P": 0.10, "BoB_Pct": 21.9, "ID": "300014"},
        {"Name": "Russell Henley", "Salary": 7700, "StartingStrokes": -2, "BaseProjection": 78.0, "Ownership": 16.0, "SG_OTT": 0.25, "SG_APP": 0.95, "SG_P": 0.25, "BoB_Pct": 21.7, "ID": "300015"},
        {"Name": "Keegan Bradley", "Salary": 7600, "StartingStrokes": -6, "BaseProjection": 83.5, "Ownership": 23.0, "SG_OTT": 0.30, "SG_APP": 0.70, "SG_P": 0.15, "BoB_Pct": 22.1, "ID": "300016"},
        {"Name": "Tommy Fleetwood", "Salary": 7500, "StartingStrokes": -1, "BaseProjection": 75.0, "Ownership": 13.5, "SG_OTT": 0.45, "SG_APP": 0.60, "SG_P": 0.35, "BoB_Pct": 21.8, "ID": "300017"},
        {"Name": "Justin Thomas", "Salary": 7400, "StartingStrokes": -1, "BaseProjection": 74.5, "Ownership": 14.0, "SG_OTT": 0.35, "SG_APP": 0.80, "SG_P": -0.15, "BoB_Pct": 22.5, "ID": "300018"},
        {"Name": "Akshay Bhatia", "Salary": 7300, "StartingStrokes": -2, "BaseProjection": 73.0, "Ownership": 12.0, "SG_OTT": 0.40, "SG_APP": 0.75, "SG_P": 0.05, "BoB_Pct": 22.7, "ID": "300019"},
        {"Name": "Robert MacIntyre", "Salary": 7200, "StartingStrokes": -2, "BaseProjection": 72.0, "Ownership": 11.5, "SG_OTT": 0.30, "SG_APP": 0.35, "SG_P": 0.45, "BoB_Pct": 22.0, "ID": "300020"},
        {"Name": "Byeong Hun An", "Salary": 7100, "StartingStrokes": -2, "BaseProjection": 71.5, "Ownership": 10.5, "SG_OTT": 0.65, "SG_APP": 0.40, "SG_P": -0.20, "BoB_Pct": 22.4, "ID": "300021"},
        {"Name": "Billy Horschel", "Salary": 7000, "StartingStrokes": -1, "BaseProjection": 69.5, "Ownership": 9.5, "SG_OTT": 0.20, "SG_APP": 0.30, "SG_P": 0.50, "BoB_Pct": 21.2, "ID": "300022"},
        {"Name": "Christiaan Bezuidenhout", "Salary": 6900, "StartingStrokes": -2, "BaseProjection": 69.0, "Ownership": 9.0, "SG_OTT": -0.10, "SG_APP": 0.50, "SG_P": 0.60, "BoB_Pct": 21.0, "ID": "300023"},
        {"Name": "Aaron Rai", "Salary": 6800, "StartingStrokes": -1, "BaseProjection": 68.5, "Ownership": 8.5, "SG_OTT": 0.35, "SG_APP": 0.85, "SG_P": 0.05, "BoB_Pct": 20.8, "ID": "300024"},
        {"Name": "Taylor Pendrith", "Salary": 6700, "StartingStrokes": -1, "BaseProjection": 67.0, "Ownership": 7.5, "SG_OTT": 0.50, "SG_APP": 0.25, "SG_P": 0.30, "BoB_Pct": 21.5, "ID": "300025"},
        {"Name": "Cameron Young", "Salary": 6600, "StartingStrokes": 0, "BaseProjection": 65.5, "Ownership": 7.0, "SG_OTT": 0.70, "SG_APP": 0.45, "SG_P": -0.40, "BoB_Pct": 21.8, "ID": "300026"},
        {"Name": "Tom Hoge", "Salary": 6500, "StartingStrokes": 0, "BaseProjection": 64.0, "Ownership": 6.0, "SG_OTT": 0.05, "SG_APP": 1.05, "SG_P": -0.15, "BoB_Pct": 21.6, "ID": "300027"},
        {"Name": "Matthieu Pavon", "Salary": 6400, "StartingStrokes": -1, "BaseProjection": 63.0, "Ownership": 5.5, "SG_OTT": 0.25, "SG_APP": 0.20, "SG_P": 0.35, "BoB_Pct": 20.5, "ID": "300028"},
        {"Name": "Chris Kirk", "Salary": 6300, "StartingStrokes": 0, "BaseProjection": 62.5, "Ownership": 5.0, "SG_OTT": 0.15, "SG_APP": 0.40, "SG_P": 0.15, "BoB_Pct": 20.2, "ID": "300029"},
        {"Name": "J.T. Poston", "Salary": 6200, "StartingStrokes": 0, "BaseProjection": 61.0, "Ownership": 4.5, "SG_OTT": 0.05, "SG_APP": 0.25, "SG_P": 0.40, "BoB_Pct": 20.0, "ID": "300030"}
    ]
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 3. Sidebar Ingestion & Strategy Controls
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Slate Ingestion")
data_source = st.sidebar.radio("Data Source", ["Use Built-in Tour Championship Slate (30 Players)", "Upload DraftKings CSV", "Google Sheets URL"])

if data_source == "Upload DraftKings CSV":
    uploaded = st.sidebar.file_uploader("Upload DK PGA CSV", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = get_tour_championship_slate()
elif data_source == "Google Sheets URL":
    gsheet_url = st.sidebar.text_input("Paste Google Sheet Share URL:")
    if gsheet_url:
        try:
            sheet_id = gsheet_url.split("/d/")[1].split("/")[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            df_raw = pd.read_csv(csv_url)
            st.sidebar.success("Loaded Google Sheet!")
        except Exception as e:
            st.sidebar.error(f"Error loading Sheet: {e}")
            df_raw = get_tour_championship_slate()
    else:
        df_raw = get_tour_championship_slate()
else:
    df_raw = get_tour_championship_slate()

# Ensure required columns exist
df_slate = df_raw.copy()
if "BaseProjection" not in df_slate.columns:
    df_slate["BaseProjection"] = df_slate["AvgPointsPerGame"] if "AvgPointsPerGame" in df_slate.columns else 70.0
if "Ownership" not in df_slate.columns:
    df_slate["Ownership"] = 15.0
if "StartingStrokes" not in df_slate.columns:
    df_slate["StartingStrokes"] = 0
if "SG_APP" not in df_slate.columns:
    df_slate["SG_APP"] = 0.5
if "SG_OTT" not in df_slate.columns:
    df_slate["SG_OTT"] = 0.5
if "SG_P" not in df_slate.columns:
    df_slate["SG_P"] = 0.2
if "BoB_Pct" not in df_slate.columns:
    df_slate["BoB_Pct"] = 22.0
if "ID" not in df_slate.columns:
    df_slate["ID"] = [f"3000{i:02d}" for i in range(len(df_slate))]

st.sidebar.markdown("---")
st.sidebar.header("🎯 Game Theory & Roster Architecture")

# Roster Archetype Selector
roster_arch = st.sidebar.selectbox(
    "Roster Construction Model",
    [
        "Balanced 6-Pack (Fades $10k+ Studs)",
        "Single-Stud Anchor (1x $10k+ Stud + 5x Mid-Tier)",
        "Stars & Scrubs (2+ $10k+ Studs)",
        "Unconstrained (Pure Median Optimal)"
    ],
    index=0
)

# Salary Bounds (Unspent Salary Lever)
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    min_salary = st.number_input("Min Salary ($)", value=46500, min_value=40000, max_value=50000, step=100)
with col_s2:
    max_salary = st.number_input("Max Salary ($)", value=49400, min_value=45000, max_value=50000, step=100)

# Ownership & Portfolio Controls
st.sidebar.subheader("Contest Ownership Ceilings")
max_cum_own = st.sidebar.slider("Max Cumulative Ownership (%)", min_value=60.0, max_value=220.0, value=120.0, step=5.0)
num_lineups = st.sidebar.slider("Lineups to Generate", min_value=1, max_value=15, value=3, step=1)
max_overlap = st.sidebar.slider("Max Overlap (Shared Golfers)", min_value=1, max_value=5, value=4, step=1)

# Locks & Excludes
all_golfer_names = sorted(df_slate["Name"].tolist())
locked_players = st.sidebar.multiselect("🔒 Lock Players (Must Include)", all_golfer_names)
excluded_players = st.sidebar.multiselect("❌ Exclude Players (Completely Fade)", all_golfer_names)

# Custom Course-Fit Model Multipliers (East Lake Profile)
st.sidebar.markdown("---")
st.sidebar.header("⛳ East Lake Course-Fit Weighting")
with st.sidebar.expander("Adjust SG & Birdie Multipliers", expanded=False):
    w_app = st.slider("SG: Approach Weight", 0.0, 5.0, 2.5, 0.1)
    w_ott = st.slider("SG: Off-The-Tee Weight", 0.0, 5.0, 1.8, 0.1)
    w_p = st.slider("SG: Putting (Bermuda) Weight", 0.0, 5.0, 1.2, 0.1)
    w_bob = st.slider("Birdie or Better % Weight", 0.0, 2.0, 0.8, 0.1)
    w_strokes = st.slider("Starting Strokes Bonus (pts/stroke)", 0.0, 3.0, 1.5, 0.1)

# Compute Adjusted Dynamic Projections
df_slate["Projection"] = np.round(
    df_slate["BaseProjection"] 
    + (df_slate["SG_APP"] * w_app * 3.0) 
    + (df_slate["SG_OTT"] * w_ott * 2.5) 
    + (df_slate["SG_P"] * w_p * 2.0)
    + ((df_slate["BoB_Pct"] - 22.0) * w_bob * 1.5)
    + ((-df_slate["StartingStrokes"]) * w_strokes),
    2
)

# -----------------------------------------------------------------------------
# 4. Fast MILP Optimization Engine
# -----------------------------------------------------------------------------
def optimize_pga_slate(
    df_players,
    arch_type,
    min_sal,
    max_sal,
    max_own,
    num_lineups_to_gen,
    max_overlap_val,
    locks,
    excludes
):
    df_p = df_players.copy().reset_index(drop=True)
    n = len(df_p)
    
    generated_lineups = []
    previous_lineup_sets = []

    for l_idx in range(num_lineups_to_gen):
        prob = LpProblem(f"PGA_TC_GPP_{l_idx+1}", LpMaximize)

        # Decision Variables (Binary indicator for each golfer)
        x = [LpVariable(f"g_{i}", cat="Binary") for i in range(n)]

        # 1. Exactly 6 golfers
        prob += lpSum(x) == 6

        # 2. Salary Cap Bounds
        prob += lpSum([df_p.loc[i, "Salary"] * x[i] for i in range(n)]) <= max_sal
        prob += lpSum([df_p.loc[i, "Salary"] * x[i] for i in range(n)]) >= min_sal

        # 3. Cumulative Ownership Constraint
        prob += lpSum([df_p.loc[i, "Ownership"] * x[i] for i in range(n)]) <= max_own

        # 4. Player Locks and Excludes
        for i in range(n):
            p_name = df_p.loc[i, "Name"]
            if p_name in locks:
                prob += x[i] == 1
            if p_name in excludes:
                prob += x[i] == 0

        # 5. Roster Architecture Constraints
        stud_indices = [i for i in range(n) if df_p.loc[i, "Salary"] >= 10000]
        
        if arch_type == "Balanced 6-Pack (Fades $10k+ Studs)":
            prob += lpSum([x[i] for i in stud_indices]) == 0
        elif arch_type == "Single-Stud Anchor (1x $10k+ Stud + 5x Mid-Tier)":
            prob += lpSum([x[i] for i in stud_indices]) == 1
        elif arch_type == "Stars & Scrubs (2+ $10k+ Studs)":
            prob += lpSum([x[i] for i in stud_indices]) >= 2

        # 6. Max Overlap Between Lineups
        for prev_set in previous_lineup_sets:
            prob += lpSum([x[i] for i in prev_set]) <= max_overlap_val

        # Objective Function: Maximize Projections
        prob += lpSum([df_p.loc[i, "Projection"] * x[i] for i in range(n)])

        # Solve with CBC solver
        solver = PULP_CBC_CMD(msg=False, timeLimit=5)
        status = prob.solve(solver)

        if LpStatus[status] != "Optimal":
            st.warning(f"Lineup #{l_idx+1}: Infeasible solution under constraints. Try loosening Salary or Ownership limits.")
            break

        current_indices = [i for i in range(n) if x[i].varValue > 0.5]
        previous_lineup_sets.append(current_indices)

        l_df = df_p.iloc[current_indices].copy().sort_values(by="Salary", ascending=False).reset_index(drop=True)
        generated_lineups.append(l_df)

    return generated_lineups

# -----------------------------------------------------------------------------
# 5. Application UI & Tabs
# -----------------------------------------------------------------------------
tab_opt, tab_viz, tab_raw = st.tabs(["🚀 Lineup Optimizer", "📊 Leverage & Course-Fit Analytics", "📋 30-Player Field Data"])

with tab_opt:
    st.subheader("Generate Tour Championship Tournament Lineups")
    
    col_btn, col_badge = st.columns([1, 3])
    with col_btn:
        solve_now = st.button("⚡ Solve & Optimize Lineups", type="primary", use_container_width=True)
    with col_badge:
        rem_low = 50000 - max_salary
        rem_high = 50000 - min_salary
        st.write(f"Targeting: **{roster_arch}** | Leaving: **${rem_low:,} - ${rem_high:,} Unspent** | Own Cap: **{max_cum_own}%**")

    if solve_now:
        with st.spinner("Executing Integer Linear Program..."):
            lineups = optimize_pga_slate(
                df_slate,
                arch_type=roster_arch,
                min_sal=min_salary,
                max_sal=max_salary,
                max_own=max_cum_own,
                num_lineups_to_gen=num_lineups,
                max_overlap_val=max_overlap,
                locks=locked_players,
                excludes=excluded_players
            )

        if lineups:
            st.success(f"Generated {len(lineups)} optimal lineup(s) for the Tour Championship!")
            
            # Aggregate Player Exposure
            all_selected = pd.concat(lineups)
            exposure_df = all_selected["Name"].value_counts().reset_index()
            exposure_df.columns = ["Golfer", "Count"]
            exposure_df["Exposure %"] = (exposure_df["Count"] / len(lineups) * 100).round(1).astype(str) + "%"

            # Render Lineup Cards
            for idx, l_df in enumerate(lineups):
                tot_sal = l_df["Salary"].sum()
                rem_sal = 50000 - tot_sal
                tot_proj = l_df["Projection"].sum()
                tot_own = l_df["Ownership"].sum()
                avg_strokes = l_df["StartingStrokes"].mean()
                tot_bs = (l_df["SG_OTT"] + l_df["SG_APP"]).sum()

                st.markdown(f"### 📋 Lineup #{idx+1} — `{roster_arch.split('(')[0].strip()}`")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Salary", f"${tot_sal:,}", delta=f"${rem_sal:,} unspent")
                m2.metric("Projected Points", f"{tot_proj:.2f} pts")
                m3.metric("Cumulative Ownership", f"{tot_own:.1f}%")
                m4.metric("Avg Starting Stroke", f"{avg_strokes:.1f}")
                m5.metric("Total Ball Striking (SG)", f"+{tot_bs:.2f}")

                st.dataframe(
                    l_df[["Name", "Salary", "StartingStrokes", "Projection", "Ownership", "SG_OTT", "SG_APP", "SG_P", "BoB_Pct"]].style.format({
                        "Salary": "${:,.0f}",
                        "Projection": "{:.2f}",
                        "Ownership": "{:.1f}%",
                        "SG_OTT": "{:+.2f}",
                        "SG_APP": "{:+.2f}",
                        "SG_P": "{:+.2f}",
                        "BoB_Pct": "{:.1f}%"
                    }),
                    use_container_width=True
                )
                st.markdown("---")

            # Exposure Summary Section
            st.subheader("📊 Portfolio Player Exposure Breakdown")
            col_exp1, col_exp2 = st.columns([2, 3])
            with col_exp1:
                st.dataframe(exposure_df, use_container_width=True)

            # DraftKings CSV Export Generation
            export_rows = []
            for l_df in lineups:
                ids = l_df["ID"].tolist()
                while len(ids) < 6:
                    ids.append("")
                export_rows.append(ids[:6])

            df_dk_export = pd.DataFrame(export_rows, columns=["G", "G", "G", "G", "G", "G"])
            csv_export_data = df_dk_export.to_csv(index=False)

            with col_exp2:
                st.markdown("**Export to DraftKings Contest Lobby**")
                st.write("Download your lineups formatted in DraftKings' official 6-Golfer CSV upload template:")
                st.download_button(
                    label="📥 Download DraftKings Upload CSV",
                    data=csv_export_data,
                    file_name="DK_Tour_Championship_Lineups.csv",
                    mime="text/csv",
                    use_container_width=True
                )

with tab_viz:
    st.subheader("Tournament Leverage & Course-Fit Visualizations")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**Leverage Matrix (Projected Points vs. Ownership)**")
        fig1 = px.scatter(
            df_slate,
            x="Ownership",
            y="Projection",
            size="Salary",
            color="StartingStrokes",
            text="Name",
            labels={"Ownership": "Projected Ownership %", "Projection": "Model Projected Points"},
            color_continuous_scale="Viridis_r"
        )
        fig1.update_traces(textposition="top center")
        fig1.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col_v2:
        st.markdown("**Ball-Striking Precision (SG: Off-the-Tee vs. SG: Approach)**")
        fig2 = px.scatter(
            df_slate,
            x="SG_OTT",
            y="SG_APP",
            size="BoB_Pct",
            color="Salary",
            text="Name",
            labels={"SG_OTT": "SG: Off The Tee", "SG_APP": "SG: Approach"},
            color_continuous_scale="Blues"
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

with tab_raw:
    st.subheader("30-Player Field Model & Projections Table")
    st.dataframe(
        df_slate.style.format({
            "Salary": "${:,.0f}",
            "BaseProjection": "{:.2f}",
            "Projection": "{:.2f}",
            "Ownership": "{:.1f}%",
            "SG_OTT": "{:+.2f}",
            "SG_APP": "{:+.2f}",
            "SG_P": "{:+.2f}",
            "BoB_Pct": "{:.1f}%"
        }),
        use_container_width=True
    )
