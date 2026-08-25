import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import milp, LinearConstraint, Bounds

# -----------------------------------------------------------------------------
# 1. Page Configuration & Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NFL DFS Mega-GPP Optimizer (832k Milly Maker)",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏈 NFL DFS Mega-GPP Optimizer (832k-Entry Single Bullet)")
st.markdown("""
Custom **Mixed Integer Linear Programming (MILP)** optimization engine engineered specifically for 
**ultra-large-field contests (800k+ entries / Milly Maker)**. Features **QB Game Stacking (Double Stack + Bring-Back)**, 
**Anti-Correlation Firewalls (No RB/WR vs DST)**, **95th-Percentile Ceiling Weighting**, and **Ownership Ceilings**.
""")

# -----------------------------------------------------------------------------
# 2. Built-in Realistic Main Slate Pool (Game Matchups & SG Metrics)
# -----------------------------------------------------------------------------
def get_default_nfl_slate():
    data = [
        # KC @ BUF (O/U 51.5)
        {"Name": "Josh Allen", "Position": "QB", "Salary": 8200, "Team": "BUF", "Opponent": "KC", "Projection": 24.1, "Ceiling": 36.5, "Ownership": 14.0, "ID": "50001"},
        {"Name": "Patrick Mahomes", "Position": "QB", "Salary": 7800, "Team": "KC", "Opponent": "BUF", "Projection": 22.4, "Ceiling": 34.0, "Ownership": 11.5, "ID": "50002"},
        {"Name": "James Cook", "Position": "RB", "Salary": 6800, "Team": "BUF", "Opponent": "KC", "Projection": 16.0, "Ceiling": 27.0, "Ownership": 13.5, "ID": "50003"},
        {"Name": "Isiah Pacheco", "Position": "RB", "Salary": 6600, "Team": "KC", "Opponent": "BUF", "Projection": 15.2, "Ceiling": 25.0, "Ownership": 12.0, "ID": "50004"},
        {"Name": "Rashee Rice", "Position": "WR", "Salary": 6900, "Team": "KC", "Opponent": "BUF", "Projection": 16.8, "Ceiling": 28.0, "Ownership": 16.0, "ID": "50005"},
        {"Name": "Xavier Worthy", "Position": "WR", "Salary": 5800, "Team": "KC", "Opponent": "BUF", "Projection": 12.5, "Ceiling": 24.0, "Ownership": 8.5, "ID": "50006"},
        {"Name": "Khalil Shakir", "Position": "WR", "Salary": 5200, "Team": "BUF", "Opponent": "KC", "Projection": 12.0, "Ceiling": 21.5, "Ownership": 9.0, "ID": "50007"},
        {"Name": "Keon Coleman", "Position": "WR", "Salary": 5400, "Team": "BUF", "Opponent": "KC", "Projection": 11.8, "Ceiling": 22.0, "Ownership": 7.0, "ID": "50008"},
        {"Name": "Travis Kelce", "Position": "TE", "Salary": 6200, "Team": "KC", "Opponent": "BUF", "Projection": 14.5, "Ceiling": 26.0, "Ownership": 13.0, "ID": "50009"},
        {"Name": "Dalton Kincaid", "Position": "TE", "Salary": 5100, "Team": "BUF", "Opponent": "KC", "Projection": 11.5, "Ceiling": 20.0, "Ownership": 8.0, "ID": "50010"},
        {"Name": "Bills DST", "Position": "DST", "Salary": 2900, "Team": "BUF", "Opponent": "KC", "Projection": 6.5, "Ceiling": 16.0, "Ownership": 5.0, "ID": "50011"},
        {"Name": "Chiefs DST", "Position": "DST", "Salary": 2800, "Team": "KC", "Opponent": "BUF", "Projection": 6.2, "Ceiling": 15.0, "Ownership": 4.5, "ID": "50012"},

        # PHI @ DAL (O/U 48.5)
        {"Name": "Jalen Hurts", "Position": "QB", "Salary": 7900, "Team": "PHI", "Opponent": "DAL", "Projection": 23.5, "Ceiling": 35.0, "Ownership": 13.0, "ID": "50013"},
        {"Name": "Dak Prescott", "Position": "QB", "Salary": 7100, "Team": "DAL", "Opponent": "PHI", "Projection": 19.8, "Ceiling": 31.0, "Ownership": 9.5, "ID": "50014"},
        {"Name": "Saquon Barkley", "Position": "RB", "Salary": 8000, "Team": "PHI", "Opponent": "DAL", "Projection": 20.2, "Ceiling": 33.0, "Ownership": 18.5, "ID": "50015"},
        {"Name": "Rico Dowdle", "Position": "RB", "Salary": 5600, "Team": "DAL", "Opponent": "PHI", "Projection": 11.0, "Ceiling": 19.0, "Ownership": 6.5, "ID": "50016"},
        {"Name": "CeeDee Lamb", "Position": "WR", "Salary": 8500, "Team": "DAL", "Opponent": "PHI", "Projection": 21.0, "Ceiling": 36.0, "Ownership": 21.0, "ID": "50017"},
        {"Name": "A.J. Brown", "Position": "WR", "Salary": 8100, "Team": "PHI", "Opponent": "DAL", "Projection": 18.5, "Ceiling": 32.0, "Ownership": 15.0, "ID": "50018"},
        {"Name": "DeVonta Smith", "Position": "WR", "Salary": 6600, "Team": "PHI", "Opponent": "DAL", "Projection": 14.8, "Ceiling": 25.0, "Ownership": 11.0, "ID": "50019"},
        {"Name": "Jalen Tolbert", "Position": "WR", "Salary": 4800, "Team": "DAL", "Opponent": "PHI", "Projection": 9.5, "Ceiling": 18.0, "Ownership": 5.0, "ID": "50020"},
        {"Name": "Jake Ferguson", "Position": "TE", "Salary": 5000, "Team": "DAL", "Opponent": "PHI", "Projection": 11.2, "Ceiling": 20.0, "Ownership": 8.0, "ID": "50021"},
        {"Name": "Dallas Goedert", "Position": "TE", "Salary": 4900, "Team": "PHI", "Opponent": "DAL", "Projection": 10.8, "Ceiling": 19.0, "Ownership": 7.5, "ID": "50022"},
        {"Name": "Eagles DST", "Position": "DST", "Salary": 3100, "Team": "PHI", "Opponent": "DAL", "Projection": 7.0, "Ceiling": 17.0, "Ownership": 6.0, "ID": "50023"},
        {"Name": "Cowboys DST", "Position": "DST", "Salary": 2700, "Team": "DAL", "Opponent": "PHI", "Projection": 5.8, "Ceiling": 14.0, "Ownership": 3.5, "ID": "50024"},

        # BAL @ CIN (O/U 49.0)
        {"Name": "Lamar Jackson", "Position": "QB", "Salary": 8000, "Team": "BAL", "Opponent": "CIN", "Projection": 24.5, "Ceiling": 38.0, "Ownership": 15.0, "ID": "50025"},
        {"Name": "Joe Burrow", "Position": "QB", "Salary": 7300, "Team": "CIN", "Opponent": "BAL", "Projection": 21.0, "Ceiling": 33.0, "Ownership": 10.0, "ID": "50026"},
        {"Name": "Derrick Henry", "Position": "RB", "Salary": 7700, "Team": "BAL", "Opponent": "CIN", "Projection": 19.5, "Ceiling": 32.0, "Ownership": 16.0, "ID": "50027"},
        {"Name": "Chase Brown", "Position": "RB", "Salary": 6200, "Team": "CIN", "Opponent": "BAL", "Projection": 14.5, "Ceiling": 25.0, "Ownership": 11.0, "ID": "50028"},
        {"Name": "Ja'Marr Chase", "Position": "WR", "Salary": 8600, "Team": "CIN", "Opponent": "BAL", "Projection": 21.5, "Ceiling": 37.0, "Ownership": 20.0, "ID": "50029"},
        {"Name": "Tee Higgins", "Position": "WR", "Salary": 6400, "Team": "CIN", "Opponent": "BAL", "Projection": 14.2, "Ceiling": 26.0, "Ownership": 10.5, "ID": "50030"},
        {"Name": "Zay Flowers", "Position": "WR", "Salary": 6500, "Team": "BAL", "Opponent": "CIN", "Projection": 15.0, "Ceiling": 26.0, "Ownership": 12.0, "ID": "50031"},
        {"Name": "Rashod Bateman", "Position": "WR", "Salary": 4700, "Team": "BAL", "Opponent": "CIN", "Projection": 9.8, "Ceiling": 19.0, "Ownership": 4.5, "ID": "50032"},
        {"Name": "Mark Andrews", "Position": "TE", "Salary": 5300, "Team": "BAL", "Opponent": "CIN", "Projection": 11.5, "Ceiling": 21.0, "Ownership": 8.0, "ID": "50033"},
        {"Name": "Mike Gesicki", "Position": "TE", "Salary": 4100, "Team": "CIN", "Opponent": "BAL", "Projection": 8.5, "Ceiling": 16.0, "Ownership": 4.0, "ID": "50034"},
        {"Name": "Ravens DST", "Position": "DST", "Salary": 3200, "Team": "BAL", "Opponent": "CIN", "Projection": 7.2, "Ceiling": 18.0, "Ownership": 7.0, "ID": "50035"},
        {"Name": "Bengals DST", "Position": "DST", "Salary": 2600, "Team": "CIN", "Opponent": "BAL", "Projection": 5.5, "Ceiling": 14.0, "Ownership": 3.0, "ID": "50036"},

        # DET @ GB (O/U 48.0)
        {"Name": "Jordan Love", "Position": "QB", "Salary": 6700, "Team": "GB", "Opponent": "DET", "Projection": 19.0, "Ceiling": 31.0, "Ownership": 8.5, "ID": "50037"},
        {"Name": "Jared Goff", "Position": "QB", "Salary": 6500, "Team": "DET", "Opponent": "GB", "Projection": 18.5, "Ceiling": 30.0, "Ownership": 7.5, "ID": "50038"},
        {"Name": "Jahmyr Gibbs", "Position": "RB", "Salary": 7400, "Team": "DET", "Opponent": "GB", "Projection": 18.0, "Ceiling": 30.0, "Ownership": 14.0, "ID": "50039"},
        {"Name": "Josh Jacobs", "Position": "RB", "Salary": 7000, "Team": "GB", "Opponent": "DET", "Projection": 17.0, "Ceiling": 28.0, "Ownership": 12.5, "ID": "50040"},
        {"Name": "David Montgomery", "Position": "RB", "Salary": 6400, "Team": "DET", "Opponent": "GB", "Projection": 14.0, "Ceiling": 24.0, "Ownership": 10.0, "ID": "50041"},
        {"Name": "Amon-Ra St. Brown", "Position": "WR", "Salary": 8300, "Team": "DET", "Opponent": "GB", "Projection": 20.5, "Ceiling": 34.0, "Ownership": 18.0, "ID": "50042"},
        {"Name": "Jayden Reed", "Position": "WR", "Salary": 6700, "Team": "GB", "Opponent": "DET", "Projection": 15.5, "Ceiling": 27.0, "Ownership": 12.0, "ID": "50043"},
        {"Name": "Jameson Williams", "Position": "WR", "Salary": 5900, "Team": "DET", "Opponent": "GB", "Projection": 12.8, "Ceiling": 25.0, "Ownership": 9.5, "ID": "50044"},
        {"Name": "Christian Watson", "Position": "WR", "Salary": 5300, "Team": "GB", "Opponent": "DET", "Projection": 11.0, "Ceiling": 23.0, "Ownership": 6.0, "ID": "50045"},
        {"Name": "Sam LaPorta", "Position": "TE", "Salary": 5500, "Team": "DET", "Opponent": "GB", "Projection": 12.5, "Ceiling": 22.0, "Ownership": 10.0, "ID": "50046"},
        {"Name": "Tucker Kraft", "Position": "TE", "Salary": 4600, "Team": "GB", "Opponent": "DET", "Projection": 10.2, "Ceiling": 19.0, "Ownership": 6.5, "ID": "50047"},
        {"Name": "Lions DST", "Position": "DST", "Salary": 3000, "Team": "DET", "Opponent": "GB", "Projection": 6.8, "Ceiling": 16.0, "Ownership": 5.5, "ID": "50048"},
        {"Name": "Packers DST", "Position": "DST", "Salary": 2800, "Team": "GB", "Opponent": "DET", "Projection": 6.0, "Ceiling": 15.0, "Ownership": 4.0, "ID": "50049"},

        # Value Plays & Standalone Pieces
        {"Name": "Breece Hall", "Position": "RB", "Salary": 7500, "Team": "NYJ", "Opponent": "MIA", "Projection": 18.5, "Ceiling": 31.0, "Ownership": 15.0, "ID": "50050"},
        {"Name": "De'Von Achane", "Position": "RB", "Salary": 7200, "Team": "MIA", "Opponent": "NYJ", "Projection": 17.5, "Ceiling": 32.0, "Ownership": 14.5, "ID": "50051"},
        {"Name": "Kenneth Walker III", "Position": "RB", "Salary": 6500, "Team": "SEA", "Opponent": "ARI", "Projection": 15.8, "Ceiling": 27.0, "Ownership": 11.5, "ID": "50052"},
        {"Name": "Tyrone Tracy Jr.", "Position": "RB", "Salary": 5500, "Team": "NYG", "Opponent": "CAR", "Projection": 13.5, "Ceiling": 23.0, "Ownership": 9.0, "ID": "50053"},
        {"Name": "Malik Nabers", "Position": "WR", "Salary": 7400, "Team": "NYG", "Opponent": "CAR", "Projection": 17.8, "Ceiling": 31.0, "Ownership": 15.5, "ID": "50054"},
        {"Name": "Brian Thomas Jr.", "Position": "WR", "Salary": 6200, "Team": "JAX", "Opponent": "IND", "Projection": 14.5, "Ceiling": 27.0, "Ownership": 11.0, "ID": "50055"},
        {"Name": "Ladd McConkey", "Position": "WR", "Salary": 5700, "Team": "LAC", "Opponent": "TEN", "Projection": 13.2, "Ceiling": 24.0, "Ownership": 8.0, "ID": "50056"},
        {"Name": "Jauan Jennings", "Position": "WR", "Salary": 5100, "Team": "SF", "Opponent": "TB", "Projection": 12.0, "Ceiling": 22.0, "Ownership": 7.5, "ID": "50057"},
        {"Name": "Trey McBride", "Position": "TE", "Salary": 5800, "Team": "ARI", "Opponent": "SEA", "Projection": 13.8, "Ceiling": 24.0, "Ownership": 11.5, "ID": "50058"},
        {"Name": "Cade Otton", "Position": "TE", "Salary": 4400, "Team": "TB", "Opponent": "SF", "Projection": 10.5, "Ceiling": 19.5, "Ownership": 7.0, "ID": "50059"},
        {"Name": "Chargers DST", "Position": "DST", "Salary": 3400, "Team": "LAC", "Opponent": "TEN", "Projection": 7.8, "Ceiling": 20.0, "Ownership": 8.5, "ID": "50060"}
    ]
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 3. Sidebar Ingestion & Strategy Controls
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Slate Ingestion")
data_source = st.sidebar.radio("Data Source", ["Use Built-in NFL Main Slate Pool", "Upload DraftKings CSV", "Google Sheets URL"])

if data_source == "Upload DraftKings CSV":
    uploaded = st.sidebar.file_uploader("Upload DK NFL CSV", type=["csv"])
    if uploaded is not None:
        df_slate = pd.read_csv(uploaded)
    else:
        df_slate = get_default_nfl_slate()
elif data_source == "Google Sheets URL":
    gsheet_url = st.sidebar.text_input("Paste Google Sheet Share URL:")
    if gsheet_url:
        try:
            sheet_id = gsheet_url.split("/d/")[1].split("/")[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            df_slate = pd.read_csv(csv_url)
            st.sidebar.success("Loaded Google Sheet!")
        except Exception as e:
            st.sidebar.error(f"Error loading Sheet: {e}")
            df_slate = get_default_nfl_slate()
    else:
        df_slate = get_default_nfl_slate()
else:
    df_slate = get_default_nfl_slate()

if "Ceiling" not in df_slate.columns:
    df_slate["Ceiling"] = df_slate["Projection"] * 1.6
if "Ownership" not in df_slate.columns:
    df_slate["Ownership"] = 10.0
if "ID" not in df_slate.columns:
    df_slate["ID"] = [f"500{i:02d}" for i in range(len(df_slate))]

st.sidebar.markdown("---")
st.sidebar.header("🎯 832k Milly Maker Stacking Rules")

stack_model = st.sidebar.selectbox(
    "Primary QB Stacking Type",
    [
        "Double Stack (QB + 2 Pass Catchers)",
        "Single Stack (QB + 1 Pass Catcher)",
        "Unconstrained (No Forced QB Stack)"
    ],
    index=0
)

force_bring_back = st.sidebar.checkbox("Force Opposing Bring-Back (Opponent WR/RB/TE)", value=True)

st.sidebar.subheader("🛡️ Correlation Firewalls")
no_rb_vs_dst = st.sidebar.checkbox("No RB vs Opposing DST (Anti-Synergy)", value=True)
no_qb_vs_dst = st.sidebar.checkbox("No QB/WR vs Opposing DST", value=True)

flex_pref = st.sidebar.selectbox(
    "FLEX Positional Rule",
    ["Allow WR, RB, or TE in FLEX (Default)", "Force WR in FLEX (Highest Full-PPR Ceiling)", "Force RB in FLEX"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Portfolio & Ownership Ceilings")

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    min_salary = st.number_input("Min Salary ($)", value=49200, min_value=45000, max_value=50000, step=100)
with col_s2:
    max_salary = st.number_input("Max Salary ($)", value=50000, min_value=48000, max_value=50000, step=100)

max_cum_own = st.sidebar.slider("Max Cumulative Ownership (%)", min_value=70.0, max_value=160.0, value=115.0, step=5.0)
ceiling_weight = st.sidebar.slider("Ceiling vs Floor Weight (0=Median, 1=Pure 95th% Ceiling)", 0.0, 1.0, 0.75, 0.05)

all_players = sorted(df_slate["Name"].tolist())
locked_players = st.sidebar.multiselect("🔒 Lock Players (Must Include)", all_players)
excluded_players = st.sidebar.multiselect("❌ Exclude Players (Completely Fade)", all_players)

# Dynamic Blended Projection
df_slate["BlendedScore"] = np.round(
    (df_slate["Projection"] * (1.0 - ceiling_weight)) + (df_slate["Ceiling"] * ceiling_weight),
    2
)

# -----------------------------------------------------------------------------
# 4. Mixed Integer Linear Programming Solver (SciPy MILP)
# -----------------------------------------------------------------------------
def solve_nfl_lineup(
    df_p,
    stack_type,
    bring_back,
    no_opp_dst_flag,
    no_qb_dst_flag,
    flex_rule,
    min_sal,
    max_sal,
    max_own,
    locks,
    excludes
):
    df_players = df_p.copy().reset_index(drop=True)
    n = len(df_players)
    
    c = -df_players["BlendedScore"].values
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    A_rows = []
    bl = []
    bu = []
    
    # Total 9 players
    A_rows.append(np.ones(n))
    bl.append(9.0); bu.append(9.0)
    
    # Positional constraints
    qb_mask = (df_players["Position"] == "QB").astype(float)
    rb_mask = (df_players["Position"] == "RB").astype(float)
    wr_mask = (df_players["Position"] == "WR").astype(float)
    te_mask = (df_players["Position"] == "TE").astype(float)
    dst_mask = (df_players["Position"] == "DST").astype(float)
    flex_mask = rb_mask + wr_mask + te_mask
    
    A_rows.append(qb_mask.values); bl.append(1.0); bu.append(1.0)
    A_rows.append(dst_mask.values); bl.append(1.0); bu.append(1.0)
    A_rows.append(flex_mask.values); bl.append(7.0); bu.append(7.0)
    
    if flex_rule == "Force WR in FLEX (Highest Full-PPR Ceiling)":
        A_rows.append(rb_mask.values); bl.append(2.0); bu.append(2.0)
        A_rows.append(wr_mask.values); bl.append(4.0); bu.append(4.0)
        A_rows.append(te_mask.values); bl.append(1.0); bu.append(1.0)
    elif flex_rule == "Force RB in FLEX":
        A_rows.append(rb_mask.values); bl.append(3.0); bu.append(3.0)
        A_rows.append(wr_mask.values); bl.append(3.0); bu.append(3.0)
        A_rows.append(te_mask.values); bl.append(1.0); bu.append(1.0)
    else:
        A_rows.append(rb_mask.values); bl.append(2.0); bu.append(3.0)
        A_rows.append(wr_mask.values); bl.append(3.0); bu.append(4.0)
        A_rows.append(te_mask.values); bl.append(1.0); bu.append(2.0)

    # Salary constraints
    A_rows.append(df_players["Salary"].values.astype(float))
    bl.append(float(min_sal)); bu.append(float(max_sal))
    
    # Ownership constraint
    A_rows.append(df_players["Ownership"].values.astype(float))
    bl.append(0.0); bu.append(float(max_own))
    
    # Locks & Excludes
    for i in range(n):
        p_name = df_players.loc[i, "Name"]
        if p_name in locks:
            row = np.zeros(n); row[i] = 1.0; A_rows.append(row); bl.append(1.0); bu.append(1.0)
        if p_name in excludes:
            row = np.zeros(n); row[i] = 1.0; A_rows.append(row); bl.append(0.0); bu.append(0.0)
            
    # QB Stacking Constraints
    qb_indices = np.where(qb_mask == 1)[0]
    for q_i in qb_indices:
        q_team = df_players.loc[q_i, "Team"]
        q_opp = df_players.loc[q_i, "Opponent"]
        
        team_pc_mask = np.zeros(n)
        for i in range(n):
            if df_players.loc[i, "Team"] == q_team and df_players.loc[i, "Position"] in ["WR", "TE"]:
                team_pc_mask[i] = 1.0
                
        opp_bring_back_mask = np.zeros(n)
        for i in range(n):
            if df_players.loc[i, "Team"] == q_opp and df_players.loc[i, "Position"] in ["WR", "TE", "RB"]:
                opp_bring_back_mask[i] = 1.0
                
        if stack_type == "Double Stack (QB + 2 Pass Catchers)":
            row = team_pc_mask.copy(); row[q_i] -= 2.0; A_rows.append(row); bl.append(0.0); bu.append(10.0)
        elif stack_type == "Single Stack (QB + 1 Pass Catcher)":
            row = team_pc_mask.copy(); row[q_i] -= 1.0; A_rows.append(row); bl.append(0.0); bu.append(10.0)
            
        if bring_back:
            row_bb = opp_bring_back_mask.copy(); row_bb[q_i] -= 1.0; A_rows.append(row_bb); bl.append(0.0); bu.append(10.0)

    # Correlation Firewalls
    dst_indices = np.where(dst_mask == 1)[0]
    for d_i in dst_indices:
        d_opp = df_players.loc[d_i, "Opponent"]
        for off_i in range(n):
            if df_players.loc[off_i, "Team"] == d_opp:
                pos = df_players.loc[off_i, "Position"]
                if (pos == "RB" and no_opp_dst_flag) or (pos in ["QB", "WR"] and no_qb_dst_flag):
                    row = np.zeros(n); row[d_i] = 1.0; row[off_i] = 1.0
                    A_rows.append(row); bl.append(0.0); bu.append(1.0)

    A_mat = np.array(A_rows)
    constraints = LinearConstraint(A_mat, bl, bu)
    
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    if res.status != 0:
        return None
        
    sel_idx = np.where(res.x > 0.5)[0]
    lineup_df = df_players.iloc[sel_idx].copy()
    pos_order = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "DST": 5}
    lineup_df["SortKey"] = lineup_df["Position"].map(pos_order)
    return lineup_df.sort_values(by=["SortKey", "Salary"], ascending=[True, False]).drop(columns=["SortKey"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 5. UI Application Execution & Roster Cards
# -----------------------------------------------------------------------------
tab_opt, tab_viz, tab_raw = st.tabs(["🚀 Optimal 1-Bullet Lineup", "📊 Game Stacks & Correlation Matrix", "📋 Full Player Pool"])

with tab_opt:
    st.subheader("Generate Millionaire Maker Optimal Single Entry")
    
    col_btn, col_badge = st.columns([1, 3])
    with col_btn:
        solve_btn = st.button("⚡ Solve Milly Maker Lineup", type="primary", use_container_width=True)
    with col_badge:
        st.write(f"Stacking: **{stack_model}** | Bring-Back: **{'Yes' if force_bring_back else 'No'}** | Own Cap: **{max_cum_own}%**")

    if solve_btn:
        with st.spinner("Executing Mixed Integer Linear Program with Game Correlation Constraints..."):
            lineup = solve_nfl_lineup(
                df_slate,
                stack_type=stack_model,
                bring_back=force_bring_back,
                no_opp_dst_flag=no_rb_vs_dst,
                no_qb_dst_flag=no_qb_vs_dst,
                flex_rule=flex_pref,
                min_sal=min_salary,
                max_sal=max_salary,
                max_own=max_cum_own,
                locks=locked_players,
                excludes=excluded_players
            )

        if lineup is not None:
            tot_sal = lineup["Salary"].sum()
            rem_sal = 50000 - tot_sal
            tot_proj = lineup["Projection"].sum()
            tot_ceil = lineup["Ceiling"].sum()
            tot_own = lineup["Ownership"].sum()
            
            qb_row = lineup[lineup["Position"] == "QB"].iloc[0]
            qb_team = qb_row["Team"]
            qb_opp = qb_row["Opponent"]
            
            st.success(f"Optimal Correlated Single-Bullet Generated around **{qb_row['Name']} ({qb_team})** vs **{qb_opp}**!")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Salary", f"${tot_sal:,}", delta=f"${rem_sal:,} unspent")
            m2.metric("Median Projected Pts", f"{tot_proj:.1f} pts")
            m3.metric("95th% Ceiling Upside", f"{tot_ceil:.1f} pts")
            m4.metric("Cumulative Ownership", f"{tot_own:.1f}%")
            
            st.markdown(f"### 📋 Roster Architecture (`{qb_team}` vs `{qb_opp}` Primary Game Stack)")
            
            st.dataframe(
                lineup[["Position", "Name", "Team", "Opponent", "Salary", "Projection", "Ceiling", "Ownership"]].style.format({
                    "Salary": "${:,.0f}",
                    "Projection": "{:.1f}",
                    "Ceiling": "{:.1f}",
                    "Ownership": "{:.1f}%"
                }),
                use_container_width=True
            )
            
            # Formatted DraftKings CSV Download
            qbs = lineup[lineup["Position"] == "QB"]["ID"].tolist()
            rbs = lineup[lineup["Position"] == "RB"]["ID"].tolist()
            wrs = lineup[lineup["Position"] == "WR"]["ID"].tolist()
            tes = lineup[lineup["Position"] == "TE"]["ID"].tolist()
            dsts = lineup[lineup["Position"] == "DST"]["ID"].tolist()
            
            flex_id = rbs.pop() if len(rbs) > 2 else (wrs.pop() if len(wrs) > 3 else tes.pop())
            dk_row = [qbs[0], rbs[0], rbs[1], wrs[0], wrs[1], wrs[2], tes[0], flex_id, dsts[0]]
            df_dk = pd.DataFrame([dk_row], columns=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"])
            
            st.download_button(
                label="📥 Download DraftKings NFL Lineup CSV",
                data=df_dk.to_csv(index=False),
                file_name="DK_NFL_MillyMaker_Lineup.csv",
                mime="text/csv"
            )
        else:
            st.error("No feasible lineup found under these constraints. Try loosening the Max Ownership cap or Salary constraints.")

with tab_viz:
    st.subheader("Leverage & Ceiling Visualizations")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**Leverage Matrix (Ceiling Points vs. Ownership)**")
        fig1 = px.scatter(
            df_slate,
            x="Ownership",
            y="Ceiling",
            size="Salary",
            color="Position",
            text="Name",
            labels={"Ownership": "Projected Ownership %", "Ceiling": "95th-Percentile Ceiling"}
        )
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_v2:
        st.markdown("**Salary-to-Ceiling Value Breakdown**")
        fig2 = px.bar(
            df_slate.sort_values(by="Ceiling", ascending=False).head(15),
            x="Name",
            y="Ceiling",
            color="Position",
            text="Salary"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab_raw:
    st.subheader("Player Slate Database")
    st.dataframe(df_slate, use_container_width=True)
