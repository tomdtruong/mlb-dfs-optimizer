import io
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
**Anti-Correlation Firewalls (No RB/WR vs DST)**, **95th-Percentile Ceiling Weighting**, and **Buzz & Quality Filters**.
""")

# -----------------------------------------------------------------------------
# 2. Built-in Realistic Main Slate Pool (Fallback Data)
# -----------------------------------------------------------------------------
def get_default_nfl_slate():
    data = [
        {"Name": "Josh Allen", "Position": "QB", "Salary": 8200, "Team": "BUF", "Opponent": "KC", "Projection": 24.1, "Ceiling": 36.5, "Ownership": 14.0, "ID": "50001"},
        {"Name": "Patrick Mahomes", "Position": "QB", "Salary": 7800, "Team": "KC", "Opponent": "BUF", "Projection": 22.4, "Ceiling": 34.0, "Ownership": 11.5, "ID": "50002"},
        {"Name": "James Cook", "Position": "RB", "Salary": 6800, "Team": "BUF", "Opponent": "KC", "Projection": 16.0, "Ceiling": 27.0, "Ownership": 13.5, "ID": "50003"},
        {"Name": "Isiah Pacheco", "Position": "RB", "Salary": 6600, "Team": "KC", "Opponent": "BUF", "Projection": 15.2, "Ceiling": 25.0, "Ownership": 12.0, "ID": "50004"},
        {"Name": "Rashee Rice", "Position": "WR", "Salary": 6900, "Team": "KC", "Opponent": "BUF", "Projection": 16.8, "Ceiling": 28.0, "Ownership": 16.0, "ID": "50005"},
        {"Name": "Xavier Worthy", "Position": "WR", "Salary": 5800, "Team": "KC", "Opponent": "BUF", "Projection": 12.5, "Ceiling": 24.0, "Ownership": 8.5, "ID": "50006"},
        {"Name": "Khalil Shakir", "Position": "WR", "Salary": 5200, "Team": "BUF", "Opponent": "KC", "Projection": 12.0, "Ceiling": 21.5, "Ownership": 9.0, "ID": "50007"},
        {"Name": "Travis Kelce", "Position": "TE", "Salary": 6200, "Team": "KC", "Opponent": "BUF", "Projection": 14.5, "Ceiling": 26.0, "Ownership": 13.0, "ID": "50009"},
        {"Name": "Dalton Kincaid", "Position": "TE", "Salary": 5100, "Team": "BUF", "Opponent": "KC", "Projection": 11.5, "Ceiling": 20.0, "Ownership": 8.0, "ID": "50010"},
        {"Name": "Bills DST", "Position": "DST", "Salary": 2900, "Team": "BUF", "Opponent": "KC", "Projection": 6.5, "Ceiling": 16.0, "Ownership": 5.0, "ID": "50011"},
        {"Name": "Chiefs DST", "Position": "DST", "Salary": 2800, "Team": "KC", "Opponent": "BUF", "Projection": 6.2, "Ceiling": 15.0, "Ownership": 4.5, "ID": "50012"},
        {"Name": "Jalen Hurts", "Position": "QB", "Salary": 7900, "Team": "PHI", "Opponent": "DAL", "Projection": 23.5, "Ceiling": 35.0, "Ownership": 13.0, "ID": "50013"},
        {"Name": "Dak Prescott", "Position": "QB", "Salary": 7100, "Team": "DAL", "Opponent": "PHI", "Projection": 19.8, "Ceiling": 31.0, "Ownership": 9.5, "ID": "50014"},
        {"Name": "Saquon Barkley", "Position": "RB", "Salary": 8000, "Team": "PHI", "Opponent": "DAL", "Projection": 20.2, "Ceiling": 33.0, "Ownership": 18.5, "ID": "50015"},
        {"Name": "CeeDee Lamb", "Position": "WR", "Salary": 8500, "Team": "DAL", "Opponent": "PHI", "Projection": 21.0, "Ceiling": 36.0, "Ownership": 21.0, "ID": "50017"},
        {"Name": "A.J. Brown", "Position": "WR", "Salary": 8100, "Team": "PHI", "Opponent": "DAL", "Projection": 18.5, "Ceiling": 32.0, "Ownership": 15.0, "ID": "50018"},
        {"Name": "Jake Ferguson", "Position": "TE", "Salary": 5000, "Team": "DAL", "Opponent": "PHI", "Projection": 11.2, "Ceiling": 20.0, "Ownership": 8.0, "ID": "50021"},
        {"Name": "Eagles DST", "Position": "DST", "Salary": 3100, "Team": "PHI", "Opponent": "DAL", "Projection": 7.0, "Ceiling": 17.0, "Ownership": 6.0, "ID": "50023"},
        {"Name": "Lamar Jackson", "Position": "QB", "Salary": 8000, "Team": "BAL", "Opponent": "CIN", "Projection": 24.5, "Ceiling": 38.0, "Ownership": 15.0, "ID": "50025"},
        {"Name": "Joe Burrow", "Position": "QB", "Salary": 7300, "Team": "CIN", "Opponent": "BAL", "Projection": 21.0, "Ceiling": 33.0, "Ownership": 10.0, "ID": "50026"},
        {"Name": "Derrick Henry", "Position": "RB", "Salary": 7700, "Team": "BAL", "Opponent": "CIN", "Projection": 19.5, "Ceiling": 32.0, "Ownership": 16.0, "ID": "50027"},
        {"Name": "Chase Brown", "Position": "RB", "Salary": 6200, "Team": "CIN", "Opponent": "BAL", "Projection": 14.5, "Ceiling": 25.0, "Ownership": 11.0, "ID": "50028"},
        {"Name": "Ja'Marr Chase", "Position": "WR", "Salary": 8600, "Team": "CIN", "Opponent": "BAL", "Projection": 21.5, "Ceiling": 37.0, "Ownership": 20.0, "ID": "50029"},
        {"Name": "Tee Higgins", "Position": "WR", "Salary": 6400, "Team": "CIN", "Opponent": "BAL", "Projection": 14.2, "Ceiling": 26.0, "Ownership": 10.5, "ID": "50030"},
        {"Name": "Mark Andrews", "Position": "TE", "Salary": 5300, "Team": "BAL", "Opponent": "CIN", "Projection": 11.5, "Ceiling": 21.0, "Ownership": 8.0, "ID": "50033"},
        {"Name": "Ravens DST", "Position": "DST", "Salary": 3200, "Team": "BAL", "Opponent": "CIN", "Projection": 7.2, "Ceiling": 18.0, "Ownership": 7.0, "ID": "50035"}
    ]
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 3. Robust DraftKings CSV Ingestion Engine
# -----------------------------------------------------------------------------
def extract_opponent(row):
    game_info = str(row.get('Game Info', ''))
    team = str(row.get('TeamAbbrev', '')).strip()
    if '@' in game_info:
        matchup = game_info.split()[0]
        teams = matchup.split('@')
        if len(teams) == 2:
            away, home = teams[0].strip(), teams[1].strip()
            if team == away:
                return home
            elif team == home:
                return away
    return ''

def parse_draftkings_nfl_csv(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = content.splitlines(keepends=True)
    
    header_idx = -1
    for idx, line in enumerate(lines):
        if "Position" in line and "Salary" in line and "Name" in line:
            header_idx = idx
            break
            
    if header_idx != -1:
        csv_text = "".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text))
    else:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    df = df.dropna(how="all", axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()

    if "Salary" in df.columns:
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
    if "AvgPointsPerGame" in df.columns:
        df["AvgPointsPerGame"] = pd.to_numeric(df["AvgPointsPerGame"], errors="coerce").fillna(0.0)

    if "TeamAbbrev" in df.columns and "Team" not in df.columns:
        df["Team"] = df["TeamAbbrev"].astype(str).str.strip()
    if "Opponent" not in df.columns:
        df["Opponent"] = df.apply(extract_opponent, axis=1)

    df = df[(df["Salary"] >= 2000) & (df["Position"].isin(["QB", "RB", "WR", "TE", "DST"]))].copy().reset_index(drop=True)

    if "Projection" not in df.columns:
        df["Projection"] = df["AvgPointsPerGame"]
        
    if "Ceiling" not in df.columns:
        df["Ceiling"] = np.round(df["Projection"] * 1.6, 1)

    if "Ownership" not in df.columns:
        val_score = df["Projection"] / (df["Salary"] / 1000.0)
        min_v, max_v = val_score.min(), val_score.max()
        if max_v > min_v:
            df["Ownership"] = np.round(2.0 + 20.0 * (val_score - min_v) / (max_v - min_v), 1)
        else:
            df["Ownership"] = 10.0

    if "ID" not in df.columns:
        df["ID"] = [f"500{i:03d}" for i in range(len(df))]

    return df

# -----------------------------------------------------------------------------
# 4. Sidebar Ingestion & Strategy Controls
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Slate Ingestion")
data_source = st.sidebar.radio("Data Source", ["Upload DraftKings CSV", "Use Built-in Default NFL Slate"])

if data_source == "Upload DraftKings CSV":
    uploaded = st.sidebar.file_uploader("Upload DK NFL CSV (DKSalaries.csv)", type=["csv"])
    if uploaded is not None:
        try:
            df_slate = parse_draftkings_nfl_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df_slate)} NFL players!")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
            df_slate = get_default_nfl_slate()
    else:
        st.sidebar.info("Upload your DraftKings CSV to optimize the live slate.")
        df_slate = get_default_nfl_slate()
else:
    df_slate = get_default_nfl_slate()

# -----------------------------------------------------------------------------
# 5. Buzz, Role Verification & Quality Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔥 Buzz & Role Quality Controls")

# Prune inactive / dead punt players
min_proj_floor = st.sidebar.slider("Min Player Projection Floor", 0.0, 10.0, 6.0, 0.5, help="Filters out 3rd-string backups and dead salary fillers")
df_slate = df_slate[df_slate["Projection"] >= min_proj_floor].copy().reset_index(drop=True)

# Custom Player Buzz Adjuster
with st.sidebar.expander("⚡ Adjust Specific Player 'Buzz' / Upgrades", expanded=False):
    if len(df_slate) > 0:
        selected_buzz_player = st.selectbox("Select Player to Adjust", sorted(df_slate["Name"].unique()))
        buzz_adj = st.slider(f"Buzz Multiplier for {selected_buzz_player}", 0.50, 1.50, 1.00, 0.05)
        
        if buzz_adj != 1.00:
            p_idx = df_slate[df_slate["Name"] == selected_buzz_player].index
            df_slate.loc[p_idx, "Ceiling"] = np.round(df_slate.loc[p_idx, "Ceiling"] * buzz_adj, 1)
            df_slate.loc[p_idx, "Projection"] = np.round(df_slate.loc[p_idx, "Projection"] * buzz_adj, 1)

# -----------------------------------------------------------------------------
# 6. Stacking Rules & Firewalls
# -----------------------------------------------------------------------------
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
    min_salary = st.number_input("Min Salary ($)", value=49000, min_value=45000, max_value=50000, step=100)
with col_s2:
    max_salary = st.number_input("Max Salary ($)", value=50000, min_value=48000, max_value=50000, step=100)

max_cum_own = st.sidebar.slider("Max Cumulative Ownership (%)", min_value=70.0, max_value=160.0, value=125.0, step=5.0)
ceiling_weight = st.sidebar.slider("Ceiling vs Floor Weight (0=Median, 1=Pure 95th% Ceiling)", 0.0, 1.0, 0.75, 0.05)

all_players = sorted(df_slate["Name"].tolist()) if len(df_slate) > 0 else []
locked_players = st.sidebar.multiselect("🔒 Lock Players (Must Include)", all_players)
excluded_players = st.sidebar.multiselect("❌ Exclude Players (Completely Fade)", all_players)

# Dynamic Blended Projection
df_slate["BlendedScore"] = np.round(
    (df_slate["Projection"] * (1.0 - ceiling_weight)) + (df_slate["Ceiling"] * ceiling_weight),
    2
)

# -----------------------------------------------------------------------------
# 7. Mixed Integer Linear Programming Solver (SciPy MILP)
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
    if n < 9:
        return None
    
    c = -df_players["BlendedScore"].values
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    A_rows = []
    bl = []
    bu = []
    
    # 1. Total Roster Size = 9 players
    A_rows.append(np.ones(n))
    bl.append(9.0); bu.append(9.0)
    
    # 2. Position Indicators
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

    # 3. Salary constraints
    A_rows.append(df_players["Salary"].values.astype(float))
    bl.append(float(min_sal)); bu.append(float(max_sal))
    
    # 4. Ownership constraint
    A_rows.append(df_players["Ownership"].values.astype(float))
    bl.append(0.0); bu.append(float(max_own))
    
    # 5. Locks & Excludes
    for i in range(n):
        p_name = df_players.loc[i, "Name"]
        if p_name in locks:
            row = np.zeros(n); row[i] = 1.0; A_rows.append(row); bl.append(1.0); bu.append(1.0)
        if p_name in excludes:
            row = np.zeros(n); row[i] = 1.0; A_rows.append(row); bl.append(0.0); bu.append(0.0)
            
    # 6. QB Stacking Constraints
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

    # 7. Correlation Firewalls
    dst_indices = np.where(dst_mask == 1)[0]
    for d_i in dst_indices:
        d_opp = df_players.loc[d_i, "Opponent"]
        for off_i in range(n):
            if df_players.loc[off_i, "Team"] == d_opp:
                pos = df_players.loc[off_i, "Position"]
                if (pos == "RB" and no_opp_dst_flag) or (pos in ["QB", "WR"] and no_qb_dst_flag):
                    row = np.zeros(n); row[d_i] = 1.0; row[off_i] = 1.0; A_rows.append(row); bl.append(0.0); bu.append(1.0)

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
# 8. UI Application Execution & Roster Cards
# -----------------------------------------------------------------------------
tab_opt, tab_viz, tab_raw = st.tabs(["🚀 Optimal 1-Bullet Lineup", "📊 Game Stacks & Leverage", "📋 Full Player Pool"])

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
            st.error("No feasible lineup found under these constraints. Try loosening the Max Ownership cap, lowering the projection floor, or adjusting Salary constraints.")

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
