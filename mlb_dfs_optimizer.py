import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DFS GPP & Showdown Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Sport / Engine Selector
# -----------------------------------------------------------------------------
sport_mode = st.sidebar.radio(
    "🎯 Select Sport / Contest Engine", 
    ["⚾ MLB GPP Tournament Optimizer", "⛳ PGA Round 4 Showdown & Retro"]
)

# =============================================================================
# ENGINE 1: MLB GPP OPTIMIZER
# =============================================================================
if sport_mode == "⚾ MLB GPP Tournament Optimizer":
    st.title("⚾ DraftKings MLB GPP Optimizer & Correlation Engine")
    st.markdown("""
    Optimize DraftKings MLB tournament lineups using **0-1 Mixed Integer Linear Programming (MILP)** 
    with **5-3, 5-2, 4-4 stacking**, **anti-correlation filters**, and **ownership controls**.
    """)

    def get_sample_mlb_slate():
        data = [
            {"Name": "Gerrit Cole", "Position": "P", "Team": "NYY", "Opponent": "BOS", "Salary": 10200, "Projection": 22.4, "Ownership": 28.5, "BattingOrder": 0, "ImpliedRuns": 4.2},
            {"Name": "Brayan Bello", "Position": "P", "Team": "BOS", "Opponent": "NYY", "Salary": 7400, "Projection": 14.1, "Ownership": 12.0, "BattingOrder": 0, "ImpliedRuns": 5.3},
            {"Name": "Anthony Volpe", "Position": "SS", "Team": "NYY", "Opponent": "BOS", "Salary": 4200, "Projection": 8.9, "Ownership": 15.2, "BattingOrder": 1, "ImpliedRuns": 5.3},
            {"Name": "Juan Soto", "Position": "OF", "Team": "NYY", "Opponent": "BOS", "Salary": 6100, "Projection": 12.8, "Ownership": 24.1, "BattingOrder": 2, "ImpliedRuns": 5.3},
            {"Name": "Aaron Judge", "Position": "OF", "Team": "NYY", "Opponent": "BOS", "Salary": 6400, "Projection": 14.2, "Ownership": 27.5, "BattingOrder": 3, "ImpliedRuns": 5.3},
            {"Name": "Giancarlo Stanton", "Position": "OF", "Team": "NYY", "Opponent": "BOS", "Salary": 4800, "Projection": 9.7, "Ownership": 14.8, "BattingOrder": 4, "ImpliedRuns": 5.3},
            {"Name": "Jazz Chisholm Jr.", "Position": "3B/2B", "Team": "NYY", "Opponent": "BOS", "Salary": 4900, "Projection": 9.4, "Ownership": 16.0, "BattingOrder": 5, "ImpliedRuns": 5.3},
            {"Name": "Austin Wells", "Position": "C", "Team": "NYY", "Opponent": "BOS", "Salary": 3800, "Projection": 7.8, "Ownership": 11.2, "BattingOrder": 6, "ImpliedRuns": 5.3},
            {"Name": "Anthony Rizzo", "Position": "1B", "Team": "NYY", "Opponent": "BOS", "Salary": 3500, "Projection": 7.1, "Ownership": 8.4, "BattingOrder": 7, "ImpliedRuns": 5.3},
            {"Name": "Jarren Duran", "Position": "OF", "Team": "BOS", "Opponent": "NYY", "Salary": 5300, "Projection": 10.5, "Ownership": 17.3, "BattingOrder": 1, "ImpliedRuns": 4.2},
            {"Name": "Rafael Devers", "Position": "3B", "Team": "BOS", "Opponent": "NYY", "Salary": 5600, "Projection": 11.2, "Ownership": 19.5, "BattingOrder": 2, "ImpliedRuns": 4.2},
            {"Name": "Tyler O'Neill", "Position": "OF", "Team": "BOS", "Opponent": "NYY", "Salary": 4600, "Projection": 9.0, "Ownership": 13.0, "BattingOrder": 3, "ImpliedRuns": 4.2},
            {"Name": "Triston Casas", "Position": "1B", "Team": "BOS", "Opponent": "NYY", "Salary": 4300, "Projection": 8.5, "Ownership": 10.5, "BattingOrder": 4, "ImpliedRuns": 4.2},
            {"Name": "Masataka Yoshida", "Position": "OF", "Team": "BOS", "Opponent": "NYY", "Salary": 3700, "Projection": 7.6, "Ownership": 7.8, "BattingOrder": 5, "ImpliedRuns": 4.2},
            {"Name": "Connor Wong", "Position": "C", "Team": "BOS", "Opponent": "NYY", "Salary": 3400, "Projection": 6.8, "Ownership": 6.2, "BattingOrder": 6, "ImpliedRuns": 4.2},
            {"Name": "Ceddanne Rafaela", "Position": "SS/OF", "Team": "BOS", "Opponent": "NYY", "Salary": 3600, "Projection": 7.2, "Ownership": 8.0, "BattingOrder": 7, "ImpliedRuns": 4.2},
            {"Name": "Yoshinobu Yamamoto", "Position": "P", "Team": "LAD", "Opponent": "COL", "Salary": 9600, "Projection": 19.8, "Ownership": 21.0, "BattingOrder": 0, "ImpliedRuns": 4.5},
            {"Name": "Kyle Freeland", "Position": "P", "Team": "COL", "Opponent": "LAD", "Salary": 5000, "Projection": 9.2, "Ownership": 4.1, "BattingOrder": 0, "ImpliedRuns": 7.0},
            {"Name": "Shohei Ohtani", "Position": "OF", "Team": "LAD", "Opponent": "COL", "Salary": 6700, "Projection": 15.8, "Ownership": 34.0, "BattingOrder": 1, "ImpliedRuns": 7.0},
            {"Name": "Mookie Betts", "Position": "SS/OF", "Team": "LAD", "Opponent": "COL", "Salary": 6200, "Projection": 13.5, "Ownership": 28.0, "BattingOrder": 2, "ImpliedRuns": 7.0},
            {"Name": "Freddie Freeman", "Position": "1B", "Team": "LAD", "Opponent": "COL", "Salary": 5800, "Projection": 12.6, "Ownership": 23.5, "BattingOrder": 3, "ImpliedRuns": 7.0},
            {"Name": "Teoscar Hernandez", "Position": "OF", "Team": "LAD", "Opponent": "COL", "Salary": 5100, "Projection": 11.0, "Ownership": 20.2, "BattingOrder": 4, "ImpliedRuns": 7.0},
            {"Name": "Max Muncy", "Position": "3B", "Team": "LAD", "Opponent": "COL", "Salary": 4700, "Projection": 9.9, "Ownership": 15.4, "BattingOrder": 5, "ImpliedRuns": 7.0},
            {"Name": "Will Smith", "Position": "C", "Team": "LAD", "Opponent": "COL", "Salary": 5200, "Projection": 10.4, "Ownership": 18.0, "BattingOrder": 6, "ImpliedRuns": 7.0},
            {"Name": "Gavin Lux", "Position": "2B", "Team": "LAD", "Opponent": "COL", "Salary": 3900, "Projection": 8.1, "Ownership": 11.0, "BattingOrder": 7, "ImpliedRuns": 7.0},
            {"Name": "Ezequiel Tovar", "Position": "SS", "Team": "COL", "Opponent": "LAD", "Salary": 4400, "Projection": 8.7, "Ownership": 12.1, "BattingOrder": 1, "ImpliedRuns": 4.5},
            {"Name": "Brenton Doyle", "Position": "OF", "Team": "COL", "Opponent": "LAD", "Salary": 4600, "Projection": 9.1, "Ownership": 14.0, "BattingOrder": 2, "ImpliedRuns": 4.5},
            {"Name": "Ryan McMahon", "Position": "3B", "Team": "COL", "Opponent": "LAD", "Salary": 4200, "Projection": 8.3, "Ownership": 10.2, "BattingOrder": 3, "ImpliedRuns": 4.5},
            {"Name": "Michael Toglia", "Position": "1B/OF", "Team": "COL", "Opponent": "LAD", "Salary": 3800, "Projection": 7.9, "Ownership": 9.5, "BattingOrder": 4, "ImpliedRuns": 4.5},
            {"Name": "Brendan Rodgers", "Position": "2B", "Team": "COL", "Opponent": "LAD", "Salary": 3500, "Projection": 7.0, "Ownership": 6.5, "BattingOrder": 5, "ImpliedRuns": 4.5},
            {"Name": "Jacob Stallings", "Position": "C", "Team": "COL", "Opponent": "LAD", "Salary": 3100, "Projection": 5.9, "Ownership": 4.2, "BattingOrder": 6, "ImpliedRuns": 4.5}
        ]
        return pd.DataFrame(data)

    st.sidebar.subheader("📁 MLB Slate Data")
    mlb_data_mode = st.sidebar.radio("Data Source", ["Use Sample Slate", "Upload DraftKings CSV"], key="mlb_src")
    if mlb_data_mode == "Upload DraftKings CSV":
        up_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="mlb_csv")
        if up_file:
            try:
                raw_bytes = up_file.read(1024).decode("utf-8", errors="ignore")
                up_file.seek(0)
                if "Instructions" in raw_bytes or "Locate the player" in raw_bytes:
                    df_mlb = pd.read_csv(up_file, skiprows=7)
                    cols = ['Position', 'Name', 'Salary', 'TeamAbbrev', 'AvgPointsPerGame', 'Game Info']
                    df_mlb = df_mlb[[c for c in cols if c in df_mlb.columns]].dropna().copy()
                    df_mlb['Team'] = df_mlb.get('TeamAbbrev', 'TEAM')
                    df_mlb['Projection'] = pd.to_numeric(df_mlb.get('AvgPointsPerGame', 0), errors='coerce').fillna(0.0)
                    def parse_opp(row):
                        g, t = str(row.get('Game Info', '')), str(row.get('Team', ''))
                        if '@' in g:
                            teams = g.split(' ')[0].split('@')
                            if len(teams) == 2: return teams[1] if t == teams[0] else teams[0]
                        return "OPP"
                    df_mlb['Opponent'] = df_mlb.apply(parse_opp, axis=1)
                    df_mlb['BattingOrder'] = 0
                    df_mlb['ImpliedRuns'] = 4.5
                    df_mlb['Ownership'] = 10.0
                else:
                    df_mlb = pd.read_csv(up_file)
            except Exception as e:
                st.sidebar.error(f"Error loading: {e}")
                df_mlb = get_sample_mlb_slate()
        else:
            df_mlb = get_sample_mlb_slate()
    else:
        df_mlb = get_sample_mlb_slate()

    st.sidebar.subheader("⚙️ MLB Tournament Settings")
    stack_strategy = st.sidebar.selectbox("Stack Strategy", ["5-3 Stack (Primary + Secondary)", "5-2 Stack", "4-4 Stack", "Unconstrained"], index=0)
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1: min_sal = st.sidebar.number_input("Min Salary", value=48000, step=100)
    with col_s2: max_sal = st.sidebar.number_input("Max Salary", value=50000, step=100)
    no_p_h = st.sidebar.checkbox("Block Pitchers vs Opposing Bats", value=True)
    no_p_p = st.sidebar.checkbox("Block Opposing Pitchers", value=True)
    starters_only = st.sidebar.checkbox("Starters Only (Batting Order 1–9)", value=True)
    max_own = st.sidebar.slider("Max Cumulative Own (%)", 80.0, 350.0, 260.0, 5.0)
    n_lineups = st.sidebar.slider("Lineups to Generate", 1, 10, 3, 1)

    t_opt, t_data = st.tabs(["🚀 Lineup Optimizer", "📋 Slate Data"])
    with t_data:
        st.dataframe(df_mlb, use_container_width=True)
    with t_opt:
        if st.button("⚡ Solve & Optimize MLB Lineups", type="primary"):
            df_active = df_mlb[df_mlb["Projection"] > 0].copy()
            df_active["Position"] = df_active["Position"].astype(str).str.strip().replace({"SP": "P", "RP": "P"})
            p_pool = df_active[df_active["Position"] == "P"].groupby("Team", as_index=False).apply(lambda g: g.nlargest(1, "Projection")).reset_index(drop=True)
            h_pool = df_active[df_active["Position"] != "P"]
            if starters_only and "BattingOrder" in h_pool.columns and (h_pool["BattingOrder"] > 0).any():
                h_pool = h_pool[h_pool["BattingOrder"].between(1, 9)].groupby("Team", as_index=False).apply(lambda g: g.nsmallest(9, "BattingOrder")).reset_index(drop=True)
            else:
                h_pool = h_pool.groupby("Team", as_index=False).apply(lambda g: g.nlargest(9, "Projection")).reset_index(drop=True)
            
            df_p = pd.concat([p_pool, h_pool]).reset_index(drop=True)
            n_p = len(df_p)
            slots = ["P1", "P2", "C", "1B", "2B", "3B", "SS", "OF1", "OF2", "OF3"]
            teams = df_p["Team"].unique().tolist()
            
            def is_el(pos, s):
                p = str(pos).split("/")
                return ("P" in p) if s.startswith("P") else (("OF" in p) if s.startswith("OF") else (s in p))
            
            x, y = {}, {i: LpVariable(f"y_{i}", cat="Binary") for i in range(n_p)}
            prob = LpProblem("MLB_DK", LpMaximize)
            for i in range(n_p):
                for s in slots:
                    if is_el(df_p.loc[i, "Position"], s):
                        x[i, s] = LpVariable(f"x_{i}_{s}", cat="Binary")
                prob += lpSum([x[i, s] for s in slots if (i, s) in x]) == y[i]
            for s in slots:
                prob += lpSum([x[i, s] for i in range(n_p) if (i, s) in x]) == 1
            prob += lpSum([y[i] for i in range(n_p)]) == 10
            prob += lpSum([df_p.loc[i, "Salary"] * y[i] for i in range(n_p)]) <= max_sal
            prob += lpSum([df_p.loc[i, "Salary"] * y[i] for i in range(n_p)]) >= min_sal
            prob += lpSum([df_p.loc[i, "Ownership"] * y[i] for i in range(n_p)]) <= max_own
            
            hitters = [i for i in range(n_p) if df_p.loc[i, "Position"] != "P"]
            pitchers = [i for i in range(n_p) if df_p.loc[i, "Position"] == "P"]
            if no_p_h:
                for p_i in pitchers:
                    opp = df_p.loc[p_i, "Opponent"]
                    for h_i in hitters:
                        if df_p.loc[h_i, "Team"] == opp: prob += y[p_i] + y[h_i] <= 1
            if no_p_p:
                for p1 in pitchers:
                    for p2 in pitchers:
                        if p1 < p2 and df_p.loc[p1, "Team"] == df_p.loc[p2, "Opponent"]: prob += y[p1] + y[p2] <= 1
            
            if stack_strategy == "5-3 Stack (Primary + Secondary)":
                prim = {t: LpVariable(f"prim_{t}", cat="Binary") for t in teams}
                sec = {t: LpVariable(f"sec_{t}", cat="Binary") for t in teams}
                prob += lpSum([prim[t] for t in teams]) == 1
                prob += lpSum([sec[t] for t in teams]) == 1
                for t in teams:
                    prob += prim[t] + sec[t] <= 1
                    t_h = [i for i in hitters if df_p.loc[i, "Team"] == t]
                    prob += lpSum([y[i] for i in t_h]) >= 5 * prim[t] + 3 * sec[t]
                    prob += lpSum([y[i] for i in t_h]) <= 5 * prim[t] + 3 * sec[t]

            prob += lpSum([df_p.loc[i, "Projection"] * y[i] for i in range(n_p)])
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=10))
            
            if LpStatus[prob.status] == "Optimal":
                sel = [i for i in range(n_p) if y[i].varValue > 0.5]
                res_df = df_p.iloc[sel][["Position", "Name", "Team", "Opponent", "Salary", "Projection", "Ownership"]].reset_index(drop=True)
                st.success("Lineup generated!")
                m1, m2, m3 = st.columns(3)
                m1.metric("Salary", f"${res_df['Salary'].sum():,}")
                m2.metric("Projected Points", f"{res_df['Projection'].sum():.2f}")
                m3.metric("Ownership", f"{res_df['Ownership'].sum():.1f}%")
                st.dataframe(res_df, use_container_width=True)

# =============================================================================
# ENGINE 2: PGA ROUND 4 SHOWDOWN & RETROSPECTIVE
# =============================================================================
else:
    st.title("⛳ PGA Round 4 Showdown Optimizer & Retrospective")
    st.markdown("""
    Optimize single-round Sunday Showdown lineups or run deep retrospectives comparing your roster against tournament winners.
    """)

    pga_tab1, pga_tab2 = st.tabs(["🚀 Round 4 Showdown Optimizer", "📊 Round 4 Retrospective Scorecard"])

    def get_sample_pga_r4_slate():
        return pd.DataFrame([
            {"Name": "Scottie Scheffler", "Salary": 11500, "R4_Projection": 68.5, "Ownership": 42.0, "Leaderboard_Pos": 1, "SG_Approach": 2.4, "SG_Putting": 0.8},
            {"Name": "Xander Schauffele", "Salary": 10600, "R4_Projection": 64.0, "Ownership": 35.0, "Leaderboard_Pos": 2, "SG_Approach": 2.1, "SG_Putting": 1.2},
            {"Name": "Collin Morikawa", "Salary": 9800, "R4_Projection": 61.5, "Ownership": 26.0, "Leaderboard_Pos": 4, "SG_Approach": 3.1, "SG_Putting": -0.2},
            {"Name": "Russell Henley", "Salary": 8800, "R4_Projection": 58.0, "Ownership": 20.0, "Leaderboard_Pos": 7, "SG_Approach": 2.8, "SG_Putting": 0.5},
            {"Name": "Shane Lowry", "Salary": 8200, "R4_Projection": 54.5, "Ownership": 15.0, "Leaderboard_Pos": 10, "SG_Approach": 1.9, "SG_Putting": 0.9},
            {"Name": "Alex Noren", "Salary": 7800, "R4_Projection": 51.0, "Ownership": 13.0, "Leaderboard_Pos": 14, "SG_Approach": 1.4, "SG_Putting": 1.1},
            {"Name": "Christiaan Bezuidenhout", "Salary": 7500, "R4_Projection": 49.5, "Ownership": 11.0, "Leaderboard_Pos": 18, "SG_Approach": 1.0, "SG_Putting": 1.8},
            {"Name": "Davis Thompson", "Salary": 7200, "R4_Projection": 48.0, "Ownership": 9.0, "Leaderboard_Pos": 20, "SG_Approach": 1.2, "SG_Putting": 0.6},
            {"Name": "Andrew Putnam", "Salary": 6900, "R4_Projection": 46.0, "Ownership": 7.0, "Leaderboard_Pos": 24, "SG_Approach": 0.8, "SG_Putting": 1.4},
            {"Name": "Mark Hubbard", "Salary": 6500, "R4_Projection": 42.0, "Ownership": 4.0, "Leaderboard_Pos": 28, "SG_Approach": 0.3, "SG_Putting": -0.4},
        ])

    with pga_tab1:
        st.subheader("Sunday Round 4 Showdown Lineup Optimizer")
        st.sidebar.subheader("📁 PGA R4 Slate Ingestion")
        pga_mode = st.sidebar.radio("R4 Data Source", ["Use Sample BMW R4 Slate", "Upload R4 CSV"], key="pga_r4_src")
        
        if pga_mode == "Upload R4 CSV":
            up_pga = st.sidebar.file_uploader("Upload R4 Slate CSV", type=["csv"], key="pga_up")
            if up_pga:
                df_pga = pd.read_csv(up_pga)
            else:
                df_pga = get_sample_pga_r4_slate()
        else:
            df_pga = get_sample_pga_r4_slate()

        df_pga.columns = [c.strip() for c in df_pga.columns]
        proj_col = [c for c in df_pga.columns if "proj" in c.lower() or "fpts" in c.lower() or "point" in c.lower()]
        proj_name = proj_col[0] if proj_col else "R4_Projection"
        if proj_name not in df_pga.columns: df_pga["R4_Projection"] = 50.0

        pga_min_sal = st.sidebar.number_input("R4 Min Salary ($)", value=48000, step=100)
        pga_max_sal = st.sidebar.number_input("R4 Max Salary ($)", value=50000, step=100)
        pga_max_own = st.sidebar.slider("R4 Max Cum Ownership (%)", 60.0, 200.0, 130.0, 5.0)
        chaser_boost = st.sidebar.checkbox("Boost Chasers (T4-T15 by +5%)", value=True)

        st.dataframe(df_pga, use_container_width=True)

        if st.button("⚡ Solve Optimal Round 4 Showdown Lineup", type="primary"):
            df_solve = df_pga.copy().reset_index(drop=True)
            n_g = len(df_solve)
            
            projs = pd.to_numeric(df_solve[proj_name], errors="coerce").fillna(0.0).values.copy()
            if chaser_boost and "Leaderboard_Pos" in df_solve.columns:
                for idx, pos in enumerate(df_solve["Leaderboard_Pos"]):
                    try:
                        p_val = int(pos)
                        if 4 <= p_val <= 15: projs[idx] *= 1.05
                        elif p_val <= 2: projs[idx] *= 0.98
                    except: pass
            
            prob_pga = LpProblem("PGA_R4", LpMaximize)
            gx = [LpVariable(f"g_{i}", cat="Binary") for i in range(n_g)]
            
            prob_pga += lpSum(gx) == 6
            prob_pga += lpSum([df_solve.loc[i, "Salary"] * gx[i] for i in range(n_g)]) <= pga_max_sal
            prob_pga += lpSum([df_solve.loc[i, "Salary"] * gx[i] for i in range(n_g)]) >= pga_min_sal
            if "Ownership" in df_solve.columns:
                prob_pga += lpSum([df_solve.loc[i, "Ownership"] * gx[i] for i in range(n_g)]) <= pga_max_own
                
            prob_pga += lpSum([projs[i] * gx[i] for i in range(n_g)])
            prob_pga.solve(PULP_CBC_CMD(msg=False, timeLimit=5))
            
            if LpStatus[prob_pga.status] == "Optimal":
                sel_g = [i for i in range(n_g) if gx[i].varValue > 0.5]
                res_pga = df_solve.iloc[sel_g].copy()
                res_pga["Adjusted_R4_Proj"] = np.round(projs[sel_g], 2)
                
                st.success("Optimal Round 4 Showdown Lineup Solved!")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Salary", f"${res_pga['Salary'].sum():,}")
                col_m2.metric("R4 Projected Points", f"{res_pga['Adjusted_R4_Proj'].sum():.2f}")
                col_m3.metric("Cumulative Ownership", f"{res_pga['Ownership'].sum():.1f}%" if "Ownership" in res_pga.columns else "N/A")
                
                st.dataframe(res_pga, use_container_width=True)

    with pga_tab2:
        st.subheader("Round 4 Post-Tournament Scorecard & Attribution")
        st.markdown("Compare your Sunday lineup against the contest winner to evaluate leaderboard aggression and putting luck.")

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            my_r4_file = st.file_uploader("Upload My R4 Lineup CSV", type=["csv"], key="my_r4_csv")
        with col_u2:
            win_r4_file = st.file_uploader("Upload Winning R4 Lineup CSV", type=["csv"], key="win_r4_csv")

        if my_r4_file and win_r4_file:
            df_my = pd.read_csv(my_r4_file)
            df_win = pd.read_csv(win_r4_file)
        else:
            st.info("Displaying BMW Championship Sunday Demo Data (Upload custom CSVs above to evaluate your contest).")
            demo_base = get_sample_pga_r4_slate()
            df_my = demo_base.iloc[[0, 1, 3, 5, 7, 9]].copy()
            df_win = demo_base.iloc[[1, 2, 3, 4, 5, 8]].copy()

        def clean_retro(df):
            d = df.copy()
            d.columns = [c.strip().lower().replace(" ", "_").replace(":", "").replace("%", "") for c in d.columns]
            return d

        m_c = clean_retro(df_my)
        w_c = clean_retro(df_win)

        def get_retro_metrics(df, label):
            sal = df["salary"].sum() if "salary" in df.columns else 0
            own = df["ownership"].sum() if "ownership" in df.columns else 0.0
            pts_col = [c for c in df.columns if "point" in c or "fpts" in c or "r4_proj" in c or "actual" in c]
            pts = df[pts_col[0]].sum() if pts_col else 0.0
            
            pos_col = [c for c in df.columns if "pos" in c or "rank" in c]
            if pos_col:
                pos = pd.to_numeric(df[pos_col[0]], errors="coerce").fillna(20)
                leaders = (pos <= 3).sum()
                chasers = ((pos >= 4) & (pos <= 15)).sum()
                backmarkers = (pos > 15).sum()
            else:
                leaders, chasers, backmarkers = 0, 0, 0
                
            sg_app = df["sg_approach"].sum() if "sg_approach" in df.columns else (df["sg_app"].sum() if "sg_app" in df.columns else 0.0)
            sg_p = df["sg_putting"].sum() if "sg_putting" in df.columns else (df["sg_p"].sum() if "sg_p" in df.columns else 0.0)
            
            return {
                "Roster": label,
                "R4 Fantasy Points": round(pts, 1),
                "Total Salary": f"${int(sal):,}",
                "Salary Left": f"${int(50000 - sal):,}",
                "Cumulative Own": f"{own:.1f}%",
                "Top 3 Leaders Rostered": leaders,
                "Chasers (T4-T15) Rostered": chasers,
                "Early Wave (T16+) Rostered": backmarkers,
                "R4 SG: Approach (Total)": round(sg_app, 2) if sg_app != 0.0 else "N/A",
                "R4 SG: Putting (Total)": round(sg_p, 2) if sg_p != 0.0 else "N/A"
            }

        m_stats = get_retro_metrics(m_c, "My Lineup")
        w_stats = get_retro_metrics(w_c, "Winning Lineup")
        scorecard_df = pd.DataFrame([m_stats, w_stats]).set_index("Roster").T
        
        st.markdown("### 📋 Sunday Process Scorecard")
        st.dataframe(scorecard_df, use_container_width=True)

        st.markdown("### 🔍 Strategic Sunday Takeaways")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**My Roster R4 Points:** `{m_stats['R4 Fantasy Points']}` | **Own:** `{m_stats['Cumulative Own']}`")
            st.markdown(f"- Chaser Allocation: **{m_stats['Chasers (T4-T15) Rostered']} golfers**")
            st.markdown(f"- Leader Allocation: **{m_stats['Top 3 Leaders Rostered']} golfers**")
        with c2:
            st.markdown(f"**Winning Roster R4 Points:** `{w_stats['R4 Fantasy Points']}` | **Own:** `{w_stats['Cumulative Own']}`")
            st.markdown(f"- Chaser Allocation: **{w_stats['Chasers (T4-T15) Rostered']} golfers**")
            st.markdown(f"- Leader Allocation: **{w_stats['Top 3 Leaders Rostered']} golfers**")
