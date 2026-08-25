import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

# -----------------------------------------------------------------------------
# 1. Page Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DraftKings MLB GPP Optimizer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ DraftKings MLB GPP Optimizer & Correlation Engine")
st.markdown("""
Optimize DraftKings MLB tournament lineups using **0-1 Mixed Integer Linear Programming (MILP)** 
with strict **stacking architectures (5-3, 5-2, 4-4)**, **anti-correlation pruning** (Pitcher vs. Opponent), 
and **game-theory leverage controls**.
""")

# -----------------------------------------------------------------------------
# 2. Built-In Sample Slate Generator
# -----------------------------------------------------------------------------
def get_sample_slate_data():
    """Generates a realistic 3-game sample slate for immediate testing."""
    data = [
        # Game 1: NYY vs BOS (Fenway Park - Total: 9.5)
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

        # Game 2: LAD vs COL (Coors Field - Total: 11.5)
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
        {"Name": "Jacob Stallings", "Position": "C", "Team": "COL", "Opponent": "LAD", "Salary": 3100, "Projection": 5.9, "Ownership": 4.2, "BattingOrder": 6, "ImpliedRuns": 4.5},

        # Game 3: PHI vs ATL (Citizens Bank Park - Total: 8.5)
        {"Name": "Zack Wheeler", "Position": "P", "Team": "PHI", "Opponent": "ATL", "Salary": 10000, "Projection": 21.5, "Ownership": 25.0, "BattingOrder": 0, "ImpliedRuns": 3.9},
        {"Name": "Max Fried", "Position": "P", "Team": "ATL", "Opponent": "PHI", "Salary": 8800, "Projection": 17.2, "Ownership": 16.5, "BattingOrder": 0, "ImpliedRuns": 4.6},
        {"Name": "Kyle Schwarber", "Position": "OF", "Team": "PHI", "Opponent": "ATL", "Salary": 5700, "Projection": 11.8, "Ownership": 21.0, "BattingOrder": 1, "ImpliedRuns": 4.6},
        {"Name": "Trea Turner", "Position": "SS", "Team": "PHI", "Opponent": "ATL", "Salary": 5900, "Projection": 12.1, "Ownership": 22.0, "BattingOrder": 2, "ImpliedRuns": 4.6},
        {"Name": "Bryce Harper", "Position": "1B", "Team": "PHI", "Opponent": "ATL", "Salary": 6000, "Projection": 12.5, "Ownership": 24.0, "BattingOrder": 3, "ImpliedRuns": 4.6},
        {"Name": "Alec Bohm", "Position": "3B/1B", "Team": "PHI", "Opponent": "ATL", "Salary": 4500, "Projection": 9.1, "Ownership": 13.5, "BattingOrder": 4, "ImpliedRuns": 4.6},
        {"Name": "Nick Castellanos", "Position": "OF", "Team": "PHI", "Opponent": "ATL", "Salary": 4300, "Projection": 8.6, "Ownership": 11.0, "BattingOrder": 5, "ImpliedRuns": 4.6},
        {"Name": "J.T. Realmuto", "Position": "C", "Team": "PHI", "Opponent": "ATL", "Salary": 4600, "Projection": 8.9, "Ownership": 12.8, "BattingOrder": 6, "ImpliedRuns": 4.6},
        {"Name": "Bryson Stott", "Position": "2B/SS", "Team": "PHI", "Opponent": "ATL", "Salary": 3900, "Projection": 7.8, "Ownership": 9.0, "BattingOrder": 7, "ImpliedRuns": 4.6},
        {"Name": "Michael Harris II", "Position": "OF", "Team": "ATL", "Opponent": "PHI", "Salary": 4500, "Projection": 9.0, "Ownership": 12.0, "BattingOrder": 1, "ImpliedRuns": 3.9},
        {"Name": "Ozzie Albies", "Position": "2B", "Team": "ATL", "Opponent": "PHI", "Salary": 4800, "Projection": 9.5, "Ownership": 14.5, "BattingOrder": 2, "ImpliedRuns": 3.9},
        {"Name": "Marcell Ozuna", "Position": "OF", "Team": "ATL", "Opponent": "PHI", "Salary": 5500, "Projection": 11.5, "Ownership": 19.0, "BattingOrder": 3, "ImpliedRuns": 3.9},
        {"Name": "Matt Olson", "Position": "1B", "Team": "ATL", "Opponent": "PHI", "Salary": 5200, "Projection": 10.8, "Ownership": 16.5, "BattingOrder": 4, "ImpliedRuns": 3.9},
        {"Name": "Austin Riley", "Position": "3B", "Team": "ATL", "Opponent": "PHI", "Salary": 5000, "Projection": 10.2, "Ownership": 15.0, "BattingOrder": 5, "ImpliedRuns": 3.9},
        {"Name": "Sean Murphy", "Position": "C", "Team": "ATL", "Opponent": "PHI", "Salary": 4000, "Projection": 8.0, "Ownership": 9.5, "BattingOrder": 6, "ImpliedRuns": 3.9},
        {"Name": "Orlando Arcia", "Position": "SS", "Team": "ATL", "Opponent": "PHI", "Salary": 3200, "Projection": 6.2, "Ownership": 5.0, "BattingOrder": 7, "ImpliedRuns": 3.9},
    ]
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Settings
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Slate Ingestion")
data_mode = st.sidebar.radio("Data Source", ["Use Sample Slate (3 Games / 6 Teams)", "Upload DraftKings CSV"])

if data_mode == "Upload DraftKings CSV":
    uploaded_file = st.sidebar.file_uploader("Upload DK Slate CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Check if file has DK instructions header (rows 0-6)
            sample_bytes = uploaded_file.read(1024).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            if "Instructions" in sample_bytes or "Locate the player" in sample_bytes:
                df = pd.read_csv(uploaded_file, skiprows=7)
                cols_to_keep = ['Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary', 'Game Info', 'TeamAbbrev', 'AvgPointsPerGame']
                existing_cols = [c for c in cols_to_keep if c in df.columns]
                df = df[existing_cols].dropna(subset=['Name', 'Salary']).copy()
                if 'TeamAbbrev' in df.columns and 'Team' not in df.columns:
                    df['Team'] = df['TeamAbbrev']
                if 'AvgPointsPerGame' in df.columns and 'Projection' not in df.columns:
                    df['Projection'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(0.0)
                if 'Opponent' not in df.columns and 'Game Info' in df.columns:
                    def parse_opp(row):
                        g = str(row.get('Game Info', ''))
                        t = str(row.get('Team', ''))
                        if '@' in g:
                            teams = g.split(' ')[0].split('@')
                            if len(teams) == 2:
                                return teams[1] if t == teams[0] else teams[0]
                        return "OPP"
                    df['Opponent'] = df.apply(parse_opp, axis=1)
                if 'BattingOrder' not in df.columns:
                    df['BattingOrder'] = 0
                if 'ImpliedRuns' not in df.columns:
                    df['ImpliedRuns'] = 4.5
                if 'Ownership' not in df.columns:
                    df['Ownership'] = 10.0
            else:
                df = pd.read_csv(uploaded_file)
                if 'Projection' not in df.columns and 'AvgPointsPerGame' in df.columns:
                    df['Projection'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(0.0)
                if 'Team' not in df.columns and 'TeamAbbrev' in df.columns:
                    df['Team'] = df['TeamAbbrev']
                if 'Opponent' not in df.columns:
                    df['Opponent'] = "OPP"
                if 'BattingOrder' not in df.columns:
                    df['BattingOrder'] = 0
                if 'ImpliedRuns' not in df.columns:
                    df['ImpliedRuns'] = 4.5
                if 'Ownership' not in df.columns:
                    df['Ownership'] = 10.0
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
            df = get_sample_slate_data()
    else:
        st.info("Awaiting CSV upload. Defaulting to sample slate.")
        df = get_sample_slate_data()
else:
    df = get_sample_slate_data()

# Clean and normalize types
if "Salary" in df.columns:
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
if "Projection" in df.columns:
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce").fillna(0.0)
if "Ownership" in df.columns:
    df["Ownership"] = pd.to_numeric(df["Ownership"], errors="coerce").fillna(0.0)
if "BattingOrder" in df.columns:
    df["BattingOrder"] = pd.to_numeric(df["BattingOrder"], errors="coerce").fillna(0).astype(int)
if "ImpliedRuns" in df.columns:
    df["ImpliedRuns"] = pd.to_numeric(df["ImpliedRuns"], errors="coerce").fillna(4.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ GPP Optimization Rules")

# Stacking Configuration
stack_strategy = st.sidebar.selectbox(
    "Stack Architecture",
    ["5-3 Stack (Primary + Secondary)", "5-2 Stack", "4-4 Stack", "Unconstrained (No Stacking)"],
    index=0
)

# Salary Constraints
col_sal1, col_sal2 = st.sidebar.columns(2)
with col_sal1:
    min_salary = st.number_input("Min Salary ($)", value=48000, min_value=40000, max_value=50000, step=100)
with col_sal2:
    max_salary = st.number_input("Max Salary ($)", value=50000, min_value=45000, max_value=50000, step=100)

# Anti-Correlation Rules
st.sidebar.subheader("Anti-Correlation Filters")
no_pitcher_vs_hitter = st.sidebar.checkbox("Block Pitchers vs Opposing Hitters", value=True)
no_opposing_pitchers = st.sidebar.checkbox("Block Opposing Pitchers (Same Game)", value=True)

# Starters Only Filter
st.sidebar.subheader("Roster Integrity")
starters_only = st.sidebar.checkbox("Strict Starters Only (Batting Order 1–9)", value=True)

# GPP Tournament Controls
st.sidebar.subheader("Tournament Diversity & Ownership")
max_roster_own = st.sidebar.slider("Max Cumulative Ownership (%)", min_value=80.0, max_value=350.0, value=280.0, step=5.0)
num_lineups = st.sidebar.slider("Lineups to Generate", min_value=1, max_value=10, value=3, step=1)
max_overlap = st.sidebar.slider("Max Player Overlap Between Lineups", min_value=3, max_value=8, value=6, step=1)

# -----------------------------------------------------------------------------
# 4. Analytics & Heatmap Visualizations
# -----------------------------------------------------------------------------
tab_opt, tab_heatmap, tab_raw = st.tabs(["🚀 Lineup Optimizer", "🔥 Correlation & Matchup Heatmap", "📋 Slate Data"])

with tab_heatmap:
    st.subheader("Team Matchup & Vegas Implied Totals Heatmap")
    
    df_heat = df.copy()
    if starters_only and "BattingOrder" in df_heat.columns and (df_heat["BattingOrder"] > 0).any():
        df_heat = df_heat[(df_heat["BattingOrder"].between(1, 9)) | (df_heat["Position"] == "P")]

    if not df_heat.empty and "Team" in df_heat.columns and "BattingOrder" in df_heat.columns:
        pivot_proj = df_heat[df_heat["Position"] != "P"].pivot_table(
            index="Team", 
            columns="BattingOrder", 
            values="Projection", 
            aggfunc="mean"
        ).fillna(0)

        team_stats = df_heat.groupby(["Team", "Opponent"]).agg(
            ImpliedRuns=("ImpliedRuns", "first") if "ImpliedRuns" in df_heat.columns else ("Projection", "count"),
            TotalHitterProj=("Projection", lambda x: x[df_heat.loc[x.index, "Position"] != "P"].sum()),
            AvgOwnership=("Ownership", "mean") if "Ownership" in df_heat.columns else ("Projection", "mean")
        ).reset_index()

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**Batting Order Projection Matrix by Team**")
            fig1 = px.imshow(
                pivot_proj,
                labels=dict(x="Batting Order Slot", y="Team", color="Proj Pts"),
                x=[f"Order #{col}" for col in pivot_proj.columns],
                y=pivot_proj.index,
                color_continuous_scale="Viridis",
                text_auto=".1f",
                aspect="auto"
            )
            fig1.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig1, use_container_width=True)

        with col_h2:
            st.markdown("**Team Leverage Quadrant (Vegas Total vs Hitter Upside)**")
            fig2 = px.scatter(
                team_stats,
                x="ImpliedRuns",
                y="TotalHitterProj",
                size="AvgOwnership",
                color="Team",
                text="Team",
                labels={
                    "ImpliedRuns": "Vegas Implied Team Runs",
                    "TotalHitterProj": "Total Projected Hitter Points",
                    "AvgOwnership": "Avg Ownership %"
                }
            )
            fig2.update_traces(textposition="top center")
            fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Insufficient data to render heatmap.")

with tab_raw:
    st.subheader("Raw Slate Input Data")
    st.dataframe(df, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Fast PuLP MILP Optimization Core
# -----------------------------------------------------------------------------
def optimize_dk_mlb(df_input, stack_type, min_sal, max_sal, block_p_h, block_p_p, max_own, num_lineups_to_gen, max_overlap_val, enforce_starters=True):
    df_raw = df_input.copy()
    
    # 1. Filter out inactive / 0-projection players
    df_active = df_raw[df_raw["Projection"] > 0].copy()
    
    # Standardize position strings
    df_active["Position"] = df_active["Position"].astype(str).str.strip().replace({"SP": "P", "RP": "P"})
    
    # 2. Strict Starters Filtering:
    pitchers_filtered = df_active[df_active["Position"] == "P"].groupby("Team", as_index=False).apply(
        lambda g: g.nlargest(1, "Projection")
    ).reset_index(drop=True)
    
    hitters_raw = df_active[df_active["Position"] != "P"]
    if enforce_starters and "BattingOrder" in hitters_raw.columns and (hitters_raw["BattingOrder"] <= 9).any() and (hitters_raw["BattingOrder"] > 0).any():
        hitters_filtered = hitters_raw[hitters_raw["BattingOrder"].between(1, 9)].groupby("Team", as_index=False).apply(
            lambda g: g.nsmallest(9, "BattingOrder")
        ).reset_index(drop=True)
    else:
        hitters_filtered = hitters_raw.groupby("Team", as_index=False).apply(
            lambda g: g.nlargest(9, "Projection")
        ).reset_index(drop=True)
    
    df_players = pd.concat([pitchers_filtered, hitters_filtered]).reset_index(drop=True)
    n_players = len(df_players)
    
    slots = ["P1", "P2", "C", "1B", "2B", "3B", "SS", "OF1", "OF2", "OF3"]
    teams = df_players["Team"].unique().tolist()
    
    def is_eligible(pos_str, slot):
        pos = str(pos_str).split("/")
        if slot in ["P1", "P2"]:
            return "P" in pos
        elif slot.startswith("OF"):
            return "OF" in pos
        else:
            return slot in pos

    generated_lineups = []
    previous_lineup_sets = []

    for l_idx in range(num_lineups_to_gen):
        prob = LpProblem(f"DK_MLB_GPP_{l_idx+1}", LpMaximize)

        # Decision Variables
        x = {}
        for p_idx in range(n_players):
            pos_str = str(df_players.loc[p_idx, "Position"])
            for slot in slots:
                if is_eligible(pos_str, slot):
                    x[p_idx, slot] = LpVariable(f"x_{p_idx}_{slot}", cat="Binary")

        # Binary indicator y[p_idx] for overall selection
        y = {p_idx: LpVariable(f"y_{p_idx}", cat="Binary") for p_idx in range(n_players)}

        # Link player selection to slot assignments
        for p_idx in range(n_players):
            eligible_slots = [slot for slot in slots if (p_idx, slot) in x]
            prob += lpSum([x[p_idx, slot] for slot in eligible_slots]) == y[p_idx]

        # Positional constraints: exactly 1 player per slot
        for slot in slots:
            eligible_players = [p_idx for p_idx in range(n_players) if (p_idx, slot) in x]
            prob += lpSum([x[p_idx, slot] for p_idx in eligible_players]) == 1

        # Total Lineup Size: Exactly 10 players
        prob += lpSum([y[p_idx] for p_idx in range(n_players)]) == 10

        # Salary Bounds
        prob += lpSum([df_players.loc[p_idx, "Salary"] * y[p_idx] for p_idx in range(n_players)]) <= max_sal
        prob += lpSum([df_players.loc[p_idx, "Salary"] * y[p_idx] for p_idx in range(n_players)]) >= min_sal

        # Cumulative Ownership Cap
        if "Ownership" in df_players.columns:
            prob += lpSum([df_players.loc[p_idx, "Ownership"] * y[p_idx] for p_idx in range(n_players)]) <= max_own

        # Anti-Correlation Constraints
        hitter_indices = [p_idx for p_idx in range(n_players) if df_players.loc[p_idx, "Position"] != "P"]
        pitcher_indices = [p_idx for p_idx in range(n_players) if "P" in str(df_players.loc[p_idx, "Position"]).split("/")]

        if block_p_h:
            for p_idx in pitcher_indices:
                p_opp = df_players.loc[p_idx, "Opponent"]
                opp_hitters = [h_idx for h_idx in hitter_indices if df_players.loc[h_idx, "Team"] == p_opp]
                for h_idx in opp_hitters:
                    prob += y[p_idx] + y[h_idx] <= 1

        if block_p_p:
            for p1 in pitcher_indices:
                for p2 in pitcher_indices:
                    if p1 < p2 and df_players.loc[p1, "Team"] == df_players.loc[p2, "Opponent"]:
                        prob += y[p1] + y[p2] <= 1

        # Team Stacking Logic
        if stack_type != "Unconstrained (No Stacking)":
            primary_indicators = {t: LpVariable(f"prim_{t}", cat="Binary") for t in teams}
            prob += lpSum([primary_indicators[t] for t in teams]) == 1

            if stack_type == "5-3 Stack (Primary + Secondary)":
                sec_indicators = {t: LpVariable(f"sec_{t}", cat="Binary") for t in teams}
                prob += lpSum([sec_indicators[t] for t in teams]) == 1

                for t in teams:
                    prob += primary_indicators[t] + sec_indicators[t] <= 1
                    t_hitters = [p_idx for p_idx in hitter_indices if df_players.loc[p_idx, "Team"] == t]
                    prob += lpSum([y[p_idx] for p_idx in t_hitters]) >= 5 * primary_indicators[t] + 3 * sec_indicators[t]
                    prob += lpSum([y[p_idx] for p_idx in t_hitters]) <= 5 * primary_indicators[t] + 3 * sec_indicators[t]

            elif stack_type == "5-2 Stack":
                sec_indicators = {t: LpVariable(f"sec_{t}", cat="Binary") for t in teams}
                prob += lpSum([sec_indicators[t] for t in teams]) == 1

                for t in teams:
                    prob += primary_indicators[t] + sec_indicators[t] <= 1
                    t_hitters = [p_idx for p_idx in hitter_indices if df_players.loc[p_idx, "Team"] == t]
                    prob += lpSum([y[p_idx] for p_idx in t_hitters]) >= 5 * primary_indicators[t] + 2 * sec_indicators[t]
                    prob += lpSum([y[p_idx] for p_idx in t_hitters]) <= 5 * primary_indicators[t] + 2 * sec_indicators[t] + 1

            elif stack_type == "4-4 Stack":
                sec_indicators = {t: LpVariable(f"sec_{t}", cat="Binary") for t in teams}
                prob += lpSum([sec_indicators[t] for t in teams]) == 1

                for t in teams:
                    prob += primary_indicators[t] + sec_indicators[t] <= 1
                    t_hitters = [p_idx for p_idx in hitter_indices if df_players.loc[p_idx, "Team"] == t]
                    prob += lpSum([y[p_idx] for p_idx in t_hitters]) >= 4 * primary_indicators[t] + 4 * sec_indicators[t]
                    prob += lpSum([y[p_idx] for p_idx in t_hitters]) <= 4 * primary_indicators[t] + 4 * sec_indicators[t]
        else:
            for t in teams:
                t_hitters = [p_idx for p_idx in hitter_indices if df_players.loc[p_idx, "Team"] == t]
                prob += lpSum([y[p_idx] for p_idx in t_hitters]) <= 5

        # Max Overlap Diversity Constraint
        for prev_set in previous_lineup_sets:
            prob += lpSum([y[p_idx] for p_idx in prev_set]) <= max_overlap_val

        # Objective Function
        prob += lpSum([df_players.loc[p_idx, "Projection"] * y[p_idx] for p_idx in range(n_players)])

        # Solve with fast CBC solver (0.1s)
        solver = PULP_CBC_CMD(msg=False, timeLimit=10, gapRel=0.01)
        status = prob.solve(solver)

        if LpStatus[status] != "Optimal":
            st.warning(f"Lineup {l_idx+1}: Infeasible solution under constraints.")
            break

        current_lineup_indices = [p_idx for p_idx in range(n_players) if y[p_idx].varValue > 0.5]
        previous_lineup_sets.append(current_lineup_indices)

        lineup_records = []
        for p_idx in current_lineup_indices:
            assigned_slot = [slot for slot in slots if (p_idx, slot) in x and x[p_idx, slot].varValue > 0.5][0]
            lineup_records.append({
                "Slot": assigned_slot.replace("1", "").replace("2", "").replace("3", ""),
                "Name": df_players.loc[p_idx, "Name"],
                "Team": df_players.loc[p_idx, "Team"],
                "Opponent": df_players.loc[p_idx, "Opponent"],
                "Pos": df_players.loc[p_idx, "Position"],
                "Salary": df_players.loc[p_idx, "Salary"],
                "Proj": df_players.loc[p_idx, "Projection"],
                "Own%": df_players.loc[p_idx, "Ownership"] if "Ownership" in df_players.columns else 0.0,
                "Order": df_players.loc[p_idx, "BattingOrder"] if "BattingOrder" in df_players.columns else "-"
            })

        slot_order = {"P": 1, "C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "OF": 7}
        df_lineup = pd.DataFrame(lineup_records)
        df_lineup["SortKey"] = df_lineup["Slot"].map(slot_order)
        df_lineup = df_lineup.sort_values(by="SortKey").drop(columns=["SortKey"]).reset_index(drop=True)
        
        generated_lineups.append(df_lineup)

    return generated_lineups

# -----------------------------------------------------------------------------
# 6. Optimizer Execution & Results UI
# -----------------------------------------------------------------------------
with tab_opt:
    st.subheader("Generate GPP Tournament Lineups")
    
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_solver = st.button("⚡ Solve & Optimize Lineups", type="primary", use_container_width=True)
    with col_info:
        st.write(f"Targeting: **{stack_strategy}** | Salary: **${min_salary:,} - ${max_salary:,}** | Starters Only: **{'ON' if starters_only else 'OFF'}**")

    if run_solver:
        with st.spinner("Solving Integer Linear Program..."):
            lineups = optimize_dk_mlb(
                df,
                stack_type=stack_strategy,
                min_sal=min_salary,
                max_sal=max_salary,
                block_p_h=no_pitcher_vs_hitter,
                block_p_p=no_opposing_pitchers,
                max_own=max_roster_own,
                num_lineups_to_gen=num_lineups,
                max_overlap_val=max_overlap,
                enforce_starters=starters_only
            )

        if lineups:
            st.success(f"Generated {len(lineups)} optimal GPP lineup(s)!")
            
            for idx, l_df in enumerate(lineups):
                total_s = l_df["Salary"].sum()
                total_p = l_df["Proj"].sum()
                total_o = l_df["Own%"].sum()
                
                hitters = l_df[l_df["Slot"] != "P"]
                team_counts = hitters["Team"].value_counts().to_dict()
                stack_desc = " + ".join([f"{count}x {team}" for team, count in team_counts.items()])

                st.markdown(f"### 📋 Lineup #{idx+1} — Stack: `{stack_desc}`")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Salary", f"${total_s:,}", delta=f"${50000 - total_s:,} left")
                m2.metric("Projected Points", f"{total_p:.2f} pts")
                m3.metric("Cumulative Ownership", f"{total_o:.1f}%")
                m4.metric("Stack Composition", stack_desc)

                st.dataframe(
                    l_df.style.format({
                        "Salary": "${:,.0f}",
                        "Proj": "{:.1f}",
                        "Own%": "{:.1f}%"
                    }),
                    use_container_width=True
                )
                st.markdown("---")
