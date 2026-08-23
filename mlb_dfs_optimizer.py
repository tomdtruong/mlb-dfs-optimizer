import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus
import requests
import io
import re
import unicodedata
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Page Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DraftKings MLB GPP Optimizer & Automation Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ DraftKings MLB GPP Optimizer & Automation Engine")
st.markdown("""
All-in-one MLB DFS tournament engine with **automated raw CSV ingestion**, **live MLB.com confirmed starting lineups/pitchers**, 
**Vegas implied run calculations**, and **0-1 Integer Linear Programming (MILP)** optimization.
""")

# -----------------------------------------------------------------------------
# 2. Level 1: Raw DraftKings CSV Parser
# -----------------------------------------------------------------------------
def parse_raw_dk_csv(file_or_content):
    """
    Level 1: Robust parser for raw DraftKings CSV or pre-cleaned CSV.
    Strips instructions, extracts matchups, and prepares clean standard columns.
    """
    if isinstance(file_or_content, (str, bytes)):
        lines = (file_or_content if isinstance(file_or_content, str) else file_or_content.decode("utf-8", errors="ignore")).splitlines()
    else:
        lines = file_or_content.getvalue().decode("utf-8", errors="ignore").splitlines()

    header_idx = 0
    for idx, line in enumerate(lines[:15]):
        if "Position" in line and "Salary" in line and "Name" in line:
            header_idx = idx
            break

    df_raw = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    
    col_map = {
        "TeamAbbrev": "Team",
        "AvgPointsPerGame": "Projection",
        "Roster Position": "RosterPosition"
    }
    df = df_raw.rename(columns=col_map).dropna(subset=["Name", "Salary"]).copy()

    if "Position" not in df.columns:
        df["Position"] = df.get("RosterPosition", "OF")

    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(3000).astype(int)

    if "Projection" not in df.columns or df["Projection"].isnull().all():
        df["Projection"] = np.round((df["Salary"] / 1000.0) * 1.8, 2)
    else:
        df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce").fillna(0.0)

    # Extract opponent from Game Info if missing
    if "Opponent" not in df.columns and "Game Info" in df.columns:
        def extract_opp(row):
            info = str(row.get("Game Info", ""))
            team = str(row.get("Team", ""))
            if "@" in info:
                matchup = info.split()[0]
                parts = matchup.split("@")
                if len(parts) == 2:
                    return parts[1] if team == parts[0] else parts[0]
            return "UNKNOWN"
        df["Opponent"] = df.apply(extract_opp, axis=1)

    # Standardize Pitcher position
    df["Position"] = df["Position"].astype(str).str.strip().replace({"SP": "P", "RP": "P"})

    # Ensure ID exists
    if "ID" not in df.columns:
        df["ID"] = range(100000, 100000 + len(df))

    # Baseline Ownership
    if "Ownership" not in df.columns:
        pts_per_dollar = df["Projection"] / (df["Salary"] / 1000.0 + 1e-5)
        df["Ownership"] = np.clip(
            np.round(df["Projection"] * 1.8 + pts_per_dollar * 2.5 - 1.0, 1),
            1.0, 50.0
        )

    # Baseline Batting Order
    if "BattingOrder" not in df.columns:
        df["BattingOrder"] = 0
        for team in df["Team"].unique():
            h_mask = (df["Team"] == team) & (df["Position"] != "P")
            sorted_hitters = df[h_mask].sort_values(by=["Salary", "Projection"], ascending=[False, False])
            for order, idx_val in enumerate(sorted_hitters.index, 1):
                df.loc[idx_val, "BattingOrder"] = order

    # Baseline Implied Runs
    if "ImpliedRuns" not in df.columns:
        df["ImpliedRuns"] = 4.5

    df["IsConfirmedStarter"] = False
    return df

# -----------------------------------------------------------------------------
# 3. Level 2: Live MLB Stats API Integration
# -----------------------------------------------------------------------------
def normalize_name(name):
    """Normalize names for accurate cross-source matching."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = re.sub(r'[\.,\']', '', name)
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name, flags=re.IGNORECASE)
    return ' '.join(name.lower().split())

def fetch_live_mlb_starters(target_date=None):
    """
    Level 2: Queries the free MLB Stats API for confirmed starting pitchers & 1-9 lineups.
    """
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}&hydrate=lineups,probablePitcher"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return None, None, f"MLB API HTTP {resp.status_code}"
        data = resp.json()
    except Exception as e:
        return None, None, f"Connection error: {str(e)}"

    confirmed_pitchers = []
    confirmed_hitters = []

    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            teams_dict = game.get("teams", {})
            for side in ["away", "home"]:
                team_data = teams_dict.get(side, {})
                team_abbr = team_data.get("team", {}).get("abbreviation")
                sp_info = team_data.get("probablePitcher", {})
                if sp_info and sp_info.get("fullName"):
                    confirmed_pitchers.append({
                        "name": sp_info.get("fullName"),
                        "team": team_abbr,
                        "role": "SP"
                    })

            lineups_obj = game.get("lineups", {})
            for side_lineup, side_key in [("awayPlayers", "away"), ("homePlayers", "home")]:
                players = lineups_obj.get(side_lineup, [])
                team_abbr = teams_dict.get(side_key, {}).get("team", {}).get("abbreviation")
                for order_idx, player in enumerate(players, 1):
                    if player.get("fullName"):
                        confirmed_hitters.append({
                            "name": player.get("fullName"),
                            "team": team_abbr,
                            "order": order_idx
                        })

    return confirmed_pitchers, confirmed_hitters, "OK"

def sync_mlb_lineups_with_df(df_input, pitchers_list, hitters_list):
    """
    Matches live MLB lineup data to DraftKings player pool.
    """
    df = df_input.copy()
    df["NormName"] = df["Name"].apply(normalize_name)
    df["IsConfirmedStarter"] = False
    
    # Match Pitchers
    for p in pitchers_list:
        norm_p = normalize_name(p["name"])
        team_p = p.get("team")
        mask = (df["Position"] == "P") & (df["NormName"] == norm_p)
        if team_p:
            mask = mask & (df["Team"] == team_p)
            
        match_idx = df[mask].index
        if len(match_idx) == 0:
            last_p = norm_p.split()[-1]
            mask_last = (df["Position"] == "P") & (df["NormName"].str.contains(last_p))
            if team_p:
                mask_last = mask_last & (df["Team"] == team_p)
            match_idx = df[mask_last].index
            
        if len(match_idx) > 0:
            df.loc[match_idx[0], "IsConfirmedStarter"] = True
            df.loc[match_idx[0], "BattingOrder"] = 0

    # Match Hitters
    for h in hitters_list:
        norm_h = normalize_name(h["name"])
        team_h = h.get("team")
        order_h = h.get("order", 0)
        mask = (df["Position"] != "P") & (df["NormName"] == norm_h)
        if team_h:
            mask = mask & (df["Team"] == team_h)
            
        match_idx = df[mask].index
        if len(match_idx) == 0:
            last_h = norm_h.split()[-1]
            mask_last = (df["Position"] != "P") & (df["NormName"].str.contains(last_h))
            if team_h:
                mask_last = mask_last & (df["Team"] == team_h)
            match_idx = df[mask_last].index
            
        if len(match_idx) > 0:
            df.loc[match_idx[0], "IsConfirmedStarter"] = True
            df.loc[match_idx[0], "BattingOrder"] = order_h

    df = df.drop(columns=["NormName"])
    return df

# -----------------------------------------------------------------------------
# 4. Built-In Sample Slate Generator
# -----------------------------------------------------------------------------
def get_sample_slate_data():
    """Generates a realistic 3-game sample slate for immediate testing."""
    data = [
        {"Name": "Gerrit Cole", "Position": "P", "Team": "NYY", "Opponent": "BOS", "Salary": 10200, "Projection": 22.4, "Ownership": 28.5, "BattingOrder": 0, "ImpliedRuns": 4.2, "ID": 1001, "IsConfirmedStarter": True},
        {"Name": "Brayan Bello", "Position": "P", "Team": "BOS", "Opponent": "NYY", "Salary": 7400, "Projection": 14.1, "Ownership": 12.0, "BattingOrder": 0, "ImpliedRuns": 5.3, "ID": 1002, "IsConfirmedStarter": True},
        {"Name": "Anthony Volpe", "Position": "SS", "Team": "NYY", "Opponent": "BOS", "Salary": 4200, "Projection": 8.9, "Ownership": 15.2, "BattingOrder": 1, "ImpliedRuns": 5.3, "ID": 1003, "IsConfirmedStarter": True},
        {"Name": "Juan Soto", "Position": "OF", "Team": "NYY", "Opponent": "BOS", "Salary": 6100, "Projection": 12.8, "Ownership": 24.1, "BattingOrder": 2, "ImpliedRuns": 5.3, "ID": 1004, "IsConfirmedStarter": True},
        {"Name": "Aaron Judge", "Position": "OF", "Team": "NYY", "Opponent": "BOS", "Salary": 6400, "Projection": 14.2, "Ownership": 27.5, "BattingOrder": 3, "ImpliedRuns": 5.3, "ID": 1005, "IsConfirmedStarter": True},
        {"Name": "Giancarlo Stanton", "Position": "OF", "Team": "NYY", "Opponent": "BOS", "Salary": 4800, "Projection": 9.7, "Ownership": 14.8, "BattingOrder": 4, "ImpliedRuns": 5.3, "ID": 1006, "IsConfirmedStarter": True},
        {"Name": "Jazz Chisholm Jr.", "Position": "3B/2B", "Team": "NYY", "Opponent": "BOS", "Salary": 4900, "Projection": 9.4, "Ownership": 16.0, "BattingOrder": 5, "ImpliedRuns": 5.3, "ID": 1007, "IsConfirmedStarter": True},
        {"Name": "Austin Wells", "Position": "C", "Team": "NYY", "Opponent": "BOS", "Salary": 3800, "Projection": 7.8, "Ownership": 11.2, "BattingOrder": 6, "ImpliedRuns": 5.3, "ID": 1008, "IsConfirmedStarter": True},
        {"Name": "Anthony Rizzo", "Position": "1B", "Team": "NYY", "Opponent": "BOS", "Salary": 3500, "Projection": 7.1, "Ownership": 8.4, "BattingOrder": 7, "ImpliedRuns": 5.3, "ID": 1009, "IsConfirmedStarter": True},
        {"Name": "Jarren Duran", "Position": "OF", "Team": "BOS", "Opponent": "NYY", "Salary": 5300, "Projection": 10.5, "Ownership": 17.3, "BattingOrder": 1, "ImpliedRuns": 4.2, "ID": 1010, "IsConfirmedStarter": True},
        {"Name": "Rafael Devers", "Position": "3B", "Team": "BOS", "Opponent": "NYY", "Salary": 5600, "Projection": 11.2, "Ownership": 19.5, "BattingOrder": 2, "ImpliedRuns": 4.2, "ID": 1011, "IsConfirmedStarter": True},
        {"Name": "Tyler O'Neill", "Position": "OF", "Team": "BOS", "Opponent": "NYY", "Salary": 4600, "Projection": 9.0, "Ownership": 13.0, "BattingOrder": 3, "ImpliedRuns": 4.2, "ID": 1012, "IsConfirmedStarter": True},
        {"Name": "Triston Casas", "Position": "1B", "Team": "BOS", "Opponent": "NYY", "Salary": 4300, "Projection": 8.5, "Ownership": 10.5, "BattingOrder": 4, "ImpliedRuns": 4.2, "ID": 1013, "IsConfirmedStarter": True},
        {"Name": "Masataka Yoshida", "Position": "OF", "Team": "BOS", "Opponent": "NYY", "Salary": 3700, "Projection": 7.6, "Ownership": 7.8, "BattingOrder": 5, "ImpliedRuns": 4.2, "ID": 1014, "IsConfirmedStarter": True},
        {"Name": "Connor Wong", "Position": "C", "Team": "BOS", "Opponent": "NYY", "Salary": 3400, "Projection": 6.8, "Ownership": 6.2, "BattingOrder": 6, "ImpliedRuns": 4.2, "ID": 1015, "IsConfirmedStarter": True},
        {"Name": "Ceddanne Rafaela", "Position": "SS/OF", "Team": "BOS", "Opponent": "NYY", "Salary": 3600, "Projection": 7.2, "Ownership": 8.0, "BattingOrder": 7, "ImpliedRuns": 4.2, "ID": 1016, "IsConfirmedStarter": True},

        {"Name": "Yoshinobu Yamamoto", "Position": "P", "Team": "LAD", "Opponent": "COL", "Salary": 9600, "Projection": 19.8, "Ownership": 21.0, "BattingOrder": 0, "ImpliedRuns": 4.5, "ID": 1017, "IsConfirmedStarter": True},
        {"Name": "Kyle Freeland", "Position": "P", "Team": "COL", "Opponent": "LAD", "Salary": 5000, "Projection": 9.2, "Ownership": 4.1, "BattingOrder": 0, "ImpliedRuns": 7.0, "ID": 1018, "IsConfirmedStarter": True},
        {"Name": "Shohei Ohtani", "Position": "OF", "Team": "LAD", "Opponent": "COL", "Salary": 6700, "Projection": 15.8, "Ownership": 34.0, "BattingOrder": 1, "ImpliedRuns": 7.0, "ID": 1019, "IsConfirmedStarter": True},
        {"Name": "Mookie Betts", "Position": "SS/OF", "Team": "LAD", "Opponent": "COL", "Salary": 6200, "Projection": 13.5, "Ownership": 28.0, "BattingOrder": 2, "ImpliedRuns": 7.0, "ID": 1020, "IsConfirmedStarter": True},
        {"Name": "Freddie Freeman", "Position": "1B", "Team": "LAD", "Opponent": "COL", "Salary": 5800, "Projection": 12.6, "Ownership": 23.5, "BattingOrder": 3, "ImpliedRuns": 7.0, "ID": 1021, "IsConfirmedStarter": True},
        {"Name": "Teoscar Hernandez", "Position": "OF", "Team": "LAD", "Opponent": "COL", "Salary": 5100, "Projection": 11.0, "Ownership": 20.2, "BattingOrder": 4, "ImpliedRuns": 7.0, "ID": 1022, "IsConfirmedStarter": True},
        {"Name": "Max Muncy", "Position": "3B", "Team": "LAD", "Opponent": "COL", "Salary": 4700, "Projection": 9.9, "Ownership": 15.4, "BattingOrder": 5, "ImpliedRuns": 7.0, "ID": 1023, "IsConfirmedStarter": True},
        {"Name": "Will Smith", "Position": "C", "Team": "LAD", "Opponent": "COL", "Salary": 5200, "Projection": 10.4, "Ownership": 18.0, "BattingOrder": 6, "ImpliedRuns": 7.0, "ID": 1024, "IsConfirmedStarter": True},
        {"Name": "Gavin Lux", "Position": "2B", "Team": "LAD", "Opponent": "COL", "Salary": 3900, "Projection": 8.1, "Ownership": 11.0, "BattingOrder": 7, "ImpliedRuns": 7.0, "ID": 1025, "IsConfirmedStarter": True},
        {"Name": "Ezequiel Tovar", "Position": "SS", "Team": "COL", "Opponent": "LAD", "Salary": 4400, "Projection": 8.7, "Ownership": 12.1, "BattingOrder": 1, "ImpliedRuns": 4.5, "ID": 1026, "IsConfirmedStarter": True},
        {"Name": "Brenton Doyle", "Position": "OF", "Team": "COL", "Opponent": "LAD", "Salary": 4600, "Projection": 9.1, "Ownership": 14.0, "BattingOrder": 2, "ImpliedRuns": 4.5, "ID": 1027, "IsConfirmedStarter": True},
        {"Name": "Ryan McMahon", "Position": "3B", "Team": "COL", "Opponent": "LAD", "Salary": 4200, "Projection": 8.3, "Ownership": 10.2, "BattingOrder": 3, "ImpliedRuns": 4.5, "ID": 1028, "IsConfirmedStarter": True},
        {"Name": "Michael Toglia", "Position": "1B/OF", "Team": "COL", "Opponent": "LAD", "Salary": 3800, "Projection": 7.9, "Ownership": 9.5, "BattingOrder": 4, "ImpliedRuns": 4.5, "ID": 1029, "IsConfirmedStarter": True},
        {"Name": "Brendan Rodgers", "Position": "2B", "Team": "COL", "Opponent": "LAD", "Salary": 3500, "Projection": 7.0, "Ownership": 6.5, "BattingOrder": 5, "ImpliedRuns": 4.5, "ID": 1030, "IsConfirmedStarter": True},
        {"Name": "Jacob Stallings", "Position": "C", "Team": "COL", "Opponent": "LAD", "Salary": 3100, "Projection": 5.9, "Ownership": 4.2, "BattingOrder": 6, "ImpliedRuns": 4.5, "ID": 1031, "IsConfirmedStarter": True}
    ]
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 5. Sidebar Setup & Live Sync
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Step 1: Ingest Slate")
data_mode = st.sidebar.radio("Data Source", ["Use Sample Slate (3 Games)", "Upload Raw DraftKings CSV"])

if data_mode == "Upload Raw DraftKings CSV":
    uploaded_file = st.sidebar.file_uploader("Drop Raw DKSalaries.csv here", type=["csv"])
    if uploaded_file is not None:
        df = parse_raw_dk_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} players from CSV!")
    else:
        st.info("Awaiting raw DraftKings CSV. Defaulting to sample.")
        df = get_sample_slate_data()
else:
    df = get_sample_slate_data()

st.sidebar.markdown("---")
st.sidebar.header("🌐 Step 2: Live MLB.com Sync")
slate_date = st.sidebar.date_input("Slate Date", value=datetime.today())
sync_btn = st.sidebar.button("🔄 Sync Live Starting Lineups", use_container_width=True)

if sync_btn:
    date_str = slate_date.strftime("%Y-%m-%d")
    with st.sidebar.status("Fetching live MLB starting cards..."):
        p_list, h_list, status = fetch_live_mlb_starters(date_str)
        if status == "OK" and (p_list or h_list):
            df = sync_mlb_lineups_with_df(df, p_list, h_list)
            st.sidebar.success(f"Synced {len(p_list)} SPs & {len(h_list)} Starting Hitters!")
        else:
            st.sidebar.warning(f"Could not reach live API ({status}). Using roster heuristics.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Step 3: GPP Optimizer Rules")

stack_strategy = st.sidebar.selectbox(
    "Stack Architecture",
    ["5-3 Stack (Primary + Secondary)", "5-2 Stack", "4-4 Stack", "Unconstrained (No Stacking)"],
    index=0
)

col_sal1, col_sal2 = st.sidebar.columns(2)
with col_sal1:
    min_salary = st.number_input("Min Salary ($)", value=48000, min_value=40000, max_value=50000, step=100)
with col_sal2:
    max_salary = st.number_input("Max Salary ($)", value=50000, min_value=45000, max_value=50000, step=100)

no_pitcher_vs_hitter = st.sidebar.checkbox("Block Pitchers vs Opposing Hitters", value=True)
no_opposing_pitchers = st.sidebar.checkbox("Block Opposing Pitchers (Same Game)", value=True)
starters_only = st.sidebar.checkbox("Strict Starters Only (Batting Order 1–9)", value=True)

max_roster_own = st.sidebar.slider("Max Cumulative Ownership (%)", min_value=80.0, max_value=350.0, value=250.0, step=5.0)
num_lineups = st.sidebar.slider("Lineups to Generate", min_value=1, max_value=10, value=3, step=1)
max_overlap = st.sidebar.slider("Max Player Overlap Between Lineups", min_value=3, max_value=8, value=6, step=1)

# -----------------------------------------------------------------------------
# 6. Level 3: Vegas Implied Run Totals & Analytics
# -----------------------------------------------------------------------------
tab_opt, tab_vegas, tab_heatmap, tab_raw = st.tabs([
    "🚀 Lineup Optimizer", 
    "🎲 Vegas Implied Runs", 
    "🔥 Correlation Heatmap", 
    "📋 Raw Pool"
])

with tab_vegas:
    st.subheader("Level 3: Game Totals & Vegas Implied Runs Matrix")
    st.markdown("""
    Compute Implied Team Totals (**ITT**) from Game Over/Unders and Run Lines:
    $$\\text{Favorite ITT} = \\frac{\\text{Total}}{2} + \\frac{|\\text{Spread}|}{2}, \\quad \\text{Underdog ITT} = \\frac{\\text{Total}}{2} - \\frac{|\\text{Spread}|}{2}$$
    """)

    unique_teams = sorted(df["Team"].unique().tolist())
    vegas_defaults = {t: float(df[df["Team"] == t]["ImpliedRuns"].iloc[0]) if "ImpliedRuns" in df.columns else 4.5 for t in unique_teams}
    
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        st.markdown("**Edit Implied Team Totals Directly:**")
        team_itt_inputs = {}
        sub_cols = st.columns(3)
        for idx, t in enumerate(unique_teams):
            with sub_cols[idx % 3]:
                team_itt_inputs[t] = st.number_input(f"{t} Implied Runs", value=vegas_defaults.get(t, 4.5), min_value=2.0, max_value=10.0, step=0.1, key=f"itt_{t}")

        # Update dataframe with edited Vegas totals
        df["ImpliedRuns"] = df["Team"].map(team_itt_inputs)

    with col_v2:
        st.markdown("**Quick Over/Under Calculator:**")
        calc_total = st.number_input("Game Total (O/U)", value=9.0, step=0.5)
        calc_spread = st.number_input("Favorite Run Spread", value=1.5, step=0.5)
        fav_itt = round((calc_total / 2.0) + (calc_spread / 2.0), 2)
        und_itt = round((calc_total / 2.0) - (calc_spread / 2.0), 2)
        st.info(f"**Favorite ITT:** `{fav_itt} runs`\n\n**Underdog ITT:** `{und_itt} runs`")

with tab_heatmap:
    st.subheader("Team Matchup & Vegas Implied Totals Heatmap")
    
    df_heat = df.copy()
    if starters_only and "BattingOrder" in df_heat.columns:
        df_heat = df_heat[(df_heat["BattingOrder"].between(1, 9)) | (df_heat["Position"] == "P")]

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
            x=[f"#{col}" for col in pivot_proj.columns],
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

with tab_raw:
    st.subheader("Raw Slate Input Pool")
    st.dataframe(df, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. PuLP MILP Optimization Core
# -----------------------------------------------------------------------------
def optimize_dk_mlb(df_input, stack_type, min_sal, max_sal, block_p_h, block_p_p, max_own, num_lineups_to_gen, max_overlap_val, enforce_starters=True):
    df_raw = df_input.copy()
    df_active = df_raw[df_raw["Projection"] > 0].copy()

    # Strict starting pitcher & hitter pruning
    has_confirmed_sp = df_active[(df_active["Position"] == "P") & (df_active.get("IsConfirmedStarter", False) == True)]
    if not has_confirmed_sp.empty and enforce_starters:
        pitchers_filtered = has_confirmed_sp.reset_index(drop=True)
    else:
        pitchers_filtered = df_active[df_active["Position"] == "P"].groupby("Team", as_index=False).apply(
            lambda g: g.nlargest(1, "Projection")
        ).reset_index(drop=True)

    hitters_raw = df_active[df_active["Position"] != "P"]
    has_confirmed_h = hitters_raw[hitters_raw.get("IsConfirmedStarter", False) == True]
    if len(has_confirmed_h) >= 18 and enforce_starters:
        hitters_filtered = has_confirmed_h.reset_index(drop=True)
    elif enforce_starters and "BattingOrder" in hitters_raw.columns and (hitters_raw["BattingOrder"] <= 9).any():
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

        x = {}
        for p_idx in range(n_players):
            pos_str = str(df_players.loc[p_idx, "Position"])
            for slot in slots:
                if is_eligible(pos_str, slot):
                    x[p_idx, slot] = LpVariable(f"x_{p_idx}_{slot}", cat="Binary")

        y = {p_idx: LpVariable(f"y_{p_idx}", cat="Binary") for p_idx in range(n_players)}

        for p_idx in range(n_players):
            eligible_slots = [slot for slot in slots if (p_idx, slot) in x]
            prob += lpSum([x[p_idx, slot] for slot in eligible_slots]) == y[p_idx]

        for slot in slots:
            eligible_players = [p_idx for p_idx in range(n_players) if (p_idx, slot) in x]
            prob += lpSum([x[p, slot] for p in eligible_players]) == 1

        prob += lpSum([y[p_idx] for p_idx in range(n_players)]) == 10

        prob += lpSum([df_players.loc[p_idx, "Salary"] * y[p_idx] for p_idx in range(n_players)]) <= max_sal
        prob += lpSum([df_players.loc[p_idx, "Salary"] * y[p_idx] for p_idx in range(n_players)]) >= min_sal

        if "Ownership" in df_players.columns:
            prob += lpSum([df_players.loc[p_idx, "Ownership"] * y[p_idx] for p_idx in range(n_players)]) <= max_own

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

        for prev_set in previous_lineup_sets:
            prob += lpSum([y[p_idx] for p_idx in prev_set]) <= max_overlap_val

        prob += lpSum([df_players.loc[p_idx, "Projection"] * y[p_idx] for p_idx in range(n_players)])

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
                "ID": df_players.loc[p_idx, "ID"],
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

def generate_dk_upload_csv(lineups):
    """
    Formats lineups into official DraftKings bulk upload CSV format:
    P,P,C,1B,2B,3B,SS,OF,OF,OF
    """
    rows = []
    for l_df in lineups:
        p_ids = l_df[l_df["Slot"] == "P"]["ID"].tolist()
        c_id = l_df[l_df["Slot"] == "C"]["ID"].tolist()
        b1_id = l_df[l_df["Slot"] == "1B"]["ID"].tolist()
        b2_id = l_df[l_df["Slot"] == "2B"]["ID"].tolist()
        b3_id = l_df[l_df["Slot"] == "3B"]["ID"].tolist()
        ss_id = l_df[l_df["Slot"] == "SS"]["ID"].tolist()
        of_ids = l_df[l_df["Slot"] == "OF"]["ID"].tolist()
        
        row_str = f"{p_ids[0] if len(p_ids)>0 else ''},{p_ids[1] if len(p_ids)>1 else ''},{c_id[0] if len(c_id)>0 else ''},{b1_id[0] if len(b1_id)>0 else ''},{b2_id[0] if len(b2_id)>0 else ''},{b3_id[0] if len(b3_id)>0 else ''},{ss_id[0] if len(ss_id)>0 else ''},{of_ids[0] if len(of_ids)>0 else ''},{of_ids[1] if len(of_ids)>1 else ''},{of_ids[2] if len(of_ids)>2 else ''}"
        rows.append(row_str)

    csv_output = "P,P,C,1B,2B,3B,SS,OF,OF,OF\n" + "\n".join(rows)
    return csv_output

# -----------------------------------------------------------------------------
# 8. Optimizer Execution & Results UI
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
            
            # Export to DraftKings CSV Button
            dk_csv_content = generate_dk_upload_csv(lineups)
            st.download_button(
                label="📥 Download DraftKings Bulk-Upload CSV",
                data=dk_csv_content,
                file_name="DK_MLB_Optimized_Upload.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown("---")
            
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
                    l_df[["Slot", "Name", "Team", "Opponent", "Pos", "Salary", "Proj", "Own%", "Order"]].style.format({
                        "Salary": "${:,.0f}",
                        "Proj": "{:.1f}",
                        "Own%": "{:.1f}%"
                    }),
                    use_container_width=True
                )
                st.markdown("---")
