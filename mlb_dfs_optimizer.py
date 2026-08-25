import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

# =============================================================================
# 1. ROUND 4 MILP OPTIMIZER CORE
# =============================================================================
def optimize_round4(df_slate, min_sal=48000, max_sal=50000, max_cum_own=130.0, chaser_boost=True):
    """
    Optimizes a 6-golfer DraftKings Round 4 Showdown roster using MILP.
    Applies situational chasing and leaderboard leverage weights.
    """
    df = df_slate.copy().reset_index(drop=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    col_map = {"player": "name", "golfer": "name", "own": "ownership", "proj": "r4_projection", "projection": "r4_projection"}
    df = df.rename(columns=col_map)
    
    # Situational Round 4 adjustments
    projs = df["r4_projection"].astype(float).values.copy()
    if chaser_boost and "leaderboard_pos" in df.columns:
        for i, pos in enumerate(df["leaderboard_pos"]):
            try:
                p = int(pos)
                if 4 <= p <= 15:   # Prime Chaser Tier
                    projs[i] *= 1.05
                elif p <= 2:       # Leaders (Defensive play tax)
                    projs[i] *= 0.98
            except (ValueError, TypeError):
                pass

    n = len(df)
    c = -projs
    integrality = np.ones(n)
    bounds = Bounds(np.zeros(n), np.ones(n))
    
    # Constraints:
    # 1. Exactly 6 golfers
    # 2. Min Salary <= Total Salary <= Max Salary
    # 3. Total Ownership <= Max Cumulative Ownership
    A = np.vstack([
        np.ones(n),
        df["salary"].astype(float).values,
        df["ownership"].astype(float).values if "ownership" in df.columns else np.zeros(n)
    ])
    b_l = np.array([6.0, min_sal, 0.0])
    b_u = np.array([6.0, max_sal, max_cum_own])
    
    constraints = LinearConstraint(A, b_l, b_u)
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    
    if res.status == 0:
        sel_idx = np.where(res.x > 0.5)[0]
        opt_lineup = df.iloc[sel_idx].copy()
        opt_lineup["Adjusted_R4_Proj"] = np.round(projs[sel_idx], 2)
        return opt_lineup
    else:
        print("Solver Status: Infeasible under current constraints.")
        return None

# =============================================================================
# 2. ROUND 4 RETROSPECTIVE COMPARATOR
# =============================================================================
def run_r4_retrospective(my_csv_path, win_csv_path):
    """
    Compares your Round 4 lineup against the contest winner across
    Leaderboard Positioning, Sunday Points, and SG Attribution.
    """
    def clean(df):
        d = df.copy()
        d.columns = [c.strip().lower().replace(" ", "_").replace(":", "").replace("%", "") for c in d.columns]
        mapping = {
            "player": "name", "golfer": "name", "own": "ownership", "fpts": "r4_points", 
            "actual": "r4_points", "points": "r4_points", "pos": "leaderboard_pos", 
            "sg_approach": "sg_app", "sg_putting": "sg_p"
        }
        return d.rename(columns=mapping)

    m = clean(pd.read_csv(my_csv_path))
    w = clean(pd.read_csv(win_csv_path))

    def extract_stats(df, label):
        sal = int(df["salary"].sum()) if "salary" in df.columns else 0
        own = df["ownership"].sum() if "ownership" in df.columns else 0.0
        pts = df["r4_points"].sum() if "r4_points" in df.columns else 0.0
        
        if "leaderboard_pos" in df.columns:
            pos = pd.to_numeric(df["leaderboard_pos"], errors="coerce").fillna(25)
            leaders = (pos <= 3).sum()
            chasers = ((pos >= 4) & (pos <= 15)).sum()
            backmarkers = (pos > 15).sum()
        else:
            leaders, chasers, backmarkers = 0, 0, 0
            
        sg_app = df["sg_app"].sum() if "sg_app" in df.columns else np.nan
        sg_p = df["sg_p"].sum() if "sg_p" in df.columns else np.nan
        
        return {
            "Roster": label,
            "R4 Actual FPTS": round(pts, 1),
            "Total Salary": f"${sal:,}",
            "Cum Ownership": f"{own:.1f}%",
            "Top-3 Leaders Rostered": leaders,
            "Chasers (T4-T15) Rostered": chasers,
            "Morning Waves (T16+)": backmarkers,
            "R4 SG: Approach (Total)": round(sg_app, 2) if not np.isnan(sg_app) else "N/A",
            "R4 SG: Putting (Total)": round(sg_p, 2) if not np.isnan(sg_p) else "N/A",
            "_raw": {"pts": pts, "own": own, "leaders": leaders, "chasers": chasers, "sg_app": sg_app, "sg_p": sg_p}
        }

    m_stats = extract_stats(m, "My R4 Lineup")
    w_stats = extract_stats(w, "Winning R4 Lineup")
    
    display_keys = [k for k in m_stats.keys() if not k.startswith("_")]
    comp_df = pd.DataFrame([
        {k: m_stats[k] for k in display_keys},
        {k: win_stats[k] for k in display_keys}
    ]).set_index("Roster").T
    
    print("=" * 80)
    print("⛳ ROUND 4 SHOWDOWN RETROSPECTIVE & LEADERBOARD SCORECARD")
    print("=" * 80)
    print(comp_df.to_string())
    print("-" * 80)
    
    # Strategic Process Takeaways
    print("\n🔍 SUNDAY ACTIONABLE INSIGHTS:")
    if w_stats["_raw"]["chasers"] > m_stats["_raw"]["chasers"]:
        print("• 🎯 Chaser Exposure: The winner capitalized on golfers sitting T4–T15 entering Sunday who attacked pin locations.")
    if w_stats["_raw"]["leaders"] > 2 and m_stats["_raw"]["leaders"] <= 1:
        print("• 🏆 Leaderboard Dominance: The top 3 leaders maintained control without final-round falters.")
    if not np.isnan(w_stats["_raw"]["sg_p"]) and w_stats["_raw"]["sg_p"] > 8.0:
        print(f"• 🎲 Sunday Putting Heater: The winning roster gained +{w_stats['_raw']['sg_p']:.1f} strokes putting in R4 alone (high Sunday variance).")
    print("=" * 80)

# =============================================================================
# 3. CLI DEMO RUNNER
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGA Round 4 Showdown Engine")
    parser.add_argument("--mode", choices=["optimize", "retro"], default="optimize", help="Run optimizer or retrospective")
    parser.add_argument("--slate", type=str, help="Path to R4 slate CSV")
    parser.add_argument("--my", type=str, help="Path to your R4 lineup CSV")
    parser.add_argument("--winner", type=str, help="Path to winning R4 lineup CSV")
    parser.add_argument("--demo", action="store_true", help="Run with demo data")
    args = parser.parse_args()

    if args.demo or (not args.slate and not args.my):
        # Demo Data
        demo_slate = pd.DataFrame([
            {"Name": "Scottie Scheffler", "Salary": 11500, "R4_Projection": 68.5, "Ownership": 42.0, "Leaderboard_Pos": 1, "SG_App": 2.4, "SG_P": 0.8},
            {"Name": "Xander Schauffele", "Salary": 10600, "R4_Projection": 64.0, "Ownership": 35.0, "Leaderboard_Pos": 2, "SG_App": 2.1, "SG_P": 1.2},
            {"Name": "Collin Morikawa", "Salary": 9800, "R4_Projection": 61.5, "Ownership": 26.0, "Leaderboard_Pos": 4, "SG_App": 3.1, "SG_P": -0.2},
            {"Name": "Russell Henley", "Salary": 8800, "R4_Projection": 58.0, "Ownership": 20.0, "Leaderboard_Pos": 7, "SG_App": 2.8, "SG_P": 0.5},
            {"Name": "Shane Lowry", "Salary": 8200, "R4_Projection": 54.5, "Ownership": 15.0, "Leaderboard_Pos": 10, "SG_App": 1.9, "SG_P": 0.9},
            {"Name": "Alex Noren", "Salary": 7800, "R4_Projection": 51.0, "Ownership": 13.0, "Leaderboard_Pos": 14, "SG_App": 1.4, "SG_P": 1.1},
            {"Name": "Christiaan Bezuidenhout", "Salary": 7500, "R4_Projection": 49.5, "Ownership": 11.0, "Leaderboard_Pos": 18, "SG_App": 1.0, "SG_P": 1.8},
            {"Name": "Andrew Putnam", "Salary": 6900, "R4_Projection": 46.0, "Ownership": 7.0, "Leaderboard_Pos": 24, "SG_App": 0.8, "SG_P": 1.4},
            {"Name": "Davis Thompson", "Salary": 7200, "R4_Projection": 48.0, "Ownership": 9.0, "Leaderboard_Pos": 20, "SG_App": 1.2, "SG_P": 0.6},
            {"Name": "Mark Hubbard", "Salary": 6500, "R4_Projection": 42.0, "Ownership": 4.0, "Leaderboard_Pos": 28, "SG_App": 0.3, "SG_P": -0.4},
        ])
        
        print("⚡ 1. RUNNING OPTIMAL ROUND 4 SHOWDOWN SOLVER...")
        opt = optimize_round4(demo_slate, min_sal=48000, max_sal=50000, max_cum_own=125.0)
        print(opt[["Name", "Salary", "Ownership", "Leaderboard_Pos", "Adjusted_R4_Proj"]].to_string(index=False))
        print(f"Total Salary: ${opt['Salary'].sum():,} | Total Ownership: {opt['Ownership'].sum():.1f}% | Total Proj: {opt['Adjusted_R4_Proj'].sum():.2f}")
        
        # Demo Retrospective
        demo_slate.to_csv("r4_demo_my.csv", index=False)
        demo_slate.iloc[[1, 2, 3, 4, 5, 7]].to_csv("r4_demo_win.csv", index=False)
        print("\n⚡ 2. RUNNING ROUND 4 RETRO COMPARATOR...")
        run_r4_retrospective("r4_demo_my.csv", "r4_demo_win.csv")
