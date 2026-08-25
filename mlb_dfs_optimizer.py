import itertools
import numpy as np
import pandas as pd

def analyze_bmw_retrospective(slate_projections_csv, my_roster_names, winning_roster_names):
    """
    Evaluates the exact combinatorial rank of your lineup vs. the winning lineup
    across all salary-valid combinations in the BMW Championship field.
    """
    df = pd.read_csv(slate_projections_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Normalize column names
    col_map = {"player": "name", "golfer": "name", "fpts": "projection", "proj": "projection", "own": "ownership"}
    df = df.rename(columns=col_map)
    
    names = df["name"].values
    salaries = df["salary"].values
    projections = df["projection"].values
    ownerships = df["ownership"].values if "ownership" in df.columns else np.zeros(len(df))
    actuals = df["actual_points"].values if "actual_points" in df.columns else np.zeros(len(df))
    
    name_to_idx = {name.lower(): i for i, name in enumerate(names)}
    
    my_indices = tuple(sorted([name_to_idx[n.lower()] for n in my_roster_names]))
    win_indices = tuple(sorted([name_to_idx[n.lower()] for n in winning_roster_names]))
    
    print("⚡ Generating and evaluating all salary-valid 6-golfer combinations...")
    
    # Generate all combinations of 6 from 50 players (15,890,700 combinations)
    all_combs = list(itertools.combinations(range(len(df)), 6))
    comb_array = np.array(all_combs, dtype=np.int32)
    
    # Vectorized salary & projection evaluation
    comb_salaries = salaries[comb_array].sum(axis=1)
    
    # Filter to salary-valid rosters ($45,000 - $50,000)
    valid_mask = (comb_salaries <= 50000) & (comb_salaries >= 45000)
    valid_combs = comb_array[valid_mask]
    valid_salaries = comb_salaries[valid_mask]
    valid_projs = projections[valid_combs].sum(axis=1)
    valid_owns = ownerships[valid_combs].sum(axis=1)
    valid_actuals = actuals[valid_combs].sum(axis=1)
    
    total_valid = len(valid_combs)
    print(f"✅ Total Salary-Valid Lineups: {total_valid:,} / {len(all_combs):,}")
    
    # Sort by projected points descending
    sort_idx = np.argsort(-valid_projs)
    sorted_combs = valid_combs[sort_idx]
    sorted_projs = valid_projs[sort_idx]
    
    # Find exact ranks
    comb_tuples = [tuple(c) for c in sorted_combs]
    my_rank = comb_tuples.index(my_indices) + 1 if my_indices in comb_tuples else None
    win_rank = comb_tuples.index(win_indices) + 1 if win_indices in comb_tuples else None
    
    # Lineup metrics summary
    def get_lineup_summary(indices, label, rank):
        sal = salaries[list(indices)].sum()
        proj = projections[list(indices)].sum()
        own = ownerships[list(indices)].sum()
        act = actuals[list(indices)].sum()
        pct = (rank / total_valid) * 100 if rank else np.nan
        return {
            "Roster": label,
            "Pre-Lock Optimizer Rank": f"#{rank:,} (Top {pct:.2f}%)" if rank else "Invalid / Filtered",
            "Projected Points": f"{proj:.2f} pts",
            "Actual Points": f"{act:.2f} pts",
            "Salary Spent": f"${sal:,}",
            "Cumulative Ownership": f"{own:.1f}%",
            "Optimality Gap to #1 Optimal": f"-{sorted_projs[0] - proj:.2f} pts"
        }
        
    summary_df = pd.DataFrame([
        get_lineup_summary(sorted_combs[0], "Pre-Tournament #1 Optimal", 1),
        get_lineup_summary(my_indices, "My Lineup", my_rank),
        get_lineup_summary(win_indices, "Contest Winning Lineup", win_rank)
    ]).set_index("Roster").T
    
    print("\n" + "=" * 80)
    print("🏆 BMW CHAMPIONSHIP RETROSPECTIVE SCORECARD")
    print("=" * 80)
    print(summary_df.to_string())
    print("=" * 80)
    
    return summary_df
