# -----------------------------------------------------------------------------
# Sidebar: Buzz, Role Verification & Quality Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔥 Buzz & Role Verification")

# 1. Prune Zero-Floor Punts (Removes Hawes/Hutchinson-type fillers)
min_proj_floor = st.sidebar.slider("Min Player Projection Floor (Filters Inactive Backups)", 0.0, 10.0, 7.0, 0.5)
df_slate = df_slate[df_slate["Projection"] >= min_proj_floor].copy().reset_index(drop=True)

# 2. Interactive Buzz Adjuster
with st.sidebar.expander("⚡ Adjust Specific Player 'Buzz' / Upgrades", expanded=False):
    st.caption("Apply positive multipliers for injury fill-ins, or negative multipliers for QB instability/bad matchups.")
    selected_buzz_player = st.selectbox("Select Player to Adjust", sorted(df_slate["Name"].unique()))
    buzz_adj = st.slider(f"Buzz Multiplier for {selected_buzz_player}", 0.50, 1.50, 1.00, 0.05)
    
    if buzz_adj != 1.00:
        p_idx = df_slate[df_slate["Name"] == selected_buzz_player].index
        df_slate.loc[p_idx, "Ceiling"] = np.round(df_slate.loc[p_idx, "Ceiling"] * buzz_adj, 1)
        df_slate.loc[p_idx, "Projection"] = np.round(df_slate.loc[p_idx, "Projection"] * buzz_adj, 1)
        st.info(f"Updated {selected_buzz_player}'s Ceiling to {df_slate.loc[p_idx, 'Ceiling'].values[0]} pts")
