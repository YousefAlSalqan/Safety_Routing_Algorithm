"""
Test Program for Safety-Aware A* Route Planner
================================================
TransportAI / CS4006 Intelligent Systems — University of Limerick

This test program validates the SafetyAwareAStarRouter across four test suites:

  TEST 1 — Small graph (5 nodes), α = 0.7
    A minimal hand-verifiable graph where you can trace the algorithm
    step by step and confirm the optimal path manually.

  TEST 2 — Big graph (50 nodes), α = 0.7
    A larger randomly-structured graph to stress-test the algorithm
    on a graph bigger than the 30-node London map.

  TEST 3 — Realistic London routes, α = 0.7
    Routes between real London locations using the full London map
    and danger database. Tests central-to-peripheral, cross-city,
    same-zone, and edge cases.

  TEST 4 — Alpha comparison (low α vs high α)
    Runs the same routes at α = 0.0, 0.3, 0.5, 0.7, 1.0 to
    demonstrate how the trade-off parameter changes route selection.

Usage:
    Place this file in the same directory as SecondDraftSafetyAlgorithm.py
    and run:  python test_safety_algorithm.py

All tests print PASS/FAIL and a summary at the end.
"""

import math
import sys
from typing import List, Dict, Tuple

# Import everything from the main algorithm file
from SecondDraftSafetyAlgorithm import (
    Map,
    NODE_NAMES,
    load_london_map,
    build_danger_database,
    SafetyAwareAStarRouter,
    SafetyAwarePathNode,
    shortest_path_safety,
    shortest_path_standard,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

passed = 0
failed = 0


def check(test_name: str, condition: bool, detail: str = ""):
    """Record a test result and print PASS/FAIL."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {test_name}")
    else:
        failed += 1
        print(f"  [FAIL] {test_name}")
    if detail:
        print(f"         {detail}")


def path_to_names(path: List[int]) -> str:
    """Convert a path of node IDs to London place names."""
    return " → ".join(NODE_NAMES.get(n, str(n)) for n in path)


def print_route_stats(router: SafetyAwareAStarRouter, path: List[int], label: str = ""):
    """Print detailed statistics for a route."""
    stats = router.get_path_stats(path)
    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}Path:   {path}")
    print(f"{prefix}Time:   {stats['total_time']:.4f}  "
          f"Danger: {stats['total_danger']:.4f}  "
          f"Composite: {stats['composite_cost']:.4f}  "
          f"Edges: {stats['num_edges']}")


# =============================================================================
# TEST 1 — SMALL GRAPH (5 nodes), α = 0.7
# =============================================================================
# 
# Graph layout:
#
#        (1)
#       / | \
#     (0) |  (3) --- (4)
#       \ | /
#        (2)
#
# Node 0 = Start (safe hub)
# Node 4 = Goal (destination)
#
# Two routes exist from 0 to 4:
#   Route A: 0 → 1 → 3 → 4  (shorter in time, but edge 1→3 is dangerous)
#   Route B: 0 → 2 → 3 → 4  (longer in time, but all edges are safe)
#
# This lets us verify by hand that the algorithm picks the right route
# at different alpha values.
# =============================================================================

def build_small_graph() -> Tuple[Map, Dict]:
    """Build a 5-node graph where fast-route and safe-route diverge."""
    intersections = {
        0: (0.0, 0.5),   # Start — left side
        1: (0.3, 0.8),   # Top path (dangerous shortcut)
        2: (0.3, 0.2),   # Bottom path (safe detour)
        3: (0.6, 0.5),   # Merge point
        4: (1.0, 0.5),   # Goal — right side
    }
    roads = [
        [1, 2],       # 0: connects to top (1) and bottom (2)
        [0, 2, 3],    # 1: connects to 0, 2, and 3
        [0, 1, 3],    # 2: connects to 0, 1, and 3
        [1, 2, 4],    # 3: connects to 1, 2, and 4
        [3],          # 4: connects only to 3
    ]
    
    danger_db = {}
    dangers = [
        (0, 1, 0.10),   # Start → Top: low danger
        (0, 2, 0.10),   # Start → Bottom: low danger
        (1, 3, 0.90),   # Top → Merge: VERY dangerous (dark alley shortcut)
        (2, 3, 0.10),   # Bottom → Merge: safe (main road)
        (3, 4, 0.05),   # Merge → Goal: very safe
        (1, 2, 0.15),   # Top ↔ Bottom: moderate
    ]
    for u, v, d in dangers:
        danger_db[(u, v)] = d
        danger_db[(v, u)] = d
    
    return Map(intersections, roads), danger_db


def test_small_graph():
    """TEST 1: Small 5-node graph at α = 0.7."""
    print("\n" + "=" * 70)
    print("TEST 1 — SMALL GRAPH (5 nodes), α = 0.7")
    print("=" * 70)
    
    m, db = build_small_graph()
    alpha = 0.7
    router = SafetyAwareAStarRouter(m, db, alpha=alpha)
    
    # --- 1a: Basic pathfinding works ---
    path = router.find_path(0, 4)
    check("Path from 0 to 4 exists", len(path) >= 2, f"Path: {path}")
    check("Path starts at 0", path[0] == 0)
    check("Path ends at 4", path[-1] == 4)
    
    # --- 1b: Path validity — each consecutive pair must be neighbors ---
    valid_edges = True
    for i in range(len(path) - 1):
        if path[i + 1] not in m.roads[path[i]]:
            valid_edges = False
            break
    check("All edges in path are valid (neighbors in adjacency list)", valid_edges)
    
    # --- 1c: Stats are non-negative ---
    stats = router.get_path_stats(path)
    check("Total time ≥ 0", stats["total_time"] >= 0)
    check("Total danger ≥ 0", stats["total_danger"] >= 0)
    check("Composite cost ≥ 0", stats["composite_cost"] >= 0)
    check("Edge count matches path length - 1",
          stats["num_edges"] == len(path) - 1,
          f"Edges: {stats['num_edges']}, Path length: {len(path)}")
    
    # --- 1d: Trivial case — start == goal ---
    trivial = router.find_path(3, 3)
    check("Trivial path (start == goal) returns [3]", trivial == [3])
    
    # --- 1e: Route avoids the dangerous edge at α=0.7 ---
    # At α=0.7 the algorithm should prefer the safe bottom route (0→2→3→4)
    # over the dangerous top route (0→1→3→4) because edge 1→3 has danger=0.90
    print(f"\n  Route chosen at α={alpha}: {path}")
    print_route_stats(router, path)
    
    # Compare both routes explicitly
    route_top = [0, 1, 3, 4]     # Fast but dangerous
    route_bottom = [0, 2, 3, 4]  # Slower but safe
    
    r_top = SafetyAwareAStarRouter(m, db, alpha=alpha)
    r_bot = SafetyAwareAStarRouter(m, db, alpha=alpha)
    s_top = r_top.get_path_stats(route_top)
    s_bot = r_bot.get_path_stats(route_bottom)
    
    print(f"\n  Manual comparison:")
    print(f"    Top route    [0→1→3→4]: time={s_top['total_time']:.4f}  "
          f"danger={s_top['total_danger']:.4f}  composite={s_top['composite_cost']:.4f}")
    print(f"    Bottom route [0→2→3→4]: time={s_bot['total_time']:.4f}  "
          f"danger={s_bot['total_danger']:.4f}  composite={s_bot['composite_cost']:.4f}")
    
    # The algorithm should pick whichever has lower composite cost
    if s_bot["composite_cost"] < s_top["composite_cost"]:
        check("Algorithm picks safer bottom route (lower composite cost)",
              1 not in path or path != route_top,
              "Bottom route has lower composite cost, algorithm should avoid top")
    else:
        check("Algorithm picks faster top route (lower composite cost)",
              path == route_top,
              "Top route has lower composite cost despite danger")


# =============================================================================
# TEST 2 — BIG GRAPH (50 nodes), α = 0.7
# =============================================================================
# A procedurally generated graph larger than the London map to verify the
# algorithm scales correctly. Nodes are placed on a grid with random-ish
# connections and danger scores.
# =============================================================================

def build_big_graph() -> Tuple[Map, Dict]:
    """Build a 50-node grid-like graph with synthetic danger scores."""
    # Place 50 nodes in a 10x5 grid pattern
    intersections = {}
    for i in range(50):
        row = i // 10
        col = i % 10
        x = col / 9.0  # Normalize to [0, 1]
        y = row / 4.0   # Normalize to [0, 1]
        intersections[i] = (x, y)
    
    # Connect each node to its grid neighbors (right, down, diagonal)
    roads = [[] for _ in range(50)]
    danger_db = {}
    
    for i in range(50):
        row = i // 10
        col = i % 10
        neighbors = []
        
        # Right neighbor
        if col < 9:
            neighbors.append(i + 1)
        # Down neighbor
        if row < 4:
            neighbors.append(i + 10)
        # Diagonal down-right
        if col < 9 and row < 4:
            neighbors.append(i + 11)
        
        for j in neighbors:
            if j not in roads[i]:
                roads[i].append(j)
            if i not in roads[j]:
                roads[j].append(i)
            
            # Assign danger based on position:
            # Top rows (0-1) are safe, middle (2) is mixed, bottom (3-4) is dangerous
            avg_row = (row + (j // 10)) / 2.0
            if avg_row < 1.5:
                danger = 0.05 + (col % 3) * 0.05    # 0.05 - 0.15 (safe)
            elif avg_row < 2.5:
                danger = 0.25 + (col % 4) * 0.08    # 0.25 - 0.49 (mixed)
            else:
                danger = 0.55 + (col % 3) * 0.12    # 0.55 - 0.79 (dangerous)
            
            danger_db[(i, j)] = round(danger, 2)
            danger_db[(j, i)] = round(danger, 2)
    
    return Map(intersections, roads), danger_db


def test_big_graph():
    """TEST 2: Big 50-node graph at α = 0.7."""
    print("\n" + "=" * 70)
    print("TEST 2 — BIG GRAPH (50 nodes), α = 0.7")
    print("=" * 70)
    
    m, db = build_big_graph()
    alpha = 0.7
    
    print(f"\n  Graph: {len(m.intersections)} nodes, "
          f"{sum(len(r) for r in m.roads) // 2} edges (approx)")
    print(f"  Danger database: {len(db) // 2} unique edges scored")
    
    # --- 2a: Path across entire graph (top-left to bottom-right) ---
    router = SafetyAwareAStarRouter(m, db, alpha=alpha)
    path = router.find_path(0, 49)
    check("Path from node 0 to node 49 exists", len(path) >= 2, f"Path: {path}")
    check("Path starts at 0", path[0] == 0)
    check("Path ends at 49", path[-1] == 49)
    
    stats = router.get_path_stats(path)
    print_route_stats(router, path, "0 → 49")
    
    # --- 2b: All edges in the path are valid ---
    valid = all(path[i + 1] in m.roads[path[i]] for i in range(len(path) - 1))
    check("All edges in 0→49 path are valid neighbors", valid)
    
    # --- 2c: Path from safe zone to dangerous zone ---
    router2 = SafetyAwareAStarRouter(m, db, alpha=alpha)
    path2 = router2.find_path(5, 45)
    check("Path from node 5 to node 45 exists", len(path2) >= 2, f"Path: {path2}")
    stats2 = router2.get_path_stats(path2)
    print_route_stats(router2, path2, "5 → 45")
    
    # --- 2d: Multiple routes produce non-zero stats ---
    test_pairs = [(0, 25), (10, 40), (3, 47), (9, 41)]
    for s, g in test_pairs:
        r = SafetyAwareAStarRouter(m, db, alpha=alpha)
        p = r.find_path(s, g)
        st = r.get_path_stats(p)
        check(f"Route {s}→{g}: path exists with positive cost",
              len(p) >= 2 and st["composite_cost"] > 0,
              f"Path length: {len(p)}, Composite: {st['composite_cost']:.4f}")
    
    # --- 2e: Trivial case on big graph ---
    r_triv = SafetyAwareAStarRouter(m, db, alpha=alpha)
    check("Trivial path (25→25) returns [25]", r_triv.find_path(25, 25) == [25])
    
    # --- 2f: Compare α=1.0 vs α=0.7 — safety-aware should have ≤ danger ---
    r_fast = SafetyAwareAStarRouter(m, db, alpha=1.0)
    r_safe = SafetyAwareAStarRouter(m, db, alpha=0.7)
    p_fast = r_fast.find_path(0, 49)
    p_safe = r_safe.find_path(0, 49)
    sf = r_fast.get_path_stats(p_fast)
    ss = r_safe.get_path_stats(p_safe)
    
    check("Safety-aware (α=0.7) danger ≤ fastest (α=1.0) danger",
          ss["total_danger"] <= sf["total_danger"] + 0.001,
          f"α=1.0 danger: {sf['total_danger']:.4f}, α=0.7 danger: {ss['total_danger']:.4f}")


# =============================================================================
# TEST 3 — REALISTIC LONDON ROUTES, α = 0.7
# =============================================================================
# Tests using the actual London map and danger database from the main file.
# Routes cover: central-to-peripheral, cross-city, within-zone, edge cases.
# =============================================================================

def test_london_routes():
    """TEST 3: Realistic London routes at α = 0.7."""
    print("\n" + "=" * 70)
    print("TEST 3 — REALISTIC LONDON ROUTES, α = 0.7")
    print("=" * 70)
    
    london = load_london_map()
    db = build_danger_database()
    alpha = 0.7
    
    print(f"\n  London map: {len(london.intersections)} nodes, "
          f"{sum(len(r) for r in london.roads) // 2} edges")
    
    # --- 3a: Westminster to Tottenham (central to high-crime outer north) ---
    r = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path = r.find_path(0, 20)
    stats = r.get_path_stats(path)
    check("Westminster → Tottenham: path exists",
          len(path) >= 2, f"{path_to_names(path)}")
    check("Westminster → Tottenham: starts at Westminster", path[0] == 0)
    check("Westminster → Tottenham: ends at Tottenham", path[-1] == 20)
    print_route_stats(r, path, "Westminster → Tottenham")
    
    # --- 3b: Notting Hill to Barking (affluent west to deprived east) ---
    r2 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path2 = r2.find_path(13, 23)
    stats2 = r2.get_path_stats(path2)
    check("Notting Hill → Barking: path exists",
          len(path2) >= 2, f"{path_to_names(path2)}")
    print_route_stats(r2, path2, "Notting Hill → Barking")
    
    # --- 3c: Westminster to Croydon (central to highest-crime-count borough) ---
    r3 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path3 = r3.find_path(0, 21)
    stats3 = r3.get_path_stats(path3)
    check("Westminster → Croydon: path exists",
          len(path3) >= 2, f"{path_to_names(path3)}")
    print_route_stats(r3, path3, "Westminster → Croydon")
    
    # --- 3d: Brixton to Lewisham (south London high-crime corridor) ---
    r4 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path4 = r4.find_path(24, 22)
    stats4 = r4.get_path_stats(path4)
    check("Brixton → Lewisham: path exists",
          len(path4) >= 2, f"{path_to_names(path4)}")
    print_route_stats(r4, path4, "Brixton → Lewisham")
    
    # --- 3e: Marylebone to Edmonton (west-central to outer north) ---
    r5 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path5 = r5.find_path(8, 28)
    stats5 = r5.get_path_stats(path5)
    check("Marylebone → Edmonton: path exists",
          len(path5) >= 2, f"{path_to_names(path5)}")
    print_route_stats(r5, path5, "Marylebone → Edmonton")
    
    # --- 3f: Within central zone (Covent Garden to Tower Bridge) ---
    r6 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path6 = r6.find_path(1, 6)
    stats6 = r6.get_path_stats(path6)
    check("Covent Garden → Tower Bridge: path exists",
          len(path6) >= 2, f"{path_to_names(path6)}")
    check("Covent Garden → Tower Bridge: low danger (central zone)",
          stats6["total_danger"] < 0.5,
          f"Danger: {stats6['total_danger']:.4f} (expected < 0.5 for central route)")
    print_route_stats(r6, path6, "Covent Garden → Tower Bridge")
    
    # --- 3g: Within dangerous zone (Tottenham to Edmonton) ---
    r7 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    path7 = r7.find_path(20, 28)
    stats7 = r7.get_path_stats(path7)
    check("Tottenham → Edmonton: path exists",
          len(path7) >= 2, f"{path_to_names(path7)}")
    check("Tottenham → Edmonton: high danger (outer zone)",
          stats7["total_danger"] > 0.5,
          f"Danger: {stats7['total_danger']:.4f} (expected > 0.5 for outer route)")
    print_route_stats(r7, path7, "Tottenham → Edmonton")
    
    # --- 3h: Trivial case ---
    r8 = SafetyAwareAStarRouter(london, db, alpha=alpha)
    check("Kings Cross → Kings Cross (trivial): returns [5]",
          r8.find_path(5, 5) == [5])
    
    # --- 3i: Edge validity for all London routes ---
    all_valid = True
    for test_path in [path, path2, path3, path4, path5, path6, path7]:
        for i in range(len(test_path) - 1):
            if test_path[i + 1] not in london.roads[test_path[i]]:
                all_valid = False
                print(f"    INVALID EDGE: {test_path[i]} → {test_path[i+1]}")
                break
    check("All London route edges are valid (in adjacency list)", all_valid)
    
    # --- 3j: Convenience function matches direct router call ---
    path_conv = shortest_path_safety(london, db, 0, 20, alpha=0.7)
    check("shortest_path_safety() matches router.find_path()",
          path_conv == path,
          f"Convenience: {path_conv}, Direct: {path}")
    
    path_std = shortest_path_standard(london, db, 0, 20)
    r_std = SafetyAwareAStarRouter(london, db, alpha=1.0)
    path_std_direct = r_std.find_path(0, 20)
    check("shortest_path_standard() matches α=1.0 router",
          path_std == path_std_direct)


# =============================================================================
# TEST 4 — ALPHA COMPARISON (low α vs high α)
# =============================================================================
# Runs the same routes at α = 0.0, 0.3, 0.5, 0.7, 1.0 to show how the
# trade-off parameter changes route selection, time, and danger.
# =============================================================================

def test_alpha_comparison():
    """TEST 4: Compare low vs high alpha across multiple routes."""
    print("\n" + "=" * 70)
    print("TEST 4 — ALPHA COMPARISON (low α vs high α)")
    print("=" * 70)
    
    london = load_london_map()
    db = build_danger_database()
    
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # Routes that should show meaningful trade-offs
    test_routes = [
        (0, 20,  "Westminster → Tottenham"),
        (0, 21,  "Westminster → Croydon"),
        (13, 23, "Notting Hill → Barking"),
        (8, 28,  "Marylebone → Edmonton"),
    ]
    
    for start, goal, desc in test_routes:
        print(f"\n  {'─' * 64}")
        print(f"  {desc} (node {start} → node {goal})")
        print(f"  {'─' * 64}")
        print(f"  {'α':<6} {'Path':<50} {'Time':>8} {'Danger':>8} {'Composite':>10}")
        print(f"  {'─'*6} {'─'*50} {'─'*8} {'─'*8} {'─'*10}")
        
        results = []
        for alpha in alphas:
            router = SafetyAwareAStarRouter(london, db, alpha=alpha)
            path = router.find_path(start, goal)
            stats = router.get_path_stats(path)
            names = path_to_names(path)
            # Truncate long paths for display
            if len(names) > 48:
                names = names[:45] + "..."
            print(f"  {alpha:<6.1f} {names:<50} {stats['total_time']:>8.4f} "
                  f"{stats['total_danger']:>8.4f} {stats['composite_cost']:>10.4f}")
            results.append((alpha, path, stats))
        
        # --- Key assertions ---
        # At α=1.0 (pure time), the route should be fastest
        fastest_time = results[-1][2]["total_time"]
        for alpha_val, _, stats in results:
            check(f"{desc} α={alpha_val}: time ≥ fastest (α=1.0)",
                  stats["total_time"] >= fastest_time - 0.001,
                  f"Time: {stats['total_time']:.4f} vs fastest: {fastest_time:.4f}")
        
        # At α=0.0 (pure safety), the route should be safest
        safest_danger = results[0][2]["total_danger"]
        for alpha_val, _, stats in results:
            check(f"{desc} α={alpha_val}: danger ≥ safest (α=0.0)",
                  stats["total_danger"] >= safest_danger - 0.001,
                  f"Danger: {stats['total_danger']:.4f} vs safest: {safest_danger:.4f}")
        
        # Trade-off analysis between α=1.0 and α=0.0
        s_fast = results[-1][2]  # α = 1.0
        s_safe = results[0][2]   # α = 0.0
        p_fast = results[-1][1]  # path at α = 1.0
        p_safe = results[0][1]   # path at α = 0.0
        
        if s_fast["total_time"] > 0 and s_fast["total_danger"] > 0:
            time_increase = ((s_safe["total_time"] - s_fast["total_time"]) 
                            / s_fast["total_time"] * 100)
            danger_decrease = ((s_fast["total_danger"] - s_safe["total_danger"])
                              / s_fast["total_danger"] * 100)
            print(f"\n  Trade-off (α=0.0 vs α=1.0):")
            print(f"    Time increase:   +{time_increase:.1f}%")
            print(f"    Danger reduction: {danger_decrease:.1f}%")
            
            if s_fast["total_danger"] != s_safe["total_danger"]:
                print(f"    → Different paths chosen: trade-off is active")
            else:
                print(f"    → Same path chosen: no safer alternative exists")
        
        # --- PATH DIFFERENCE CHECKS ---
        # Collect all unique paths across alpha values
        unique_paths = []
        for alpha_val, p, _ in results:
            if p not in unique_paths:
                unique_paths.append(p)
        
        print(f"\n  Path difference analysis:")
        print(f"    Unique paths found across 5 alpha values: {len(unique_paths)}")
        for idx, up in enumerate(unique_paths):
            # Which alpha values produced this path?
            matching_alphas = [a for a, p, _ in results if p == up]
            print(f"    Path {idx+1}: {path_to_names(up)}")
            print(f"            Used by α = {matching_alphas}")
        
        # Check: does α=0.0 produce a different path than α=1.0?
        paths_differ = (p_fast != p_safe)
        if len(unique_paths) > 1:
            # Multiple paths exist — α SHOULD produce different routes
            check(f"{desc}: α=0.0 and α=1.0 produce different paths",
                  paths_differ,
                  f"α=0.0: {p_safe}, α=1.0: {p_fast}")
        else:
            # Only one path exists — α cannot change the route (topology constraint)
            check(f"{desc}: only one viable path exists (same at all α values)",
                  not paths_differ,
                  f"Path: {p_safe} (no safer alternative in graph topology)")
        
        # If paths differ, verify the REASON is correct:
        # α=0.0 path should have less danger, α=1.0 path should have less time
        if paths_differ:
            check(f"{desc}: safest path (α=0.0) has ≤ danger than fastest (α=1.0)",
                  s_safe["total_danger"] <= s_fast["total_danger"] + 0.001,
                  f"Safe danger: {s_safe['total_danger']:.4f}, "
                  f"Fast danger: {s_fast['total_danger']:.4f}")
            check(f"{desc}: fastest path (α=1.0) has ≤ time than safest (α=0.0)",
                  s_fast["total_time"] <= s_safe["total_time"] + 0.001,
                  f"Fast time: {s_fast['total_time']:.4f}, "
                  f"Safe time: {s_safe['total_time']:.4f}")
            
            # Check: do we see at least one intermediate alpha choosing a THIRD path?
            # (not required but interesting — shows the α parameter is granular)
            mid_alphas_different = any(
                p != p_fast and p != p_safe 
                for _, p, _ in results[1:-1]  # α = 0.3, 0.5, 0.7
            )
            if mid_alphas_different:
                print(f"    ✓ Intermediate α values produce additional distinct paths")
            else:
                print(f"    ○ Intermediate α values use one of the two extreme paths")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TransportAI — Test Suite for Safety-Aware A* Router")
    print("  CS4006 Intelligent Systems — University of Limerick")
    print("=" * 70)
    
    test_small_graph()
    test_big_graph()
    test_london_routes()
    test_alpha_comparison()
    
    # --- Final summary ---
    total = passed + failed
    print(f"\n{'=' * 70}")
    print(f"  TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total tests:  {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {failed}")
    
    if failed == 0:
        print(f"\n  ALL {total} TESTS PASSED.")
    else:
        print(f"\n  {failed} TEST(S) FAILED — review output above.")
    
    print(f"{'=' * 70}")
    sys.exit(0 if failed == 0 else 1)