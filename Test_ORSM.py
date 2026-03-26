"""
test_osrm.py — More tests for the OSRM-style algorithms
"""

from osrm_style import dijkstra, bidirectional_dijkstra, ch_preprocess, ch_query


def run_test(name, graph, start, goal):
    print(f"\n{'=' * 55}")
    print(f"TEST: {name} ({start} → {goal})")
    print(f"{'=' * 55}")

    path, cost, expanded = dijkstra(graph, start, goal)
    print(f"\n  Dijkstra:              {' → '.join(path)}  cost={cost}  expanded={expanded}")

    path, cost, expanded = bidirectional_dijkstra(graph, start, goal)
    print(f"  Bidirectional Dijkstra: {' → '.join(path)}  cost={cost}  expanded={expanded}")

    ch_graph, rank = ch_preprocess(graph)
    path, cost, expanded = ch_query(ch_graph, rank, start, goal)
    if path:
        print(f"  Contraction Hierarchy:  {' → '.join(path)}  cost={cost}  expanded={expanded}")
    else:
        print(f"  Contraction Hierarchy:  No path found")


# ── Test 1: CS4006 Lecture 4 graph ──────────────────────────────
graph_lecture = {
    'A': {'B': 1, 'C': 7},
    'B': {'A': 1, 'D': 9, 'E': 1},
    'C': {'A': 7, 'E': 5},
    'D': {'B': 9, 'E': 5, 'F': 2},
    'E': {'B': 1, 'C': 5, 'D': 5, 'G': 3},
    'F': {'D': 2, 'H': 5},
    'G': {'E': 3, 'H': 5},
    'H': {'F': 5, 'G': 5},
}
run_test("CS4006 Lecture 4", graph_lecture, 'A', 'H')


# ── Test 2: Linear chain (worst case for bidirectional) ─────────
# A --1-- B --1-- C --1-- D --1-- E --1-- F
graph_chain = {
    'A': {'B': 1},
    'B': {'A': 1, 'C': 1},
    'C': {'B': 1, 'D': 1},
    'D': {'C': 1, 'E': 1},
    'E': {'D': 1, 'F': 1},
    'F': {'E': 1},
}
run_test("Linear chain", graph_chain, 'A', 'F')


# ── Test 3: Two paths — short but expensive vs long but cheap ──
#    A --1-- B --1-- C
#    |               |
#    +--10-- D --1---+
#
# A→C via B: cost 2 (short)
# A→C via D: cost 11 (long)
graph_two_paths = {
    'A': {'B': 1, 'D': 10},
    'B': {'A': 1, 'C': 1},
    'C': {'B': 1, 'D': 1},
    'D': {'A': 10, 'C': 1},
}
run_test("Two paths (short vs long)", graph_two_paths, 'A', 'C')


# ── Test 4: Grid graph (realistic road network shape) ───────────
#    A --1-- B --1-- C
#    |       |       |
#    1       5       1
#    |       |       |
#    D --1-- E --1-- F
#    |       |       |
#    1       1       1
#    |       |       |
#    G --1-- H --1-- I
#
# Fastest A→I: A→D→G→H→I (cost 4) — avoids expensive B→E edge
graph_grid = {
    'A': {'B': 1, 'D': 1},
    'B': {'A': 1, 'C': 1, 'E': 5},
    'C': {'B': 1, 'F': 1},
    'D': {'A': 1, 'E': 1, 'G': 1},
    'E': {'B': 5, 'D': 1, 'F': 1, 'H': 1},
    'F': {'C': 1, 'E': 1, 'I': 1},
    'G': {'D': 1, 'H': 1},
    'H': {'G': 1, 'E': 1, 'I': 1},
    'I': {'H': 1, 'F': 1},
}
run_test("3x3 Grid", graph_grid, 'A', 'I')


# ── Test 5: Star graph (one hub, many spokes) ──────────────────
#      B
#      |2
# C-3-HUB-1-A
#      |4
#      D
#
# Everything goes through HUB
graph_star = {
    'A':   {'HUB': 1},
    'B':   {'HUB': 2},
    'C':   {'HUB': 3},
    'D':   {'HUB': 4},
    'HUB': {'A': 1, 'B': 2, 'C': 3, 'D': 4},
}
run_test("Star (all through hub)", graph_star, 'A', 'D')


# ── Test 6: Same start and goal ─────────────────────────────────
run_test("Same node (A → A)", graph_lecture, 'A', 'A')


# ── Test 7: Larger graph — 12 nodes, simulating a small city ────
#
#  1---2---3---4
#  |   |   |   |
#  5---6---7---8
#  |   |   |   |
#  9--10--11--12
#
graph_city = {
    '1':  {'2': 2, '5': 3},
    '2':  {'1': 2, '3': 2, '6': 4},
    '3':  {'2': 2, '4': 2, '7': 1},
    '4':  {'3': 2, '8': 3},
    '5':  {'1': 3, '6': 1, '9': 2},
    '6':  {'2': 4, '5': 1, '7': 1, '10': 2},
    '7':  {'3': 1, '6': 1, '8': 1, '11': 2},
    '8':  {'4': 3, '7': 1, '12': 2},
    '9':  {'5': 2, '10': 1},
    '10': {'6': 2, '9': 1, '11': 1},
    '11': {'7': 2, '10': 1, '12': 1},
    '12': {'8': 2, '11': 1},
}
run_test("Small city (1 → 12)", graph_city, '1', '12')
run_test("Small city (9 → 4)", graph_city, '9', '4')
run_test("Small city (1 → 8)", graph_city, '1', '8')