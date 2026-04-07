"""
Safety-Aware A* Route Planner
=============================
TransportAI / CS4006 Intelligent Systems — University of Limerick
Author: Yousef Al Salqan

Extends standard A* search to balance travel time against route safety using
a composite cost function:

    c(u, v) = α · time(u, v) + (1 - α) · danger(u, v)

The parameter α ∈ [0, 1] lets the user trade time for safety. The composite
heuristic h(n) = α · euclidean(n, goal) is admissible (Hart, Nilsson & Raphael,
1968), so A* still returns optimal paths under the chosen cost function.

Danger scores are informed by UK Police open data (data.police.uk),
safeareaslondon.com, and crimerate.co.uk severity-weighted Crime Risk Scores
for the period Jan 2024 – Nov 2025.
"""

import math
import heapq


# =============================================================================
# LONDON MAP — 30 nodes representing real London areas
# =============================================================================
# Each node is a recognisable London location placed at its real GPS coordinates
# (latitude, longitude), normalised to [0, 1] × [0, 1] using a Greater London
# bounding box. The graph has three zones: central (safe), inner (mixed), outer
# (high crime), reflecting London's actual crime distribution.

NODE_NAMES = {
    0: "Westminster",   1: "Covent Garden",  2: "City of London", 3: "South Bank",
    4: "Soho",          5: "Kings Cross",    6: "Tower Bridge",   7: "Waterloo",
    8: "Marylebone",    9: "Bloomsbury",    10: "Camden",        11: "Shoreditch",
    12: "Elephant & Castle", 13: "Notting Hill", 14: "Islington", 15: "Greenwich",
    16: "Clapham",     17: "Hammersmith",  18: "Hackney",        19: "Stratford",
    20: "Tottenham",   21: "Croydon",      22: "Lewisham",       23: "Barking",
    24: "Brixton",     25: "Peckham",      26: "Wood Green",     27: "Woolwich",
    28: "Edmonton",    29: "Seven Sisters",
}

# Real GPS coordinates (latitude, longitude) for each node
GPS = {
    0: (51.4947, -0.1353),  1: (51.5117, -0.1240),  2: (51.5155, -0.0922),
    3: (51.5055, -0.1160),  4: (51.5133, -0.1312),  5: (51.5317, -0.1240),
    6: (51.5055, -0.0754),  7: (51.5031, -0.1132),  8: (51.5225, -0.1544),
    9: (51.5218, -0.1278), 10: (51.5390, -0.1426), 11: (51.5264, -0.0769),
    12: (51.4946, -0.1006), 13: (51.5092, -0.1964), 14: (51.5362, -0.1032),
    15: (51.4769, -0.0005), 16: (51.4620, -0.1380), 17: (51.4928, -0.2236),
    18: (51.5450, -0.0553), 19: (51.5430, -0.0034), 20: (51.5880, -0.0720),
    21: (51.3762, -0.0986), 22: (51.4415, -0.0117), 23: (51.5362,  0.0808),
    24: (51.4613, -0.1150), 25: (51.4738, -0.0693), 26: (51.5975, -0.1096),
    27: (51.4893,  0.0654), 28: (51.6137, -0.0625), 29: (51.5833, -0.0726),
}

# Adjacency list: which nodes connect to which (based on geographic proximity
# and major transport links). Connections create competing route options where
# fast paths through dangerous areas can be compared against safer detours.
ADJACENCY = {
    0:  [1, 3, 4, 7, 8, 13, 16, 24],   1:  [0, 2, 3, 4, 9],
    2:  [1, 3, 6, 11, 14],              3:  [0, 1, 2, 6, 7, 12],
    4:  [0, 1, 8, 9, 10, 13],           5:  [9, 10, 14, 20, 26],
    6:  [2, 3, 11, 12, 15, 25],         7:  [0, 3, 12, 16, 24],
    8:  [0, 4, 9, 10, 13, 17],          9:  [1, 4, 5, 8, 10, 14],
    10: [4, 5, 8, 9, 14, 20, 26],       11: [2, 6, 14, 18, 19],
    12: [3, 6, 7, 15, 16, 22, 24, 25],  13: [0, 4, 8, 17],
    14: [2, 5, 9, 10, 11, 18],          15: [6, 12, 22, 25, 27],
    16: [0, 7, 12, 21, 24],              17: [8, 13],
    18: [11, 14, 19, 20, 23, 29],       19: [11, 18, 23, 27],
    20: [5, 10, 18, 26, 28, 29],        21: [16, 22, 25],
    22: [12, 15, 21, 25, 27],            23: [18, 19, 27],
    24: [0, 7, 12, 16, 25],              25: [6, 12, 15, 21, 22, 24],
    26: [5, 10, 20, 28, 29],             27: [15, 19, 22, 23],
    28: [20, 26, 29],                    29: [18, 20, 26, 28],
}

# Edge danger scores ∈ [0, 1] for pedestrian risk. Scores reflect violent crime
# and robbery rates from UK Police open data, weighted to favour pedestrian
# safety (so tourist areas with heavy CCTV get LOW scores despite high total
# crime, while outer high-crime corridors get HIGH scores).
EDGE_DANGERS = [
    # Central zone — well-policed, well-lit, low pedestrian risk
    (0,1,0.10),(0,3,0.08),(0,4,0.12),(0,7,0.08),(0,8,0.06),(0,13,0.08),
    (0,16,0.15),(0,24,0.35),(1,2,0.08),(1,3,0.07),(1,4,0.10),(1,9,0.06),
    (2,3,0.08),(2,6,0.07),(2,11,0.18),(2,14,0.12),(3,6,0.10),(3,7,0.06),
    (3,12,0.25),(4,8,0.08),(4,9,0.07),(4,10,0.20),(4,13,0.10),(5,9,0.10),
    (5,10,0.22),(5,14,0.15),(5,20,0.55),(5,26,0.50),(6,11,0.20),(6,12,0.22),
    (6,15,0.12),(6,25,0.35),(7,12,0.25),(7,16,0.15),(7,24,0.38),(8,9,0.06),
    (8,10,0.18),(8,13,0.05),(8,17,0.08),(9,10,0.18),(9,14,0.12),
    # Inner ring — mixed, gentrified next to deprived
    (10,14,0.20),(10,20,0.55),(10,26,0.50),(11,14,0.18),(11,18,0.40),
    (11,19,0.30),(12,15,0.25),(12,16,0.22),(12,22,0.40),(12,24,0.45),
    (12,25,0.40),(13,17,0.08),(14,18,0.35),(15,22,0.25),(15,25,0.30),
    (15,27,0.28),(16,21,0.50),(16,24,0.35),(18,19,0.35),(18,20,0.60),
    (18,23,0.55),(18,29,0.55),(19,23,0.45),(19,27,0.35),
    # Outer zone — high crime corridors (Haringey, Croydon, Lambeth)
    (20,26,0.70),(20,28,0.75),(20,29,0.65),(21,22,0.55),(21,25,0.50),
    (22,25,0.45),(22,27,0.40),(23,27,0.45),(24,25,0.42),(26,28,0.70),
    (26,29,0.65),(28,29,0.60),
]


def build_london():
    """Build the normalised coordinate map and the danger lookup dictionary."""
    # Normalise GPS to [0, 1] using Greater London bounding box
    LAT_MIN, LAT_MAX = 51.33, 51.62
    LON_MIN, LON_MAX = -0.31, 0.08
    coords = {}
    for n, (lat, lon) in GPS.items():
        x = (lon - LON_MIN) / (LON_MAX - LON_MIN)
        y = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
        coords[n] = (round(x, 4), round(y, 4))

    # Build bidirectional danger dictionary (edges are undirected)
    danger = {}
    for u, v, d in EDGE_DANGERS:
        danger[(u, v)] = d
        danger[(v, u)] = d

    return coords, danger


# =============================================================================
# A* SEARCH — composite cost: c = α·time + (1-α)·danger
# =============================================================================

def euclidean(coords, a, b):
    """Straight-line distance between two nodes (used as time metric)."""
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def find_path(coords, danger, start, goal, alpha=0.7):
    """
    A* search with a composite time-safety cost function.

    The edge cost is c(u,v) = α · time + (1-α) · danger, and the heuristic
    h(n) = α · euclidean(n, goal) is admissible: it never overestimates
    the true remaining cost (because the safety component contributes 0 to
    the heuristic, and time is bounded below by straight-line distance).
    Admissibility guarantees A* returns the optimal path under this cost
    function (Hart, Nilsson & Raphael, 1968, Theorem 1).
    """
    if start == goal:
        return [goal]

    # Priority queue ordered by f(n) = g(n) + h(n)
    # Each entry: (f_cost, node, g_cost_so_far)
    open_set = [(alpha * euclidean(coords, start, goal), start, 0.0)]
    came_from = {start: None}
    best_g = {start: 0.0}

    while open_set:
        _, current, g = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path by walking parent pointers backwards
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        # Skip stale entries (a cheaper path to this node was already found)
        if g > best_g.get(current, float('inf')):
            continue

        # Relax each neighbour
        for neighbour in ADJACENCY[current]:
            edge_time = euclidean(coords, current, neighbour)
            edge_danger = danger.get((current, neighbour), 0.0)
            edge_cost = alpha * edge_time + (1 - alpha) * edge_danger
            tentative_g = g + edge_cost

            if tentative_g < best_g.get(neighbour, float('inf')):
                best_g[neighbour] = tentative_g
                came_from[neighbour] = current
                f = tentative_g + alpha * euclidean(coords, neighbour, goal)
                heapq.heappush(open_set, (f, neighbour, tentative_g))

    return []  # No path found


def path_stats(coords, danger, path, alpha):
    """Compute total time, total danger, and composite cost for a route."""
    total_time = 0.0
    total_danger = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_time += euclidean(coords, u, v)
        total_danger += danger.get((u, v), 0.0)
    composite = alpha * total_time + (1 - alpha) * total_danger
    return total_time, total_danger, composite


# =============================================================================
# DEMO
# =============================================================================

def names(path):
    """Convert a path of node IDs to a readable place-name string."""
    return " → ".join(NODE_NAMES.get(n, str(n)) for n in path)


if __name__ == "__main__":
    coords, danger = build_london()
    n_edges = sum(len(v) for v in ADJACENCY.values()) // 2

    print("=" * 70)
    print("  TransportAI — Safety-Aware A* Route Planner")
    print("  CS4006 Intelligent Systems — University of Limerick")
    print("=" * 70)
    print(f"\nMap: {len(coords)} nodes, {n_edges} edges")
    print(f"Danger database: {len(EDGE_DANGERS)} unique edges scored")

    # ---- Primary test: Westminster → Tottenham at four α values ----
    start, goal = 0, 20
    print(f"\n{'─' * 70}")
    print(f"Route: {NODE_NAMES[start]} → {NODE_NAMES[goal]}")
    print(f"{'─' * 70}")

    alphas = [(1.0, "Pure fastest (α=1.0)"),
              (0.7, "Balanced default (α=0.7)"),
              (0.5, "Equal weight (α=0.5)"),
              (0.0, "Pure safest (α=0.0)")]

    fastest_time = None
    for a, label in alphas:
        path = find_path(coords, danger, start, goal, alpha=a)
        t, d, c = path_stats(coords, danger, path, a)
        if fastest_time is None:
            fastest_time = t
        pct = (t - fastest_time) / fastest_time * 100 if fastest_time > 0 else 0

        print(f"\n  {label}")
        print(f"    Path:         {names(path)}")
        print(f"    Node IDs:     {path}")
        print(f"    Total time:   {t:.4f}  (+{pct:.1f}% vs fastest)")
        print(f"    Total danger: {d:.4f}")
        print(f"    Composite:    {c:.4f}")
        print(f"    Edges:        {len(path) - 1}")

    # ---- Trade-off analysis: fastest vs safer (α=0.5) ----
    print(f"\n{'─' * 70}")
    print("TRADE-OFF ANALYSIS")
    print(f"{'─' * 70}")
    p_fast = find_path(coords, danger, start, goal, alpha=1.0)
    p_safe = find_path(coords, danger, start, goal, alpha=0.5)
    tf, df, _ = path_stats(coords, danger, p_fast, 1.0)
    ts, ds, _ = path_stats(coords, danger, p_safe, 0.5)

    if tf > 0 and df > 0:
        time_cost = (ts - tf) / tf * 100
        danger_saved = (df - ds) / df * 100
        print(f"\n  Fastest route:       {names(p_fast)}")
        print(f"    Time: {tf:.4f}   Danger: {df:.4f}")
        print(f"\n  Safer route (α=0.5): {names(p_safe)}")
        print(f"    Time: {ts:.4f}   Danger: {ds:.4f}")
        print(f"\n  Danger reduction: {danger_saved:.1f}%")
        print(f"  Time cost:        +{time_cost:.1f}%")

    # ---- Additional test routes at α=0.7 ----
    print(f"\n{'─' * 70}")
    print("ADDITIONAL TEST ROUTES (α=0.7)")
    print(f"{'─' * 70}")
    extra = [
        (0, 21,  "Westminster → Croydon (central to highest-crime borough)"),
        (13, 23, "Notting Hill → Barking (affluent west to deprived east)"),
        (8, 28,  "Marylebone → Edmonton (west-central to outer north)"),
        (5, 5,   "Kings Cross → Kings Cross (trivial case)"),
        (24, 22, "Brixton → Lewisham (south London high-crime corridor)"),
    ]
    for s, g, desc in extra:
        path = find_path(coords, danger, s, g, alpha=0.7)
        t, d, _ = path_stats(coords, danger, path, 0.7)
        print(f"\n  {desc}")
        print(f"    Path:   {names(path)}")
        print(f"    IDs:    {path}")
        print(f"    Time:   {t:.4f}  Danger: {d:.4f}")

    print(f"\n{'=' * 70}\n")