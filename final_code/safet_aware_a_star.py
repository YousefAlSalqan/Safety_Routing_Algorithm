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

Map data is stored in separate files under the maps/ folder. To add a new
map, create two files in maps/:
    maps/<name>_map.py      — defines BOUNDING_BOX, NODE_NAMES, GPS, ADJACENCY
    maps/<name>_dangers.py  — defines EDGE_DANGERS

Then call load_map("<name>") to use it.
"""

import math
import heapq
import importlib


# =============================================================================
# MAP LOADER — generic loader that imports any map module by name
# =============================================================================

def load_map(name):
    """Load a map and its danger database from the maps/ folder.

    Looks for two modules:
        maps.<name>_map      — must define BOUNDING_BOX, NODE_NAMES, GPS, ADJACENCY
        maps.<name>_dangers  — must define EDGE_DANGERS

    GPS coordinates are normalised to [0, 1] using the map's bounding box, and
    edge danger entries are expanded into a bidirectional dictionary.

    Returns:
        A dictionary with keys:
            names      — {node_id: place_name}
            coords     — {node_id: (x, y)} normalised to [0, 1]
            adjacency  — {node_id: [neighbour_ids]}
            danger     — {(u, v): score} bidirectional
    """
    map_module = importlib.import_module(f"maps.{name}_map")
    danger_module = importlib.import_module(f"maps.{name}_dangers")

    bbox = map_module.BOUNDING_BOX
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]

    # Normalise GPS coordinates to [0, 1] using the bounding box
    coords = {}
    for node_id, (lat, lon) in map_module.GPS.items():
        x = (lon - lon_min) / (lon_max - lon_min)
        y = (lat - lat_min) / (lat_max - lat_min)
        coords[node_id] = (round(x, 4), round(y, 4))

    # Build bidirectional danger dictionary (edges are undirected)
    danger = {}
    for u, v, d in danger_module.EDGE_DANGERS:
        danger[(u, v)] = d
        danger[(v, u)] = d

    return {
        "names":     map_module.NODE_NAMES,
        "coords":    coords,
        "adjacency": map_module.ADJACENCY,
        "danger":    danger,
    }


# =============================================================================
# A* SEARCH — composite cost: c = α·time + (1-α)·danger
# =============================================================================

def euclidean(coords, a, b):
    """Straight-line distance between two nodes (used as time metric)."""
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def find_path(graph, start, goal, alpha=0.7):
    """
    A* search with a composite time-safety cost function.

    The edge cost is c(u,v) = α · time + (1-α) · danger, and the heuristic
    h(n) = α · euclidean(n, goal) is admissible: it never overestimates the
    true remaining cost (because the safety component contributes 0 to the
    heuristic, and time is bounded below by straight-line distance).
    Admissibility guarantees A* returns the optimal path under this cost
    function (Hart, Nilsson & Raphael, 1968, Theorem 1).
    """
    coords = graph["coords"]
    adjacency = graph["adjacency"]
    danger = graph["danger"]

    if start == goal:
        return [goal]

    # Priority queue ordered by f(n) = g(n) + h(n)
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
        for neighbour in adjacency[current]:
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


def path_stats(graph, path, alpha):
    """Compute total time, total danger, and composite cost for a route."""
    coords = graph["coords"]
    danger = graph["danger"]

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

def names(graph, path):
    """Convert a path of node IDs to a readable place-name string."""
    return " → ".join(graph["names"].get(n, str(n)) for n in path)


if __name__ == "__main__":
    # Load the London map and its danger database from the maps/ folder
    london = load_map("london")
    n_edges = sum(len(v) for v in london["adjacency"].values()) // 2

    print("=" * 70)
    print("  TransportAI — Safety-Aware A* Route Planner")
    print("  CS4006 Intelligent Systems — University of Limerick")
    print("=" * 70)
    print(f"\nMap: {len(london['coords'])} nodes, {n_edges} edges")
    print(f"Danger database: {len(london['danger']) // 2} unique edges scored")

    # ---- Primary test: Westminster → Tottenham at four α values ----
    start, goal = 0, 20
    print(f"\n{'─' * 70}")
    print(f"Route: {london['names'][start]} → {london['names'][goal]}")
    print(f"{'─' * 70}")

    alphas = [(1.0, "Pure fastest (α=1.0)"),
              (0.7, "Balanced default (α=0.7)"),
              (0.5, "Equal weight (α=0.5)"),
              (0.0, "Pure safest (α=0.0)")]

    fastest_time = None
    for a, label in alphas:
        path = find_path(london, start, goal, alpha=a)
        t, d, c = path_stats(london, path, a)
        if fastest_time is None:
            fastest_time = t
        pct = (t - fastest_time) / fastest_time * 100 if fastest_time > 0 else 0

        print(f"\n  {label}")
        print(f"    Path:         {names(london, path)}")
        print(f"    Node IDs:     {path}")
        print(f"    Total time:   {t:.4f}  (+{pct:.1f}% vs fastest)")
        print(f"    Total danger: {d:.4f}")
        print(f"    Composite:    {c:.4f}")
        print(f"    Edges:        {len(path) - 1}")

    # ---- Trade-off analysis: fastest vs safer (α=0.5) ----
    print(f"\n{'─' * 70}")
    print("TRADE-OFF ANALYSIS")
    print(f"{'─' * 70}")
    p_fast = find_path(london, start, goal, alpha=1.0)
    p_safe = find_path(london, start, goal, alpha=0.5)
    tf, df, _ = path_stats(london, p_fast, 1.0)
    ts, ds, _ = path_stats(london, p_safe, 0.5)

    if tf > 0 and df > 0:
        time_cost = (ts - tf) / tf * 100
        danger_saved = (df - ds) / df * 100
        print(f"\n  Fastest route:       {names(london, p_fast)}")
        print(f"    Time: {tf:.4f}   Danger: {df:.4f}")
        print(f"\n  Safer route (α=0.5): {names(london, p_safe)}")
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
        path = find_path(london, s, g, alpha=0.7)
        t, d, _ = path_stats(london, path, 0.7)
        print(f"\n  {desc}")
        print(f"    Path:   {names(london, path)}")
        print(f"    IDs:    {path}")
        print(f"    Time:   {t:.4f}  Danger: {d:.4f}")

    print(f"\n{'=' * 70}\n")