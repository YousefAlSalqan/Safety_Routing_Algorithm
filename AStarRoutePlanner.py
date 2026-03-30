"""
A* Route Planner — Complete Implementation for Study
=====================================================
 
Source: OanaGaskey/Route-Planner (Udacity Intro to Self-Driving Cars Nanodegree)
  https://github.com/OanaGaskey/Route-Planner
 
Academic foundation:
  - Hart, P.E., Nilsson, N.J., & Raphael, B. (1968). "A Formal Basis for the
    Heuristic Determination of Minimum Cost Paths." IEEE Transactions on Systems
    Science and Cybernetics, 4(2), 100-107.
  - The A* algorithm guarantees optimal shortest paths when the heuristic h(n) is
    *admissible* (never overestimates true cost) and *consistent* (satisfies the
    triangle inequality: h(n) <= c(n,n') + h(n') for every successor n').
 
This file contains:
  1. The Map helper class (graph representation)
  2. The AStarPathNode class (node bookkeeping)
  3. The AStarRouter class (the full A* algorithm)
  4. A standalone shortest_path() convenience function
  5. A demo with a small 10-node map
"""

import math 
import pickle 
from typing import List, Dict, Tuple, Optional

# =============================================================================
# PART 1 — HELPERS: Map Class
# =============================================================================
# The Map class wraps a graph where:
#   - intersections: dict mapping node_id -> (x, y) coordinates
#   - roads: list of lists; roads[i] = list of nodes that node i connects to
#
# This is equivalent to an adjacency-list representation of an undirected,
# weighted graph where edge weights are Euclidean distances between coordinates.
# =============================================================================

class Map: 
    """A simple graph/map representation.
    
    Attributes:
        intersections (dict): {node_id: (x, y)} — coordinates of each node.
        roads (list[list[int]]): roads[i] = list of neighbor node IDs for node i.
    """

    def __init__(self,intersections:Dict[int,Tuple[float,float]], roads: List[List[int]]):
        self.intersections = intersections 
        self.roads = roads 


def load_map_10() -> Map:
    """Returns a small 10-node map for testing.
    
    Graph structure (undirected edges):
      0 -- 7, 6, 5
      1 -- 4, 3, 2
      2 -- 4, 3, 1
      3 -- 5, 4, 1, 2
      4 -- 1, 2, 3
      5 -- 7, 0, 3
      6 -- 0
      7 -- 0, 5
      8 -- 9          (disconnected island)
      9 -- 8          (disconnected island)
    """
    intersections = {
        0: (0.7798606835438107, 0.6922727646627362),
        1: (0.7647837074641568, 0.3252670836724646),
        2: (0.7155217893995438, 0.20026498027300055),
        3: (0.7076566826610747, 0.3278339270610988),
        4: (0.8325506249953353, 0.02310946309985762),
        5: (0.49016747075266875, 0.5464878695400415),
        6: (0.8820353070895344, 0.6791919587749445),
        7: (0.46247219371675075, 0.6258061621642713),
        8: (0.11622158839385677, 0.11236327488812581),
        9: (0.1285377678230034, 0.3285840695698353),
    }
    roads = [
        [7, 6, 5],       # node 0
        [4, 3, 2],       # node 1
        [4, 3, 1],       # node 2
        [5, 4, 1, 2],    # node 3
        [1, 2, 3],       # node 4
        [7, 0, 3],       # node 5
        [0],             # node 6
        [0, 5],          # node 7
        [9],             # node 8
        [8],             # node 9
    ]
    return Map(intersections, roads)



# =============================================================================
# PART 2 — A* NODE: Bookkeeping for Each Visited Node
# =============================================================================


class AStarPathNode: 
    """Stores cost information and parent pointer for a single visited node.
    
    In A* terminology (Hart, Nilsson & Raphael, 1968):
      - g(n) = total_costs           — actual cost from start to this node
      - f(n) = assumed_costs_to_dest — g(n) + h(n), where h(n) is the heuristic
      - previous_node                — parent pointer for path reconstruction
    """
    def __init__(self, total_costs:float, assumed_costs_to_dest:float, previous_node:int): 
        self.total_costs = total_costs 
        self.assumed_costs_to_dest = assumed_costs_to_dest
        self.previous_node = previous_node


# =============================================================================
# PART 3 — A* ROUTER: The Core Algorithm
# =============================================================================
#
# How A* works (high level):
#
#   1. Start with the source node on the OPEN set (frontier).
#   2. Pick the node with the lowest f(n) = g(n) + h(n) from the frontier.
#   3. If that node is the goal → done, backtrace the path.
#   4. Otherwise, EXPAND it: for each neighbor, compute the tentative g-cost.
#      - If the neighbor hasn't been seen, or this path is cheaper, update it
#        and add it to the frontier.
#   5. Move the expanded node to the CLOSED set (explored).
#   6. Repeat from step 2.
#
# The heuristic h(n) used here is EUCLIDEAN DISTANCE (straight-line / beeline).
# This is admissible because the shortest possible path between two points in
# 2D space is the straight line — no real road path can be shorter.
#
# Admissibility proof sketch:
#   For any node n, h(n) = euclidean_dist(n, goal).
#   The true shortest path cost g*(n→goal) >= euclidean_dist(n, goal)
#   because edges follow Euclidean distances and the triangle inequality holds.
#   Therefore h(n) <= g*(n→goal), satisfying admissibility.
#   (See Hart, Nilsson & Raphael, 1968, Theorem 1)
# =============================================================================

class AStarRouter: 
    """A* search implementation for finding shortest paths on a 2D map.
    
    Attributes:
        map_data (Map): The map containing intersections and roads.
        tree (dict): {node_id: AStarPathNode} — all visited nodes with costs.
        goal (int): The target node index.
        frontier (set): The OPEN set — nodes discovered but not yet expanded.
    """

    def __init__(self, map_data: Map): 
        """Initialize the router with a map.
        
        Args:
            map_data: A Map object with .intersections and .roads attributes.
        """
        self.map_data = map_data 
        self.tree:Dict[int,AStarPathNode] = {} 
        self.goal:int = -1 
        self.frontier: set = set()
    
    def beeline_dist(self,start: int, dest:int) -> float: 
        """Compute Euclidean (straight-line) distance between two nodes.
        
        This serves as both:
          - The HEURISTIC h(n): estimated remaining cost to goal.
          - The EDGE COST c(n, n'): actual cost to travel between neighbors.
        
        In this simplified map, edges are straight lines, so edge cost = 
        Euclidean distance. In a real road network, edge costs would come 
        from actual road segment lengths or travel times.
        
        Args:
            start: Source node index.
            dest: Destination node index.
        Returns:
            Euclidean distance between the two nodes.
        """

        a = self.map_data.intersections[start] 
        b = self.map_data.intersections[dest]

        return math.sqrt((b[0] - a[0]) **2 + (b[1] - a[1]) **2) 
    
    def road_costs(self, start: int, dest: int) -> float:
        """Return the cost of traveling along the edge from start to dest.
        
        In this implementation, road cost = beeline distance (since edges 
        are straight lines on the map). In your TransportAI project, this 
        is where you'd plug in the composite cost function:
        
            f(n) = α · time(n) + (1-α) · safety(n)
        
        (See Keeney & Raiffa, 1976, for MAUT formalization)
        """
        return self.beeline_dist(start, dest)
    
    # ---- Core A* methods ----

    def expanded_intersection(self, start: int, costs: float): 
        """Expand a node: examine all its neighbors and update costs.
        
        This is the key step of A*. For the current node (start), we:
        1. Remove it from the frontier (it's now "explored" / in CLOSED set).
        2. For each neighbor:
           a. Compute tentative g-cost: g(start) + c(start, neighbor)
           b. Compute f-cost: tentative_g + h(neighbor, goal)
           c. If neighbor not yet visited, OR this path is cheaper → update it
              and add to frontier.
        
        This implements the "relaxation" step from Dijkstra (1959), enhanced
        with the heuristic guidance of A* (Hart, Nilsson & Raphael, 1968).
        
        Args:
            start: The node being expanded.
            costs: The g-cost (actual cost from source) to reach this node.
        """
        self.frontier.remove(start)

        for dest in self.map_data.roads[start]: 
            road_distance = costs + self.road_costs(start, dest)
            total_assumed_distance = road_distance + self.beeline_dist(dest, self.goal)
            
            if(dest not in self.tree or
               self.tree[dest].assumed_costs_to_dest > total_assumed_distance): 
                self.tree[dest] = AStarPathNode(road_distance, 
                                                total_assumed_distance, 
                                                start)
                self.frontier.add(dest) 
    

    def cheapest_front_node(self) -> int: 
        """Select the frontier node with the lowest f-cost.
        
        This is the "priority queue" step. In production code, you'd use a
        min-heap (e.g., Python's heapq) for O(log n) extraction. This 
        implementation uses a linear scan over the frontier set — simpler 
        but O(n) per extraction.
        
        Returns:
            Index of the cheapest frontier node, or -1 if frontier is empty.
        """
        if len(self.frontier) == 0: 
            return -1 
        
        cheapest = next(iter(self.frontier))
        cheapest_costs = self.tree[cheapest].assumed_costs_to_dest

        for front_node in self.frontier: 
            node = self.tree[front_node]
            if node.assumed_costs_to_dest < cheapest_costs: 
                cheapest_costs = node.assumed_costs_to_dest 
                cheapest = front_node

        return cheapest 
    

    def shortest_path(self, start: int, goal: int) -> List[int]:
        """Find the shortest path from start to goal using A*.
        
        Algorithm (pseudocode from Russell & Norvig, AIMA, 4th ed., Ch. 3):
        
            function A*-SEARCH(problem) returns solution or failure
                node ← NODE(problem.INITIAL, g=0, f=h(INITIAL))
                frontier ← priority queue ordered by f, with node
                reached ← {problem.INITIAL: node}
                while frontier is not empty do
                    node ← POP(frontier)      // lowest f-value
                    if problem.IS-GOAL(node.STATE) then return SOLUTION(node)
                    for each child in EXPAND(problem, node) do
                        if child.STATE not in reached or 
                           child.PATH-COST < reached[child.STATE].PATH-COST then
                            reached[child.STATE] ← child
                            add child to frontier
                return failure
        
        Args:
            start: The source node index.
            goal: The destination node index.
        Returns:
            List of node indices representing the shortest path [start, ..., goal].
            Empty list if no path exists.
        """
        # Edge case: already at the goal
        if start == goal:
            return [goal]
        
        # Initialize: clear previous state, set goal
        self.tree = {}
        self.goal = goal
        
        # Create start node: g=0, f=h(start, goal), no parent (-1)
        self.frontier = {start}
        self.tree[start] = AStarPathNode(
            0,                                    # g(start) = 0
            self.beeline_dist(start, goal),       # f(start) = 0 + h(start)
            -1                                     # no parent
        )
        
        # Expand the start node
        self.expand_intersection(start, 0)
        
        target_reached = False
        cheapest_last = -1  # Track previous cheapest to detect stuck state
        
        # ---- MAIN LOOP ----
        while True:
            # Step 1: Pick the frontier node with lowest f-cost
            cheapest_next = self.cheapest_front_node()
            
            # Step 2: Goal test — if cheapest frontier node IS the goal, done!
            # (A* guarantees this is optimal when h is admissible)
            if cheapest_next == goal:
                target_reached = True
                break
            
            # Step 3: Expand the cheapest node (if valid)
            if cheapest_next != -1:
                node = self.tree[cheapest_next]
                self.expand_intersection(cheapest_next, node.total_costs)
            
            # Step 4: Detect if we're stuck (no progress = disconnected graph)
            if cheapest_last == cheapest_next:
                return []  # No path exists
            
            cheapest_last = cheapest_next
        
        # ---- PATH RECONSTRUCTION ----
        # Backtrace from goal to start using parent pointers
        result = []
        if target_reached:
            cur_index = goal
            while cur_index != -1:
                result.append(cur_index)
                cur_index = self.tree[cur_index].previous_node
            result.reverse()  # Reverse: we built it goal→start, need start→goal
        
        return result
 
 
# =============================================================================
# PART 4 — CONVENIENCE FUNCTION
# =============================================================================
 
def shortest_path(M: Map, start: int, goal: int) -> List[int]:
    """Find the shortest path using A* search.
    
    This is the top-level function that creates an AStarRouter and runs the search.
    
    Args:
        M: A Map object.
        start: Source node index.
        goal: Destination node index.
    Returns:
        List of node indices from start to goal.
    """
    router = AStarRouter(M)
    return router.shortest_path(start, goal)
 
 
# =============================================================================
# PART 5 — DEMO / SELF-TEST
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 60)
    print("A* Route Planner — Demo")
    print("=" * 60)
    
    # Load the 10-node test map
    m = load_map_10()
    
    print("\nMap has {} intersections and {} road entries.".format(
        len(m.intersections), len(m.roads)))
    
    # Test 1: Path within the connected component (nodes 0-7)
    path = shortest_path(m, 0, 4)
    print("\nShortest path from 0 to 4: {}".format(path))
    
    # Compute total distance
    total_dist = 0
    for i in range(len(path) - 1):
        a = m.intersections[path[i]]
        b = m.intersections[path[i + 1]]
        d = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        total_dist += d
        print("  {} -> {}: distance = {:.4f}".format(path[i], path[i + 1], d))
    print("  Total distance: {:.4f}".format(total_dist))
    
    # Test 2: Path between disconnected nodes
    path2 = shortest_path(m, 0, 8)
    print("\nShortest path from 0 to 8 (disconnected): {}".format(path2))
    
    # Test 3: Start == Goal
    path3 = shortest_path(m, 5, 5)
    print("Shortest path from 5 to 5 (trivial): {}".format(path3))
    
    # Test 4: Path within the island
    path4 = shortest_path(m, 8, 9)
    print("Shortest path from 8 to 9 (island): {}".format(path4))
    
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
    


        




    









