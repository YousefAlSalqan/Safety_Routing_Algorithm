"""
osrm_style.py — OSRM's Routing Approach in Python
 
OSRM does NOT use A*. It uses bidirectional Dijkstra on a preprocessed graph.
    "In contrast to most routing servers, OSRM does not use an A* variant
     to compute the shortest path, but instead uses contraction hierarchies
     or multilevel Dijkstra's." — OpenStreetMap Wiki
 
This file implements OSRM's two algorithms:
    1. Bidirectional Dijkstra (the core query engine)
    2. Contraction Hierarchies preprocessing + upward bidirectional search
 
Based on OSRM source:
    - include/engine/routing_algorithms/routing_base.hpp
    - src/engine/routing_algorithms/shortest_path.cpp
    - Telenav/open-source-spec analysis of OSRM internals
 
References:
    - Geisberger et al. (2008), Contraction Hierarchies
    - Luxen & Vetter (2011), "Real-time routing with OpenStreetMap data"
      (the OSRM paper cited in their README)
"""

import heapq 

def bidirectional_dijkstra(graph, start, goal): 
    """"
    Bidirectional Dijkstra's algorithm for finding the shortest path between two nodes in a graph. 

    Two searches run simlumtaneously: 
    1. Forward search: from start to goal, expanding outward 
    2. Bacward search: from goal to start, expanding outward

    They meet in the middle. The meeting point with the lowest total cost (forward + backward) is the answer.
    """

    #Forward search (from start) 
    forward_queue = [(0, start)] 
    forward_cost = {start:0}
    forward_parent = {start:None}
    forward_settles = set() 

    #Backward search (from goal) 
    backward_queue = [(0, goal)]
    backward_cost = {goal:0}
    backward_parent = {goal:None}
    backward_settled = set() 

    best_cost = float('inf') 
    meeting_node = None 

    nodes_expanded = 0 

    while forward_queue and backward_queue: 
        # Termination condition (from OSRM source):
        #"forward_heap_min + reverse_heap_min < weight"
        # If the minimum unsettled cost exceeds the best known path, we're done. 

        forward_min = forward_queue[0][0] if forward_queue else float('inf')
        backward_min = backward_queue[0][0] if backward_queue else float('inf') 

        if forward_min + backward_min >= best_cost: 
            break

        if forward_queue and forward_min <= backward_min: 
            cost, node = heapq.heappop(forward_queue) 

            if cost > forward_cost.get(node,float('inf')):
                continue

            forward_settles.add(node) 
            nodes_expanded +=1 

            if node in backward_cost: 
                total = cost + backward_cost[node]
                if total < best_cost: 
                    best_cost = total 
                    meeting_node = node 

            for neighbor, weight in graph.get[node].items(): 
                new_cost = cost + weight 
                if new_cost < forward_cost.get(neighbor, float('inf')): 
                    forward_cost[neighbor] = new_cost 
                    forward_parent[neighbor] = node 
                    heapq.heappush(forward_queue, (new_cost, neighbor))

        elif backward_queue: 
            cost, node = heapq.heappop(backward_queue)

            if cost > backward_cost.get(node, float('inf')):
                continue

            backward_settled.add(node)
            nodes_expanded += 1

            if node in forward_cost:
                total = cost + forward_cost[node]
                if total < best_cost:
                    best_cost = total
                    meeting_node = node

            for neighbor, weight in graph[node].items():
                new_cost = cost + weight
                if new_cost < backward_cost.get(neighbor, float('inf')):
                    backward_cost[neighbor] = new_cost
                    backward_parent[neighbor] = node
                    heapq.heappush(backward_queue, (new_cost, neighbor))
    
    if meeting_node is None: 
        return None, float('inf'), 0 
    
    # Reconstruct path: start -> meeting <- goal 

    #forward path 
    forward_path = [] 
    node = meeting_node 
    while node is not None: 
        forward_path.append(node) 
        node = forward_parent[node]
    forward_path.reverse()

    #backward path
    backward_path = []
    node = backward_parent.get(meeting_node) 
    while node is not None:
        backward_path.append(node)
        node = backward_parent.get(node)

    full_path = forward_path + backward_path 
    return full_path, best_cost, nodes_expanded


#Contraction Hierarchies. This involves a preprocessing step to create a hierarchy of nodes, then an upward search on the contracted graph.
#This is the part where we can the safety heursitic 

def ch_preprocess(graph):
    """
     CH Preprocessing — contract nodes and add shortcuts.
 
    Returns:
        ch_graph: the graph with shortcuts added
        importance: rank of each node (higher = more important)
    """
    ch_graph = {n:dict(neighbors) for n, neighbors in graphs.items()} 

    nodes = list(ch_graph.keys()) 

    importance = {}

    for node in nodes: 
        degree = len(ch_graph[node])
        importance[node] = degree 

    order = sorted(nodes, jey = lambda n:importance[n])

    rank = {} 

    for i, node in enumerate(order): 
        rank[node] = i 

    contracted = set() 
    shortcuts_added = 0 

    #I will need to have to a look at this to see how I might change this to add safety to the equation 
    for node in order[:-2]: 
        in_neighbors = [] 
        out_neighbors = [] 

        for other, weight in ch_graph[node].items(): 
            if other not in contracted: 
                out_neighbors.append((other, weight))
        
        for other in ch_graph: 
            if other not in contracted and node in ch_graph[other]: 
                in_neighbors.append((other, ch_graph[other][node]))

        for u,w_in in in_neighbors: 
            for v,w_out in out_neighbors: 
                if u == v: 
                    continue 

                shortcut_cost = w_in +w_out 

                existing = ch_graph.get(u,{}).get(v,float('inf')) 
                if shortcut_cost < existing:
                    #Add shortcut  
                    if u not in ch_graph: 
                        ch_graph[u] = {} 
                    ch_graph[u][v] = shortcut_cost 
                    if v not in ch_graph:
                        ch_graph[v] = {} 
                    ch_graph[v][u] = shortcut_cost 
                    shortcuts_added += 1

        contracted.add(node)

    print(f"  [CH] Contracted {len(contracted)} nodes, added {shortcuts_added} shortcuts")
    return ch_graph, rank

def ch_query(ch_graph, rank, start, goal): 
    """
    CH Query - bidirectional UPWARD search 

    Forward search goes UP from start.
    Backward search goes UP from goal.
    They meet at the most important node on the shortest path.
    """

    forward_queue = [(0,start)]
    forward_cost = {start:0}
    forward_parent = {start: None}

    backward_queue = [(0,goal)] 
    backward_cost = {goal:0}
    backward_parent = {goal: None}

    best_cost = float('inf') 
    meeting_node = None 
    nodes_expanded = 0 

    while forward_queue: 
        cost, node = heapq.heappop(forward_queue) 
        if cost > forward_cost.get(node, float('inf')): 
            continue

        nodes_expanded+=1 

        if node in backward_cost:
            total = cost + backward_cost[node]
            if total < best_cost:
                best_cost = total
                meeting_node = node

        
        for neighbor, weight in ch_graph.get(node,{}).items(): 
            if rank.get(neighbor,0) <= rank.get(node,0):
                continue 

            new_cost = cost + weight 

            if new_cost < forward_cost.get(neighbor, float('inf')): 
                forward_cost[neighbor] = new_cost
                forward_parent[neighbor] = node 
                heapq.heappush(forward_queue, (new_cost, neighbor))

    
    while backward_queue: 
        cost,node = heapq.heappop(backward_queue) 
        if cost > backward_cost.get(node, float('inf')): 
            continue 
        nodes_expanded += 1 

        if node in forward_cost: 
            total = cost + forward_cost[node]
            if total < best_cost: 
                best_cost = total 
                meeting_node = node 

        for neighbor, weight in ch_graph.get(node,{}).items(): 
            if rank.get(neighbor,0) <= rank.get(node,0): 
                continue
            new_cost = cost + weight
            if new_cost < backward_cost.get(neighbor, float('inf')): 
                backward_cost[neighbor] = new_cost 
                backward_parent[neighbor] = node 
                heapq.heappush(backward_queue, (new_cost, neighbor))

    if meeting_node is None:
        return None, float('inf'), nodes_expanded
    
    forward_path = [] 
    node = meeting_node 
    while node is not None: 
        forward_path.append(node) 
        node = forward_parent(node)
    forward_path.reverse()

    backward_path = []
    node = backward_parent.get(meeting_node)
    while node is not None: 
        backward_path.append(node) 
        node = backward_parent.get(node)

    full_path = forward_path + backward_path
    return full_path, best_cost, nodes_expanded


# ═══════════════════════════════════════════════════════════════
# FOR COMPARISON: STANDARD DIJKSTRA
# ═══════════════════════════════════════════════════════════════
 
def dijkstra(graph, start, goal):
    """Standard (unidirectional) Dijkstra for comparison."""
    queue = [(0, start)]
    costs = {start: 0}
    parents = {start: None}
    nodes_expanded = 0
 
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > costs.get(node, float('inf')):
            continue
        nodes_expanded += 1
 
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1], cost, nodes_expanded
 
        for neighbor, weight in graph[node].items():
            new_cost = cost + weight
            if new_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = new_cost
                parents[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))
 
    return None, float('inf'), nodes_expanded
 
 
# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    # CS4006 Lecture 4 graph
    graph = {
        'A': {'B': 1, 'C': 7},
        'B': {'A': 1, 'D': 9, 'E': 1},
        'C': {'A': 7, 'E': 5},
        'D': {'B': 9, 'E': 5, 'F': 2},
        'E': {'B': 1, 'C': 5, 'D': 5, 'G': 3},
        'F': {'D': 2, 'H': 5},
        'G': {'E': 3, 'H': 5},
        'H': {'F': 5, 'G': 5},
    }
 
    print("=" * 55)
    print("COMPARING OSRM'S ALGORITHMS (A → H)")
    print("=" * 55)
 
    # 1. Standard Dijkstra
    path, cost, expanded = dijkstra(graph, 'A', 'H')
    print(f"\n1. Standard Dijkstra:")
    print(f"   Path: {' → '.join(path)}")
    print(f"   Cost: {cost}, Nodes expanded: {expanded}")
 
    # 2. Bidirectional Dijkstra (OSRM's core)
    path, cost, expanded = bidirectional_dijkstra(graph, 'A', 'H')
    print(f"\n2. Bidirectional Dijkstra (OSRM style):")
    print(f"   Path: {' → '.join(path)}")
    print(f"   Cost: {cost}, Nodes expanded: {expanded}")
 
    # 3. Contraction Hierarchies
    print(f"\n3. Contraction Hierarchies (OSRM style):")
    ch_graph, rank = ch_preprocess(graph)
    path, cost, expanded = ch_query(ch_graph, rank, 'A', 'H')
    if path:
        print(f"   Path: {' → '.join(path)}")
        print(f"   Cost: {cost}, Nodes expanded: {expanded}")












