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

def biderictional_dijkstra(graph, start, goal): 
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



