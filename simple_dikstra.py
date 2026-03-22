"""
This implementation is based on the paper: "A note on two problems in connexion with graphs" by E. W. Dijkstra, published in 1959."

"""
import heapq 

def dijkstra(graph, start,goal): 
    queue = [(0,start)]

    costs = {start: 0} 

    parents = {start: None}

    while queue:
        cost, node = heapq.heappop(queue) 

        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1], cost

        if cost > costs.get(node, float('inf')):
            continue

        for neighbor, weight in graph[node].items():
            new_cost = cost + weight
            if new_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = new_cost
                parents[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))
        
    return None, float('inf')  # Return None if no path is found

