import heapq 

def a_star(graph, start, goal,h): 
    queue = [(h[start], start)]
    costs = {start: 0}
    parents = {start: None}

    while queue: 
        f , node = heapq.heappop(queue)

        if node == goal: 
            path = [] 
            while node is not None: 
                path.append(node)
                node = parents[node]
                return path[::-1], costs[goal]
            
        if f > costs.get(node, float('inf')) + h[node]: 
            continue 

        for neighbor, weight in graph[node].items(): 
            new_cost = costs[node] + weight 
            if new_cost < costs.get(neighbor, float('inf')): 
                costs[neighbor] = new_cost 
                parents[neighbor] = node 
                heapq.heappush(queue, (new_cost + h[neighbor], neighbor))

    return None, float('inf')

