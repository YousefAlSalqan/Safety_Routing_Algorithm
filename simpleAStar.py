import heapq 

def a_star(graph, start, goal, h): 
    #Priority queue: (f_cost, node, path, g_cost)
    open_list = [(h[start], start,[start],0)]
    # Explored set 
    closed_set = set()

    while open_list: 
        f, current, path, g = heapq.heappop(open_list)

        if current == goal: 
            return path,g 
        
        if current in closed_set: 
            continue 
        closed_set.add(current)

        for neighbor, weight in graph[current]: 
            if neighbor not in closed_set: 
                new_g = g+weight 
                new_f = new_g +h[neighbor]
                heapq.heappush(open_list, (new_f, neighbor, path +[neighbor], new_g))
    
    return None, float('inf')

