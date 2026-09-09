import heapq

def a_star(graph, heuristic, start, goal):
    # Priority queue: (f_score, node)
    open_set = [(0, start)]
    
    # Track where each node came from to reconstruct the path
    came_from = {}
    
    # g_score: cost from start to node
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    
    # f_score = g_score + heuristic
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic[start]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        # If we reached the goal, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to get start -> goal
        
        # Explore neighbors
        for neighbor, weight in graph[current].items():
            tentative_g = g_score[current] + weight
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic[neighbor]
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

# ---- Graph from the assignment ----
graph = {
    'A': {'B': 2, 'E': 3},
    'B': {'C': 1, 'G': 9},
    'C': {},
    'E': {'D': 6},
    'D': {'G': 1},
    'G': {}
}

# Heuristic values (h(n)) as given in the PDF
heuristic = {
    'A': 11,
    'B': 6,
    'C': 99,   # effectively infinity
    'E': 6,
    'D': 1,
    'G': 0
}

start = 'A'
goal = 'G'

path = a_star(graph, heuristic, start, goal)
print(f"Shortest path from {start} to {goal}: {path}")
print(f"Expected optimal path: A -> E -> D -> G (cost = 10)")
