import heapq
from collections import deque
import copy
import random
from time import perf_counter as timer

goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

def is_goal(state): return state == goal_state

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None

def get_neighbors(state):
    i, j = find_zero(state)
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        if 0 <= (ni:=i+di) < 3 and 0 <= (nj:=j+dj) < 3:
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            neighbors.append(new_state)
    return neighbors

# Heurísticas
def manhattan(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_i, goal_j = (val-1)//3, (val-1)%3
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def misplaced(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                count += 1
    return count

def euclidean(state):
    distance = 0.0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_i, goal_j = (val-1)//3, (val-1)%3
                distance += ((i - goal_i)**2 + (j - goal_j)**2)**0.5
    return distance

# Algoritmos de Busca
def a_star_search(start, heuristic="manhattan"):
    start_time = timer()
    open_set = [(0, start)]
    came_from = {}
    g_score = {str(start): 0}
    nodes_expanded = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        
        if is_goal(current):
            path = reconstruct_path(came_from, current)
            return {
                'path': path,
                'nodes': nodes_expanded,
                'time': timer() - start_time,
                'depth': len(path)-1,
                'method': f"A* ({heuristic})"
            }
            
        for neighbor in get_neighbors(current):
            neighbor_str = str(neighbor)
            tentative_g = g_score[str(current)] + 1
            
            if neighbor_str not in g_score or tentative_g < g_score[neighbor_str]:
                g_score[neighbor_str] = tentative_g
                h = {
                    "manhattan": manhattan,
                    "misplaced": misplaced,
                    "euclidean": euclidean
                }[heuristic](neighbor)
                heapq.heappush(open_set, (tentative_g + h, neighbor))
                came_from[neighbor_str] = current
    
    return {
        'path': None,
        'nodes': nodes_expanded,
        'time': timer() - start_time,
        'depth': 0,
        'method': f"A* ({heuristic})"
    }

def bfs_search(start):
    start_time = timer()
    queue = deque([(start, [start])])
    visited = set()
    visited.add(str(start))
    nodes_expanded = 0
    
    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1
        
        if is_goal(current):
            return {
                'path': path,
                'nodes': nodes_expanded,
                'time': timer() - start_time,
                'depth': len(path)-1,
                'method': "BFS"
            }
        
        for neighbor in get_neighbors(current):
            if str(neighbor) not in visited:
                visited.add(str(neighbor))
                queue.append((neighbor, path + [neighbor]))
    
    return {
        'path': None,
        'nodes': nodes_expanded,
        'time': timer() - start_time,
        'depth': 0,
        'method': "BFS"
    }

def dfs_search(start, max_depth=30):
    start_time = timer()
    stack = [(start, [start], 0)]
    visited = set()
    visited.add(str(start))
    nodes_expanded = 0
    
    while stack:
        current, path, depth = stack.pop()
        nodes_expanded += 1
        
        if is_goal(current):
            return {
                'path': path,
                'nodes': nodes_expanded,
                'time': timer() - start_time,
                'depth': depth,
                'method': f"DFS (d={max_depth})"
            }
        
        if depth >= max_depth:
            continue
            
        for neighbor in reversed(get_neighbors(current)):
            if str(neighbor) not in visited:
                visited.add(str(neighbor))
                stack.append((neighbor, path + [neighbor], depth+1))
    
    return {
        'path': None,
        'nodes': nodes_expanded,
        'time': timer() - start_time,
        'depth': 0,
        'method': f"DFS (d={max_depth})"
    }

def reconstruct_path(came_from, current):
    path = [current]
    while str(current) in came_from:
        current = came_from[str(current)]
        path.insert(0, current)
    return path

def generate_solvable_state(moves=20):
    state = copy.deepcopy(goal_state)
    for _ in range(moves):
        state = random.choice(get_neighbors(state))
    return state

def is_solvable(state):
    flat = [num for row in state for num in row if num != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    return inversions % 2 == 0