def is_valid(map_array, x, y, visited):
    if x < 0 or x >=len(map_array[0]) or y < 0 or y >= len(map_array):
        return False
    if visited[y][x]:
        return False
    if map_array[y][x] == 0:
        return False
    return True

def playablebfs(map_array):
    height = len(map_array)
    width = len(map_array[0])
    spawn_point = None
    staircase = None
    for y in range(height):
        for x in range(width):
            if map_array[y, x] == 2:
                spawn_point = (x, y)
            elif map_array[y, x] == 3:
                staircase = (x, y)
            else:
                continue
    if spawn_point is None or staircase is None:
        return None
    
    visited = [[False] * width for _ in range(height)]
    visited[spawn_point[1]][spawn_point[0]] = True
    queue = [(spawn_point, [spawn_point])]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while queue:
        (current, path) = queue.pop(0)
        if current == staircase:
            return path
        for dx, dy in directions:
            if is_valid(map_array, current[0] + dx, current[1] + dy, visited):
                visited[current[1] + dy][current[0] + dx] = True
                queue.append(((current[0] + dx, current[1] + dy), path + [(current[0] + dx, current[1] + dy)]))
    return None
    
    


        


