grid_1 = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

grid_2 = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]


def dfs(row_index, col_index, rows, cols, grid):
    if row_index >= rows or row_index < 0 or col_index >= cols or col_index < 0 or grid[row_index][col_index] == "0":
        return False
    grid[row_index][col_index] = "0"
    dfs(row_index, col_index+1, rows, cols, grid)
    dfs(row_index+1, col_index, rows, cols, grid)
    dfs(row_index, col_index-1, rows, cols, grid)
    dfs(row_index-1, col_index, rows, cols, grid)
    return True

def countIsland(grid):
    rows,cols = len(grid),len(grid[0])
    count = 0
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == '1':
                dfs(x, y, rows, cols, grid)
                count += 1
    return count

if __name__ == "__main__":
    print(grid_1)
    print(countIsland(grid_1))
    print(grid_2)
    print(countIsland(grid_2))