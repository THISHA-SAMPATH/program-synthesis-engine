# --- Advanced Color Primitives ---

def color_recolor_most_frequent(grid: np.ndarray, target_color: int = 1) -> np.ndarray:
    """Replaces the most frequent non-background color with target_color."""
    res = grid.copy()
    non_bg = res[res != 0]
    if non_bg.size == 0:
        return res
    colors, counts = np.unique(non_bg, return_counts=True)
    most_freq = colors[np.argmax(counts)]
    res[res == most_freq] = target_color
    return res

def color_recolor_least_frequent(grid: np.ndarray, target_color: int = 2) -> np.ndarray:
    """Replaces the least frequent non-background color with target_color."""
    res = grid.copy()
    non_bg = res[res != 0]
    if non_bg.size == 0:
        return res
    colors, counts = np.unique(non_bg, return_counts=True)
    least_freq = colors[np.argmin(counts)]
    res[res == least_freq] = target_color
    return res

def fill_background(grid: np.ndarray, target_color: int = 1) -> np.ndarray:
    """Replaces all 0s (background) with target_color."""
    res = grid.copy()
    res[res == 0] = target_color
    return res

# --- Grid Scaling & Tiling ---

def scale_2x(grid: np.ndarray) -> np.ndarray:
    """Scales up the grid 2x in both dimensions."""
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

def tile_2x2(grid: np.ndarray) -> np.ndarray:
    """Tiles the grid in a 2x2 matrix pattern."""
    return np.tile(grid, (2, 2))

def tile_3x3(grid: np.ndarray) -> np.ndarray:
    """Tiles the grid in a 3x3 matrix pattern."""
    return np.tile(grid, (3, 3))