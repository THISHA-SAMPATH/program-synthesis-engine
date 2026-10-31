import numpy as np

# --- 1. Geometry & Orientation ---

def rotate_90(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.rot90(arr, k=-1)

def rotate_180(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.rot90(arr, k=2)

def rotate_270(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.rot90(arr, k=1)

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.fliplr(arr)

def flip_vertical(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.flipud(arr)

def transpose(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return arr.T

# --- 2. Color Manipulations ---

def replace_color(grid: np.ndarray, src_color: int, target_color: int) -> np.ndarray:
    res = np.array(grid, copy=True)
    res[res == src_color] = target_color
    return res

def color_recolor_most_frequent(grid: np.ndarray, target_color: int = 1) -> np.ndarray:
    """Replaces the most frequent non-background color with target_color."""
    res = np.array(grid, copy=True)
    non_bg = res[res != 0]
    if non_bg.size == 0:
        return res
    colors, counts = np.unique(non_bg, return_counts=True)
    most_freq = colors[np.argmax(counts)]
    res[res == most_freq] = target_color
    return res

def color_recolor_least_frequent(grid: np.ndarray, target_color: int = 2) -> np.ndarray:
    """Replaces the least frequent non-background color with target_color."""
    res = np.array(grid, copy=True)
    non_bg = res[res != 0]
    if non_bg.size == 0:
        return res
    colors, counts = np.unique(non_bg, return_counts=True)
    least_freq = colors[np.argmin(counts)]
    res[res == least_freq] = target_color
    return res

def fill_background(grid: np.ndarray, target_color: int = 1) -> np.ndarray:
    """Replaces all 0s (background) with target_color."""
    res = np.array(grid, copy=True)
    res[res == 0] = target_color
    return res

# --- 3. Subgrids, Scaling & Tiling ---

def crop_to_non_background(grid: np.ndarray, bg_color: int = 0) -> np.ndarray:
    arr = np.asarray(grid)
    non_bg_coords = np.argwhere(arr != bg_color)
    if non_bg_coords.size == 0:
        return arr.copy()
    min_r, min_c = non_bg_coords.min(axis=0)
    max_r, max_c = non_bg_coords.max(axis=0)
    return arr[min_r:max_r+1, min_c:max_c+1]

def scale_2x(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)

def tile_2x2(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.tile(arr, (2, 2))

def tile_3x3(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    return np.tile(arr, (3, 3))

if __name__ == "__main__":
    test_grid = np.array([[0, 1], [0, 0]])
    print("DSL Module Loaded. Test rotate_90:\n", rotate_90(test_grid))