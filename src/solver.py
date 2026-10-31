import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import src.dsl as dsl

# Define a repository of available single-argument DSL primitives
PRIMITIVES_1ARG: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rotate_90": dsl.rotate_90,
    "rotate_180": dsl.rotate_180,
    "rotate_270": dsl.rotate_270,
    "flip_horizontal": dsl.flip_horizontal,
    "flip_vertical": dsl.flip_vertical,
    "transpose": dsl.transpose,
    "crop_non_bg": dsl.crop_to_non_background,
    "scale_2x": dsl.scale_2x,
    "tile_2x2": dsl.tile_2x2,
    "tile_3x3": dsl.tile_3x3,
    "color_recolor_most_frequent": dsl.color_recolor_most_frequent,
    "color_recolor_least_frequent": dsl.color_recolor_least_frequent,
    "fill_background": dsl.fill_background
}

def apply_program(grid: np.ndarray, program: List[str]) -> np.ndarray:
    """Applies a sequence of 1-argument DSL primitives sequentially."""
    current = grid.copy()
    for func_name in program:
        if func_name in PRIMITIVES_1ARG:
            current = PRIMITIVES_1ARG[func_name](current)
    return current

def search_dsl_program(train_pairs: List[Dict[str, List[List[int]]]], max_depth: int = 2) -> Optional[List[str]]:
    """
    Searches for a DSL primitive program sequence (up to max_depth operations)
    that transforms ALL train inputs into their exact target outputs.
    """
    train_inputs = [np.array(p["input"]) for p in train_pairs]
    train_outputs = [np.array(p["output"]) for p in train_pairs]

    # Level 1 Search (Single primitives)
    for name in PRIMITIVES_1ARG.keys():
        program = [name]
        matches_all = True
        for inp, target in zip(train_inputs, train_outputs):
            res = apply_program(inp, program)
            if not np.array_equal(res, target):
                matches_all = False
                break
        if matches_all:
            return program

    # Level 2 Search (Chained primitives)
    if max_depth >= 2:
        for name1 in PRIMITIVES_1ARG.keys():
            for name2 in PRIMITIVES_1ARG.keys():
                program = [name1, name2]
                matches_all = True
                for inp, target in zip(train_inputs, train_outputs):
                    res = apply_program(inp, program)
                    if not np.array_equal(res, target):
                        matches_all = False
                        break
                if matches_all:
                    return program

    return None

def search_color_swap(train_pairs: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Searches for a direct color substitution (src_color -> target_color)."""
    train_inputs = [np.array(p["input"]) for p in train_pairs]
    train_outputs = [np.array(p["output"]) for p in train_pairs]

    for c_in in range(10):
        for c_out in range(10):
            if c_in == c_out:
                continue
            matches_all = True
            for inp, target in zip(train_inputs, train_outputs):
                res = dsl.replace_color(inp, c_in, c_out)
                if not np.array_equal(res, target):
                    matches_all = False
                    break
            if matches_all:
                return (c_in, c_out)
    return None

def solve_task(task: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
    """
    Solves a single ARC task dictionary using the DSL Program Synthesis Engine.
    Returns attempt_1 and attempt_2 per test input following competition rules.
    """
    train_pairs = task["train"]
    test_inputs = task["test"]
    
    found_program = search_dsl_program(train_pairs, max_depth=2)
    color_swap = search_color_swap(train_pairs) if found_program is None else None
    
    predictions = []

    for test_pair in test_inputs:
        inp_grid = np.array(test_pair["input"])
        
        if found_program is not None:
            pred_1 = apply_program(inp_grid, found_program).tolist()
            pred_2 = inp_grid.tolist()
        elif color_swap is not None:
            c_in, c_out = color_swap
            pred_1 = dsl.replace_color(inp_grid, c_in, c_out).tolist()
            pred_2 = inp_grid.tolist()
        else:
            pred_1 = inp_grid.tolist()
            pred_2 = dsl.rotate_90(inp_grid).tolist()

        predictions.append({
            "attempt_1": pred_1,
            "attempt_2": pred_2
        })

    return predictions

if __name__ == "__main__":
    # Quick sanity test on dummy task data
    sample_task = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 0], [0, 1]]},
            {"input": [[2, 3], [0, 0]], "output": [[0, 0], [3, 2]]}
        ],
        "test": [
            {"input": [[5, 6], [7, 8]]}
        ]
    }
    solution = solve_task(sample_task)
    print("Program Synthesis Sanity Output:", solution)