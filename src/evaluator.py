import json
import numpy as np
from typing import Dict, List, Any, Callable

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_predictions(predictions: Dict[str, List[Dict[str, List[List[int]]]]], 
                         solutions: Dict[str, List[List[List[int]]]]) -> Dict[str, float]:
    """
    Evaluates exact-match predictions against ground truth solutions.
    Follows Kaggle rules: 2 attempts per test output, exact matrix match required.
    """
    total_outputs = 0
    correct_outputs = 0

    for task_id, ground_truths in solutions.items():
        if task_id not in predictions:
            total_outputs += len(ground_truths)
            continue
            
        task_preds = predictions[task_id]
        
        for idx, target_grid in enumerate(ground_truths):
            total_outputs += 1
            if idx >= len(task_preds):
                continue
                
            pred_pair = task_preds[idx]
            attempt_1 = pred_pair.get("attempt_1", [])
            attempt_2 = pred_pair.get("attempt_2", [])
            
            # Convert to numpy arrays for strict comparison
            target_arr = np.array(target_grid)
            
            match_1 = np.array_equal(np.array(attempt_1), target_arr) if len(attempt_1) > 0 else False
            match_2 = np.array_equal(np.array(attempt_2), target_arr) if len(attempt_2) > 0 else False
            
            if match_1 or match_2:
                correct_outputs += 1

    accuracy = (correct_outputs / total_outputs) if total_outputs > 0 else 0.0
    return {
        "total_test_outputs": total_outputs,
        "correct_predictions": correct_outputs,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    print("Evaluator module initialized successfully.")