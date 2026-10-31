import json
import os
from src.solver import solve_task
from src.evaluator import evaluate_predictions

DATA_DIR = "data"
EVAL_CHALLENGES = os.path.join(DATA_DIR, "arc-agi_evaluation_challenges.json")
EVAL_SOLUTIONS = os.path.join(DATA_DIR, "arc-agi_evaluation_solutions.json")

def run_pipeline():
    if not (os.path.exists(EVAL_CHALLENGES) and os.path.exists(EVAL_SOLUTIONS)):
        print(f"Data files not found in '{DATA_DIR}/'. Place evaluation JSON files in data/ to run benchmarks.")
        return

    with open(EVAL_CHALLENGES, 'r') as f:
        challenges = json.load(f)
    with open(EVAL_SOLUTIONS, 'r') as f:
        solutions = json.load(f)

    predictions = {}
    print(f"Processing {len(challenges)} evaluation tasks...")

    for task_id, task in challenges.items():
        predictions[task_id] = solve_task(task)

    # Export submission format
    with open("submission.json", "w") as f:
        json.dump(predictions, f)
    print("Generated 'submission.json' successfully.")

    # Run Evaluator
    metrics = evaluate_predictions(predictions, solutions)
    print("\n--- Benchmark Evaluation Results ---")
    print(f"Total Test Outputs:  {metrics['total_test_outputs']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print(f"Local Accuracy:      {metrics['accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    run_pipeline()