from src.solver import solve_sudoku_with_model
# from tensorflow.keras.models import load_model
# loaded_model = keras.saving.load_model("model.keras")
from tensorflow import keras
import numpy as np

if __name__ == "__main__":
    model = keras.saving.load_model("models/sudoku_solver.keras")

    # Example puzzle
    sample_puzzle = np.array([
        [0, 4, 0, 1, 0, 0, 0, 5, 0],
        [7, 0, 0, 0, 3, 0, 0, 9, 6],
        [5, 2, 0, 0, 0, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 7],
        [0, 0, 0, 9, 0, 6, 8, 0, 0],
        [8, 0, 3, 0, 5, 0, 6, 2, 0],
        [0, 9, 0, 0, 6, 0, 5, 4, 3],
        [6, 0, 0, 0, 8, 0, 7, 0, 0],
        [2, 5, 0, 0, 9, 7, 1, 8, 0],
    ])

    solved_puzzle = solve_sudoku_with_model(model, sample_puzzle)

    print("Original puzzle:")
    print(sample_puzzle)
    print("Solved puzzle:")
    print(solved_puzzle)
