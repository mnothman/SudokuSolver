import numpy as np
# from tensorflow.keras.models import load_model
# loaded_model = keras.saving.load_model("model.keras")
from tensorflow import keras
import tensorflow as tf

import numpy as np

def enforce_sudoku_constraints(grid):
    def find_missing_numbers(values):
        all_numbers = set(range(1, 10))
        existing_numbers = set(values)
        # Remove invalid numbers from existing_numbers
        existing_numbers = {num for num in existing_numbers if 1 <= num <= 9}
        return list(all_numbers - existing_numbers)

    # rows
    for i in range(9):
        row = grid[i, :]
        missing_numbers = find_missing_numbers(row)
        for j in range(9):
            if row[j] < 1 or row[j] > 9 or list(row).count(row[j]) > 1:
                if missing_numbers:
                    grid[i, j] = missing_numbers.pop()

    # columns
    for j in range(9):
        col = grid[:, j]
        missing_numbers = find_missing_numbers(col)
        for i in range(9):
            if col[i] < 1 or col[i] > 9 or list(col).count(col[i]) > 1:
                if missing_numbers:
                    grid[i, j] = missing_numbers.pop()

    # sub grids
    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            subgrid = grid[row:row+3, col:col+3].flatten()
            missing_numbers = find_missing_numbers(subgrid)
            for i in range(3):
                for j in range(3):
                    cell = grid[row+i, col+j]
                    if cell < 1 or cell > 9 or list(subgrid).count(cell) > 1:
                        if missing_numbers:
                            grid[row+i, col+j] = missing_numbers.pop()

    return grid

def solve_sudoku_with_model(model, puzzle):
    normalized_puzzle = puzzle / 9.0

    prediction = model.predict(normalized_puzzle[np.newaxis, ...])

    print("Raw predictions (first cell):", prediction[0, 0, :])

    # Convert predictions to integers (1–9) and then after reshape to 9x9
    solved_puzzle = np.argmax(prediction, axis=-1).reshape(9, 9) + 1

    return solved_puzzle

def is_valid_sudoku(grid):
    for i in range(9):
        if len(set(grid[i, :])) != 9 or len(set(grid[:, i])) != 9:
            return False
    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            subgrid = grid[row:row+3, col:col+3].flatten()
            if len(set(subgrid)) != 9:
                return False
    return True

if __name__ == "__main__":
    model = tf.keras.models.load_model("models/sudoku_solver.keras")

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
    corrected_puzzle = enforce_sudoku_constraints(solved_puzzle)

# Debugging
    print("Original puzzle:")
    print(sample_puzzle)
    print("Solved puzzle (raw):")
    print(solved_puzzle)
    print("Corrected puzzle:")
    print(corrected_puzzle)

    if is_valid_sudoku(solved_puzzle):
        print("The solved puzzle is valid.")
    else:
        print("The solved puzzle is invalid.")
