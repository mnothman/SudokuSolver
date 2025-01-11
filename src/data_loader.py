import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.utils import to_categorical
# up to date: keras.utils.to_categorical(x, num_classes=None)

def load_sudoku_data(csv_path):
    df = pd.read_csv(csv_path)
    quizzes = np.array([list(map(int, quiz)) for quiz in df['quizzes']])
    solutions = np.array([list(map(int, solution)) for solution in df['solutions']])
    quizzes = quizzes.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))
    return quizzes, solutions

def preprocess_data(quizzes, solutions):
    mask = (quizzes == 0).astype(float)  # 1 empty, 0 filled

    quizzes_normalized = quizzes.astype(float)
    quizzes_normalized[quizzes == 0] = -1
    quizzes_normalized = quizzes / 9.0

    solutions_one_hot = keras.utils.to_categorical(solutions - 1, num_classes=9)
    solutions_one_hot = solutions_one_hot.reshape(-1, 9, 9, 9)
    return quizzes_normalized, solutions_one_hot, mask
