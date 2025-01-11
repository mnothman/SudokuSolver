from sklearn.model_selection import train_test_split
from data_loader import load_sudoku_data, preprocess_data
from model import create_sudoku_model

def train_sudoku_model(csv_path, model_save_path, max_samples=100000):
    quizzes, solutions = load_sudoku_data(csv_path)

    quizzes = quizzes[:max_samples]
    solutions = solutions[:max_samples]
    
    quizzes_normalized, solutions_one_hot, mask = preprocess_data(quizzes, solutions)

    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        quizzes_normalized, solutions_one_hot, test_size=0.2, random_state=42
    )
    mask_train, mask_val = train_test_split(mask, test_size=0.2, random_state=42)

    model = create_sudoku_model()
    model.fit(
        X_train[..., None],
        y_train * mask_train[..., None],
        epochs=50, batch_size=64,
        validation_data=(X_val[..., None], y_val * mask_val[..., None])
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_sudoku_model("data/datasetTraining/sudoku_training/sudoku.csv", "models/sudoku_solver.keras", max_samples=100000)
