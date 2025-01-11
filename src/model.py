from tensorflow import keras
# from tensorflow.keras import layers, models

def create_sudoku_model():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(9, 9, 1)),  # Add channel dimension for convolution
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(81 * 9, activation='softmax'),
        keras.layers.Reshape((9, 9, 9))  # Reshape to (9x9 grid, 9 probabilities per cell)
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model
