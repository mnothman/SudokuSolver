from tensorflow import keras
# from tensorflow.keras import layers, models

def create_sudoku_model():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(9, 9)),
        # keras.layers.Flatten(input_shape=(9, 9)), 
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(81 * 9, activation='softmax'),
        keras.layers.Reshape((9, 9, 9))
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model
