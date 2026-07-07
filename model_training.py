from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_model(input_shape=(224, 224, 3)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_data, val_data, epochs=10, batch_size=32):
    history = model.fit(train_data, 
                        validation_data=val_data, 
                        epochs=epochs, 
                        batch_size=batch_size)
    return history

if __name__ == "__main__":
    # Example usage (assuming train_data and val_data are prepared)
    model = create_model()
    # train_model(model, train_data, val_data)  # Uncomment and provide data to train the model
    # model.save('models/model.h5')  # Uncomment to save the model after training