import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score
import numpy as np

# Load the MNIST dataset (note: if you want Fashion MNIST, change the import)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Preprocessing: Normalize and reshape data
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255
x_train = np.expand_dims(x_train, -1)  # Expand dims for channel compatibility
x_test = np.expand_dims(x_test, -1)

# Reshape labels (if needed)
y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

# Print dataset details
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Define the model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(100, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# Define callback and compile model
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Train the model with the callback
history = model.fit(x_train, y_train, epochs=50, batch_size=128, callbacks=[callback])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
