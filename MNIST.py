import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset (note: if you want Fashion MNIST, change the import)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing: Normalize and reshape data
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255
x_train = np.expand_dims(x_train, -1)  # Expand dims for channel compatibility
x_test = np.expand_dims(x_test, -1)

# Print dataset details
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Define the model
model = keras.Sequential([
    keras.Input(shape=(28,28,1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

# Define callback and compile model
callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Train the model with the callback
history = model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[callback], validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Plot the training and validation accuracy
plt.figure(figsize=(14, 6))
plt.plot(np.array(history.history['accuracy'])*100, label='Training Accuracy')
plt.plot(np.array(history.history['val_accuracy'])*100, label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot the training and validation loss
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
