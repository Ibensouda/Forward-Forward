import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Build the model
def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(784,)),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Custom training loop to simulate Forward-Forward Algorithm
epochs = 5
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    # Custom gradient modification can be done here
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(inputs, labels):
    predictions = model(inputs, training=False)
    loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    return loss

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    # Training loop
    for step, (inputs, labels) in enumerate(train_dataset):
        loss = train_step(inputs, labels)
        if step % 100 == 0:
            print(f'Step {step}, Loss: {tf.reduce_mean(loss)}')
    
    # Testing loop
    test_loss = 0.0
    for step, (inputs, labels) in enumerate(test_dataset):
        loss = test_step(inputs, labels)
        test_loss += tf.reduce_sum(loss)
    test_loss /= len(x_test)
    print(f'Test Loss: {test_loss.numpy()}')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
