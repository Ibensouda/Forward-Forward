import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
from tensorflow.compiler.tf2xla.python import xla
import time
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print("4 Random Training samples and labels")
# idx1, idx2, idx3, idx4 = random.sample(range(0, x_train.shape[0]), 4)

# img1 = (x_train[idx1], y_train[idx1])
# img2 = (x_train[idx2], y_train[idx2])
# img3 = (x_train[idx3], y_train[idx3])
# img4 = (x_train[idx4], y_train[idx4])

# imgs = [img1, img2, img3, img4]

# plt.figure(figsize=(10, 10))

# for idx, item in enumerate(imgs):
#     image, label = item[0], item[1]
#     plt.subplot(2, 2, idx + 1)
#     plt.imshow(image, cmap="gray")
#     plt.title(f"Label : {label}")
# plt.show()

class FFDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        optimizer,
        loss_metric,
        num_epochs=20,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="ones",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.relu = keras.layers.ReLU()
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.threshold = 1.5
        self.num_epochs = num_epochs

    def call(self, x):
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        res = self.dense(x_dir)
        return self.relu(res)

    # algo de FF avec symba loss
    def forward_forward(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = tf.math.reduce_mean(tf.math.pow(self.call(x_pos), 2), 1)
                g_neg = tf.math.reduce_mean(tf.math.pow(self.call(x_neg), 2), 1)
                delta = tf.concat([g_pos - g_neg], 0)
                alpha = 5.0
                loss = tf.math.log(
                    1
                    + tf.math.exp(
                        (-alpha)*delta
                    )
                )   
                mean_loss = tf.cast(tf.math.reduce_mean(loss), tf.float32)
                self.loss_metric.update_state([mean_loss])
            gradients = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (
            tf.stop_gradient(self.call(x_pos)),
            tf.stop_gradient(self.call(x_neg)),
            self.loss_metric.result(),
        )

class FFNetwork(keras.Model):

    # Since each layer runs gradient-calculation and optimization locally, each
    # layer has its own optimizer that we pass. As a standard choice, we pass
    # the `Adam` optimizer with a default learning rate of 0.03 as that was
    # found to be the best rate after experimentation.
    # Loss is tracked using `loss_var` and `loss_count` variables.
    # Use legacy optimizer for Layer Optimizer to fix issue
    # https://github.com/keras-team/keras-io/issues/1241

    def __init__(
        self,
        dims,
        layer_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.05),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_optimizer = layer_optimizer
        self.loss_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.loss_count = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.layer_list = [keras.Input(shape=(dims[0],))]
        self.layer_list += [keras.layers.Dropout(0.3)]
        for d in range(len(dims) - 1):
            self.layer_list += [
                FFDense(
                    dims[d + 1],
                    optimizer=self.layer_optimizer,
                    loss_metric=keras.metrics.Mean(),
                )
            ]

    # This function makes a dynamic change to the image wherein the labels are
    # put on top of the original image (for this example, as MNIST has 10
    # unique labels, we take the top-left corner's first 10 pixels). This
    # function returns the original data tensor with the first 10 pixels being
    # a pixel-based one-hot representation of the labels.

    # écriture des labels sur les images
    @tf.function(reduce_retracing=True)
    def overlay_y_on_x(self, data):
        X_sample, y_sample = data
        max_sample = tf.reduce_max(X_sample, axis=0, keepdims=True)
        max_sample = tf.cast(max_sample, dtype=tf.float64)
        X_zeros = tf.zeros([10], dtype=tf.float64)
        X_update = xla.dynamic_update_slice(X_zeros, max_sample, [y_sample])
        X_sample = xla.dynamic_update_slice(X_sample, X_update, [0])
        return X_sample, y_sample

    # A custom `predict_one_sample` performs predictions by passing the images
    # through the network, measures the results produced by each layer (i.e.
    # how high/low the output values are with respect to the set threshold for
    # each label) and then simply finding the label with the highest values.
    # In such a case, the images are tested for their 'goodness' with all
    # labels.

    @tf.function(reduce_retracing=True)
    def predict_one_sample(self, x):
        goodness_per_label = []
        x = tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1]])
        for label in range(10):
            h, label = self.overlay_y_on_x(data=(x, label))
            h = tf.reshape(h, [-1, tf.shape(h)[0]])
            goodness = []
            for layer_idx in range(1, len(self.layer_list)):
                layer = self.layer_list[layer_idx]
                h = layer(h)
                goodness += [tf.math.reduce_mean(tf.math.pow(h, 2), 1)]
            goodness_per_label += [
                tf.expand_dims(tf.reduce_sum(goodness, keepdims=True), 1)
            ]
        goodness_per_label = tf.concat(goodness_per_label, 1)
        return tf.cast(tf.argmax(goodness_per_label, 1), tf.float64)

    def predict(self, data):
        x = data
        preds = list()
        preds = tf.map_fn(fn=self.predict_one_sample, elems=x)
        return np.asarray(preds, dtype=int)

    # This custom `train_step` function overrides the internal `train_step`
    # implementation. We take all the input image tensors, flatten them and
    # subsequently produce positive and negative samples on the images.
    # A positive sample is an image that has the right label encoded on it with
    # the `overlay_y_on_x` function. A negative sample is an image that has an
    # erroneous label present on it.
    # With the samples ready, we pass them through each `FFLayer` and perform
    # the Forward-Forward computation on it. The returned loss is the final
    # loss value over all the layers.

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data

        # Flatten op
        x = tf.reshape(x, [-1, tf.shape(x)[1] * tf.shape(x)[2]])

        x_pos, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, y))

        random_y = tf.random.shuffle(y)
        x_neg, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, random_y))

        h_pos, h_neg = x_pos, x_neg
        # Entrainement par couche
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, FFDense):
                print(f"Training layer {idx+1} now : ")
                h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)    
                self.loss_var.assign_add(loss)
                self.loss_count.assign_add(1.0)
            else:
                print(f"Passing layer {idx+1} now : ")
                x = layer(x)
        mean_res = tf.math.divide(self.loss_var, self.loss_count)
        return {"FinalLoss": mean_res}

x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255
y_train = y_train.astype(int)
y_test = y_test.astype(int)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.batch(60000)
test_dataset = test_dataset.batch(10000)
model = FFNetwork(dims=[784, 50, 50])


#redéfiniton du callback, pour tester le réseau à chaque étape d'entrainement
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_losses = []
        self.epoch_accuracies_test = []
        self.epoch_accuracies_train = []
    def on_epoch_end(self, epoch, logs=None):
        preds = model.predict(tf.convert_to_tensor(x_test))
        preds = preds.reshape((preds.shape[0], preds.shape[1]))
        results = accuracy_score(preds, y_test)
        accuracy = results * 100
        loss = logs.get('FinalLoss')
        preds2 = model.predict(tf.convert_to_tensor(x_train))
        preds2 = preds2.reshape((preds2.shape[0], preds2.shape[1]))
        results2 = accuracy_score(preds2, y_train)
        accuracy2 = results2 * 100
        loss2 = logs.get('FinalLoss train')
        
        self.epoch_accuracies_test.append(accuracy)
        self.epoch_accuracies_train.append(accuracy2)
        self.epoch_losses.append(loss)

        print(f"\nEpoch {epoch+1} - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")

start_time = time.time()
model.compile(
    loss="mse",
    jit_compile=True,
    metrics=[keras.metrics.Mean()],
)

# tester 1 epoch avec 100% dataset == 10 epochs avec 10% dataset par epoch
epochs = 20

custom_callback = CustomCallback()
history = model.fit(train_dataset, epochs=epochs, callbacks=[custom_callback], batch_size=128)
preds = model.predict(tf.convert_to_tensor(x_test))
preds = preds.reshape((preds.shape[0], preds.shape[1]))
results = accuracy_score(preds, y_test) 
final = math.ceil((time.time()-start_time))      
start_time = time.time()

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
history = model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[callback], validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
finalBP = math.ceil((time.time()-start_time))      

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


fig = plt.figure(figsize=(14, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(custom_callback.epoch_losses, label='FF Loss')
plt.plot(history.history['loss'], label='BP Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(custom_callback.epoch_accuracies_test, label='Accuracy test FF', color='orange')
plt.plot(custom_callback.epoch_accuracies_train, label='Accuracy train FF', color='blue')
print(history.history['accuracy'])
plt.plot(np.array(history.history['accuracy'])*100, label='Accuracy train BP')
plt.plot(np.array(history.history['val_accuracy'])*100, label='Accuracy test BP')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('Accuracies')
open('output.txt', 'w').close()
with open('output.txt', 'a') as f:
    f.write('Résultats Forward Forward : \n')
    f.write("précision : " + str(results) + ". Temps d'éxecution : " + str(final) + "\n")
    f.write('Précision BP : ' + str(test_accuracy) + ". Temps d'éxecution : " + str(finalBP) + "\n")
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()