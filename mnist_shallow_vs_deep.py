from tensorflow import keras
from tensorflow.keras import layers, models

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten 28x28 images into 784-length vectors & normalize (0-1)
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# 2. Build shallow model (1 hidden layer)
shallow = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 3. Build deep model (3 hidden layers)
deep = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 4. Compile both models
for model in [shallow, deep]:
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

# 5. Train and evaluate Shallow Model
print("\n--- Training Shallow Model ---")
shallow.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2)
loss_s, acc_s = shallow.evaluate(x_test, y_test, verbose=0)

# 6. Train and evaluate Deep Model
print("\n--- Training Deep Model ---")
deep.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2)
loss_d, acc_d = deep.evaluate(x_test, y_test, verbose=0)

# 7. Show results
print("\nResults:")
print(f"Shallow Model Accuracy: {acc_s:.4f}")
print(f"Deep Model Accuracy   : {acc_d:.4f}")
