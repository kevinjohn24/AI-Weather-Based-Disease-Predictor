from tensorflow import keras
from tensorflow.keras import layers, models
import time

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# 2. Build model function (2 hidden layers)
def build_model(activation):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation=activation),
        layers.Dense(128, activation=activation),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# 3. Train & test for each activation
for act in ["sigmoid", "tanh", "relu"]:
    print(f"\n--- Training with {act.upper()} activation ---")
    model = build_model(act)
    start = time.time()
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    end = time.time()
    print(f"{act.upper()} Test Accuracy: {acc:.4f} | Training Time: {end - start:.1f}s")
