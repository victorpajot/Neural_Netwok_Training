from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Downloaded')

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(512, activation="relu"),
                        layers.Dense(10, activation="softmax") ])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10, batch_size=128)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
