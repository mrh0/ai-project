import tensorflow as tf
import numpy as np
import PIL

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

img_url = "https://i.gyazo.com/75e1f97b53cf8da525f5c330c13ae236.png"
img_path = tf.keras.utils.get_file('6', origin=img_url)

img = tf.keras.utils.load_img(
    img_path, target_size=(28, 28), grayscale=True
)

np.reshape(img, (28, 28, 1))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f}% confidence."
    .format(np.argmax(score), 100 * np.max(score))
)