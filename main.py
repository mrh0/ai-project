import tensorflow as tf
import numpy as np
import PIL
import pandas as pd

#mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = emnist.load_data()

train_data = pd.read_csv("./data/emnist-letters-train.csv")
test_data = pd.read_csv("./data/emnist-letters-test.csv")

mappings = []
with open("./data/emnist-letters-mapping.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        m = line.replace("\n", "").replace("\r", "").split(" ")
        mappings.append(m)

y_train = np.array(train_data.iloc[:,0].values)
x_train = np.array(train_data.iloc[:,1:].values) / 255.0

y_test = np.array(test_data.iloc[:,0].values)
x_test = np.array(test_data.iloc[:,1:].values) / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)






train_images_number = x_train.shape[0]
train_images_height = 28
train_images_width = 28
train_images_size = train_images_height*train_images_width

x_train = x_train.reshape(train_images_number, train_images_height, train_images_width, 1)

test_images_number = x_test.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width

x_test = x_test.reshape(test_images_number, test_images_height, test_images_width, 1)


number_of_classes = 27


#y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)
#y_test = tf.keras.utils.to_categorical(y_test, number_of_classes)

model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)))
#model.add(tf.keras.layers.MaxPooling2D(2,2))

#model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#img_url = "https://i.gyazo.com/75e1f97b53cf8da525f5c330c13ae236.png"
#img_path = tf.keras.utils.get_file('imgC', origin=img_url)

img = tf.keras.utils.load_img(
    "./data/images/imgH.png", target_size=(28, 28), color_mode = "grayscale"
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array = img_array / 255.0
#print(img_array)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

def letterString(i):
    return "Letter: {}, '{}', '{}'".format(i, chr(int(mappings[i-1][1])), chr(int(mappings[i-1][2])))

print(
    "This image most likely belongs to {} with a {:.2f}% confidence."
    .format(letterString(np.argmax(score)), 100 * np.max(score))
)