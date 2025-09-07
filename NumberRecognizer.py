import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Define the model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Flatten 28x28 input into 1D
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # Hidden layer
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # Another hidden layer
# model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Output layer (10 classes)

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=65)

# # Save the model
# model.save('handwritten.keras')


# mnist = tf.keras.datasets.mnist
# (_, _), (x_test, y_test) = mnist.load_data()
# x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('handwritten.keras')
loss, accuracy = model.evaluate(x_test, y_test)
print("Model Test Loss:", loss)
print("Model Test Accuracy:", accuracy)

# ===============================
# Loop through images in folder
# ===============================
folder_path = os.path.expanduser("~/Downloads/nums4")

for filename in sorted(os.listdir(folder_path)):
    if filename.lower().startswith("x") and filename.lower().endswith(".png"):
        try:
            img_path = os.path.join(folder_path, filename)

            # Load in grayscale
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            # img = cv2.resize(img, (28, 28))
            # img = tf.keras.utils.normalize(img, axis=1)
            # img = img.astype(np.float32)
            # img = img.reshape(1, 28, 28)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Resize to 28x28 to match MNIST input
            #img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalize pixel values to the 0–1 range
            img = img.astype(np.float32) / 255.0

            # Reshape to match model input: (1, 28, 28)
            img = img.reshape(1, 28, 28)






            # Predict
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction)

            print(f"{filename} --> This digit is probably a {predicted_digit}")

            # Display
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f"{filename}: Predicted {predicted_digit}")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

