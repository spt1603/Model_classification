import os
import cv2
import tensorflow as tf

model = tf.keras.models.load_model("cats_vs_raccoons.keras")

img = cv2.imread("test/cat11.jpg")
if img is None:
    print("No image")

img = cv2.resize(img, (128, 128))
img = img/255.0
img = tf.expand_dims(img, axis=0)

pred = model.predict(img)
print(pred)
DATASET_PATH = "dataset"
class_names = os.listdir(DATASET_PATH)
predicted_class = class_names[int(bool(pred[0]>0.5))]
print(predicted_class)