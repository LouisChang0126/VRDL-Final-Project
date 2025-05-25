import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import kagglehub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

IMAGE_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sub = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv", index_col=0)
file_names = sub.index.values

def load_image(path):
    img = tf.io.read_file(path)
    return tf.image.decode_jpeg(img, channels=3)

path = kagglehub.model_download("google/cropnet/tensorFlow1/classifier-cassava-disease-v1/1")
cropnet = hub.KerasLayer(path, trainable=False, output_key="image_classifier:logits")

central_fracs = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
def make_views(img):
    views = []
    for cf in central_fracs:
        v = tf.image.central_crop(img, cf)
        v = tf.image.resize(v, [224, 224])
        views.append(v)
        views.append(tf.image.flip_left_right(v))
    return tf.stack(views)

T = 1.0
preds = []
for fn in file_names:
    img = load_image(os.path.join(IMAGE_DIR, fn))
    views = make_views(img) / 255.0
    logits = cropnet({"images": views}, training=False)
    logits = tf.reduce_sum(logits, axis=0) / T
    prob   = tf.nn.softmax(logits)
    preds.append(tf.argmax(prob).numpy())

sub["label"] = preds
sub.to_csv("submission.csv")
sub