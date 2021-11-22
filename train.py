from typing import get_args
import tensorflow as tf
import argparse
import os
import datetime
import requests
from tensorflow.python.lib.io import file_io
import time
import datetime
import requests
from utils import send_message_to_slack
from utils import request_deploy_api
from PIL import Image
import numpy as np
import glob

def get_args():
    parser = argparse.ArgumentParser(description='Tensorflow CIFAR-10 Example')
  
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)')

    parser.add_argument(
        '--batchsize',
        type=int,
        default=128,
        metavar='N',
        help='batch size (default: 128)')

    args = parser.parse_args()
    return args

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def load_data(test_size=0.2):
    LABELS = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}

    test_directory = "./data/test"
    train_directory = "./data/train"

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for label in os.listdir(train_directory):
        if label not in LABELS.keys():
            continue
        for file in glob.glob(f"{train_directory}/{label}/*.jpg"):
            image = Image.open(file)
            imgArray = np.array(image)
            train_x.append(imgArray)
            train_y.append(LABELS[label])

    for label in os.listdir(test_directory):
        if label not in LABELS.keys():
            continue
        for file in glob.glob(f"{test_directory}/{label}/*.jpg"):
            image = Image.open(file)
            imgArray = np.array(image)
            test_x.append(imgArray)
            test_y.append(LABELS[label])

    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))

def main():

    start = time.time()

    args = get_args()
    epochs = args.epochs
    batch_size = args.batchsize
    gcp_bucket = os.getenv("GCS_BUCKET")

    bucket_path = os.path.join(gcp_bucket, "cifar_model")

    model = get_model()
    (train_x, train_y), (test_x, test_y) = load_data()
    
    train_x, test_x = train_x/255.0, test_x/255.0
    
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    loss, acc = model.evaluate(test_x, test_y)
    print("model acc: {:.4f}, model loss: {:.4f}".format(acc, loss))

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    save_path = "save_at_{}_acc_{}_loss_.h5".format(timestamp, acc, loss)
    model.save(save_path)

    gs_path = os.path.join(bucket_path, save_path)

    with file_io.FileIO(save_path, mode='rb') as input_file:
        with file_io.FileIO(gs_path, mode='wb+') as output_file:
            output_file.write(input_file.read())

    end = time.time()
    sec = (end - start) 
    training_time = str(datetime.timedelta(seconds=sec)).split(".")[0]

    slack_url = os.getenv("WEB_HOOK_URL")
    if slack_url != None:
        send_message_to_slack(slack_url, acc, loss, training_time, gs_path)

    request_deploy_api(gs_path)
    
if __name__ == '__main__':
  main()
    # (train_x, train_y), (test_x, test_y) = load_data()
    # print(len(train_x), len(train_y), len(test_x), len(test_y))