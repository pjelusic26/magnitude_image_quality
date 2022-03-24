from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets
import models
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, required = True, help = "Path to dataset")
args = vars(ap.parse_args())

print("[INFO] Loading house attributes")

# inputPath = os.path.sep.join([args["dataset"], "HouseInfo.txt"])
df = datasets.load_house_attributes(args["dataset"])

print(f"[INFO] Dataset shape: {df.shape}")
print("[INFO] Constructing train/test split...")

(train, test) = train_test_split(df, test_size = 0.25, random_state = 42)
print(f"[INFO] Dataset train shape: {train.shape}")
print(f"[INFO] Dataset test shape: {test.shape}")

maxPrice = train["1"].max()
trainY = train["1"] / maxPrice
testY = test["1"] / maxPrice

print("[INFO] processing data...")

(trainX, testX) = datasets.process_house_attributes(df, train, test)
print(f"[INFO] Dataset train X shape: {trainX.shape}")
print(f"[INFO] Dataset test X shape: {testX.shape}")

model = models.create_mlp(trainX.shape[1], regress = True)
opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss = 'mean_absolute_percentage_error', optimizer = opt)

print("[INFO] training model...")

model.fit(
    x = trainX,
    y = trainY,
    validation_data = (testX, testY),
    epochs = 200,
    batch_size = 8
)

print("[INFO] predicting house prices...")

preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean_diff = np.mean(absPercentDiff)
std_diff = np.std(absPercentDiff)

print(f"[INFO] Mean: {mean_diff}")
print(f"[INFO] Std: {std_diff}")

print(f"[INFO] Predictions: {preds}")