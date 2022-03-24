from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets
import models
import numpy as np
import argparse
import locale
import os

nl = '\n'

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required = True, help="Path to image dataset")
args = vars(ap.parse_args())

print("[INFO] Loading house attributes")

inputPath = os.path.sep.join([args["dataset"], "32_gamma_34_greyscale_clean.csv"])
df = datasets.load_house_attributes(inputPath)

# Select a part of the dataset
# df = df.head(20)

print(f"df shape: {df.shape}")

print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])
# images = images / 255.0
print(f"images shape: {images.shape}")

# Sklearn split
X_train_inputs, X_test_inputs, X_train_images, X_test_images = train_test_split(
    df, 
    images,
    test_size = 0.25, 
    random_state = 42
)

# maxPrice = X_train_inputs.iloc[:,-1].max()
trainY = X_train_inputs.iloc[:,-1:]
print(f"Train output shape:{trainY.shape}")
print(f"Train output:{nl}{trainY}")
testY = X_test_inputs.iloc[:,-1:]
print(f"Test output shape:{testY.shape}")
print(f"Test output:{nl}{testY}")

model = models.create_cnn(32, 32, 1, regress = True)
opt = Adam(lr = 1e-3, decay = 1e-3 / 25)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

model.fit(
    x = X_train_images, 
    y = trainY, 
    validation_data = (X_test_images, testY),
    epochs = 25,
    batch_size = 32
)

print("[INFO] predicting house prices...")
preds = model.predict(X_test_images)

diff = preds - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean_diff = np.mean(absPercentDiff)
std_diff = np.std(absPercentDiff)
max_diff = np.amax(absPercentDiff)
min_diff = np.amin(absPercentDiff)

print(f"[INFO] Mean: {mean_diff}")
print(f"[INFO] Std: {std_diff}")
print(f"[INFO] Max: {max_diff}")
print(f"[INFO] Min: {min_diff}")

print(f"[INFO] Predictions VS Real price:{nl}{preds}{nl}{testY}")
