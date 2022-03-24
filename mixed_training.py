import datasets
import models
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
import numpy as np
import argparse
import os

'''
Pt 1 - Numerical Input Data
https://pyimagesearch.com/2019/01/21/regression-with-keras/

Pt 2 - Image Input Data
https://pyimagesearch.com/2019/01/28/keras-regression-and-cnns/

Pt 3 - Combined Input Data
https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
'''

nl = '\n'

### COMMAND LINE ARGUMENTS ###
### COMMAND LINE ARGUMENTS ###
### COMMAND LINE ARGUMENTS ###

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, required = True, 
    help = "Path to image dataset")
args = vars(ap.parse_args())

### / ###
### / ###
### / ###

### DATASET ###
### DATASET ###
### DATASET ###

print(f"[INFO] loading frequency domain attributes...")
inputPath = os.path.sep.join([args["dataset"], "32_gamma_34_greyscale_clean.csv"])
df = datasets.load_house_attributes(inputPath)
df = datasets.process_house_attributes(df)
print(df.head())

print(f"[INFO] loading grayscale images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0

# Sklearn split
X_train_inputs, X_test_inputs, X_train_images, X_test_images = train_test_split(
    df, 
    images,
    test_size = 0.25, 
    random_state = 42
)

maxPrice = X_train_inputs.iloc[:,-1].max()
print(f"[INFO] Max price: {maxPrice}")
trainY = X_train_inputs.iloc[:,-1:]
testY = X_test_inputs.iloc[:,-1:]

### / ###
### / ###
### / ###

### MODEL ###
### MODEL ###
### MODEL ###

mlp = models.create_mlp(X_train_inputs.shape[1], regress = False)
cnn = models.create_cnn(32, 32, 1, regress = False)
combined_input = concatenate([mlp.output, cnn.output])
print(f"[INFO] Combined input shape: {combined_input.shape}")

x = Dense(16, activation = 'relu')(combined_input)
x = Dense(8, activation = 'relu')(combined_input)
x = Dense(1, activation = 'linear')(x)

model = Model(inputs = [mlp.input, cnn.input], outputs = x)

### / ###
### / ###
### / ###

### TRAINING ###
### TRAINING ###
### TRAINING ###

opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss = 'mean_absolute_percentage_error', optimizer = opt)

print(f"[INFO] training model...")
model.fit(
    x = [X_train_inputs, X_train_images],
    y = trainY,
    validation_data = ([X_test_inputs, X_test_images], testY),
    epochs = 20,
    batch_size = 128
)

print(f"[INFO] predicting gamma...")
preds = model.predict([X_test_inputs, X_test_images])

### / ###
### / ###
### / ###

### EVALUATION ###
### EVALUATION ###
### EVALUATION ###

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

### / ###
### / ###
### / ###