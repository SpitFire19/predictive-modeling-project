import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_pinball_loss

from pathlib import Path
import sys
# Get the grandparent directory (the parent of the folder your script is in)
parent_root = Path(__file__).resolve().parents[1]

# Add it to sys.path
sys.path.append(str(parent_root))

# Import your package
from data_utils import AdaptiveKalman, FeatureEngineerExpert, DefaultFeatureEngineerExpert


Q_FINAL = 2500.0
R_FINAL = 150.0
BIAS_SHIFT = -1500.0

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

fe = FeatureEngineerExpert().fit(train)
train = fe.transform(train)
test = fe.transform(test)

TARGET = "Net_demand"

# print(train.columns.tolist())
    
# Les jours feries en facteur
# Load avec vent el le soleil, avec lasso pour choisir les features. prevoir la moyenne (quantile gaussien), enlever bh et time et nebulosity
# GBM model complet, importance de variables dans le modele gb, permutation based performance, faire attention a overfit avec boosting
# (tree-based gradient boosting) + te(Net.demand.1, Net.demand.7)
# GAM a 15 variables + Kalman
# Utiliser plus de liaisons (BAM = Big Additive Model)
# Predire la mediane (ou moyenne) peut etre mieux que le quantile 0.8
# 117 variables apres le lasso
# Utiliser les GAM univaries univaries, tracer une courbe pour chacune des covariables


exclude = ["Date", "Load", "Net_demand", 'Solar_power', 'Wind_power']
exclude = ["Date","Net_demand","Load","Solar_power","Wind_power","WeekDays","Id","Usage","Year","Month","toy"]
features = [col for col in train.columns if col not in exclude]

n_features = len(features)
mask_train = train["Date"] < "2022-04-01"
mask_cal = (train["Date"] >= "2022-04-01") & (train["Date"] < "2022-06-01")
mask_val = train["Date"] >= "2022-06-01"

print(train.head())
# print(train.dtypes)

X_train, y_train = train[mask_train][features], train[mask_train]["Net_demand"]
X_cal, y_cal = train[mask_cal][features], train[mask_cal]["Net_demand"]
X_val, y_val = train[mask_val][features], train[mask_val]["Net_demand"]
X_test = test[features]

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Add, Flatten

from tensorflow.keras.callbacks import ReduceLROnPlateau

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf


def pinball_loss(y_true, y_pred):
    alpha = 0.8  # The quantile you are targeting
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(alpha * error, (alpha - 1) * error), axis=-1)

def sliding_window(data, window_size):
    X = []
    # We iterate through the data, grabbing a 14-day block each time
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

# 1. Create the 14-day sequences
# This turns your (3317, 58) DataFrame into a (3303, 14, 58) NumPy array
window_len = 30
X_train_3d = sliding_window(X_train, window_len)
X_val_3d = sliding_window(X_val, window_len)

# 2. Adjust your target (y)
# Since the first 14 days don't have enough history to make a prediction, 
# you must drop the first 14 values of your target.
y_train_3d = y_train[window_len:].values.reshape(-1, 1)
y_val_3d = y_val[window_len:].values.reshape(-1, 1)
print(f"Training - X: {X_train_3d.shape}, y: {y_train_3d.shape}")
print(f"Validation - X: {X_val_3d.shape}, y: {y_val_3d.shape}")

def build_flat_resnet(input_dim):
    # input_dim will be 812 (14 days * 58 features)
    inputs = Input(shape=(input_dim,))
    
    # --- Initial Entry ---
    x = Dense(256)(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # --- Residual Block 1 ---
    shortcut = x 
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut]) # Adds the original 'x' back to prevent signal loss
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)

    # --- Residual Block 2 ---
    shortcut = x
    x = Dense(128)(x)
    # We must project the shortcut to 128 to match the new size for addition
    shortcut = Dense(128)(shortcut) 
    
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)

    # --- Output ---
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    return model

def build_diamond_mlp_with_dropout(time_window, n_features):
    model = Sequential([
        Input(shape=(time_window, n_features)),
        Flatten(), # Input: n_features
        
        # Layer 1: n_features -> 100
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Layer 2: 100 -> 200
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Layer 3: 200 -> 400 (The Diamond Peak)
        Dense(400, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Layer 4: 400 -> 200
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Layer 5: 200 -> 100
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output Layer: 100 -> 1
        Dense(1)
    ])
    return model

# 1. Setup Data
n_samples = len(X_train)
# Ensure X_val is also reshaped or you will get a shape error during evaluation

# 2. Build and Compile
input_dim = len(features)
model = build_diamond_mlp_with_dropout(window_len, input_dim)
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=custom_optimizer, loss=pinball_loss)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Watch the validation loss
    factor=0.2,            # Multiply the learning rate by 0.2 when it plateaus
    patience=5,            # Wait 5 epochs with no improvement before acting
    min_lr=1e-6,           # Don't let the learning rate go below this value
    verbose=1              # Print a message when the rate changes
)

# Use the 3D arrays and the adjusted targets
model.fit(
    X_train_3d, 
    y_train_3d, 
    epochs=1000, 
    batch_size=512,
    validation_data=(X_val_3d, y_val_3d),
   #  callbacks=[reduce_lr], # From the previous step
    verbose=1
)

# For evaluation, use the windowed validation set
preds = model.predict(X_val)

loss = mean_pinball_loss(y_val, preds, alpha=0.8)
print(f"Pinball Loss: {loss}")

preds_test = model.predict(X_test)
submission = pd.DataFrame({"Id": test["Id"], "Net_demand": preds_test})
submission.to_csv("nn.csv", index=False)