# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


!wget -q https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')

# Splitting the 80% of the dataset into training data and the rest into test sets
split_index = int(0.8 * len(dataset))
train_dataset = dataset[:split_index+1]
test_dataset = dataset[split_index:]

# Removing expenses from labels as that is what model is tring to calculate
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Defining both numerical and categorical columns
CATEGORICAL_COLUMNS = ['sex', 'smoker', 'region']
NUMERICAL_COLUMNS = ['age', 'bmi', 'children']

# Using one-hot encoding to process categorical features
def encode_categorical_feature(feature, vocabulary):
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocabulary, output_mode='one_hot')
    return lookup_layer(feature)

# Preprocessing categorical columns
input_layers = []
feature_encodings = []

for column in CATEGORICAL_COLUMNS:
    input_layer = tf.keras.Input(shape=(1,), name=column, dtype=tf.string)
    vocab = train_dataset[column].unique()
    encoded_feature = encode_categorical_feature(input_layer, vocab)
    input_layers.append(input_layer)
    feature_encodings.append(encoded_feature)

# Set up preprocessing layers for numerical features incorporating normailization to get rid of any outlier data
for column in NUMERICAL_COLUMNS:
    input_layer = tf.keras.Input(shape=(1,), name=column)
    normalized_feature = tf.keras.layers.Normalization()(input_layer)
    input_layers.append(input_layer)
    feature_encodings.append(normalized_feature)

# Combining all the feature encodings together
combined_features = tf.keras.layers.concatenate(feature_encodings)

# Building the model
x = layers.Dense(256, activation='relu')(combined_features)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1)(x)

model = tf.keras.Model(inputs=input_layers, outputs=output)

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['mae', 'mse'])


def dataframe_to_dataset(df, labels, shuffle=True, batch_size=32):
    df_copy = df.copy()
    dataset = tf.data.Dataset.from_tensor_slices((dict(df_copy), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df_copy))
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 32
train_ds = dataframe_to_dataset(train_dataset, train_labels, shuffle=True, batch_size=batch_size)
test_ds = dataframe_to_dataset(test_dataset, test_labels, shuffle=False, batch_size=batch_size)


# Evaluating the model on the test dataset and finding the mean absolute error
loss, mae, mse = model.evaluate(test_ds, verbose=2)
print("Testing set Mean Absolute Error:" + str(mae))
