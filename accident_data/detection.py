import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os


batch_size = 32
img_height = 48
img_width = 48

image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255,
)
dataset_folder = r"E:\drive\pfe\accident\cnn_detection\accident_data\data"
training_ds = image_data_generator.flow_from_directory(
    os.path.join(dataset_folder, 'train'),
    seed=101,
    target_size= (img_height, img_width),
    batch_size= batch_size,
    color_mode = 'grayscale'

)

testing_ds = image_data_generator.flow_from_directory(
    os.path.join(dataset_folder, 'test'),
    seed=101,
    target_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode = 'grayscale')

validation_ds =  image_data_generator.flow_from_directory(
    os.path.join(dataset_folder, 'val'),
    seed=101,
    target_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode = 'grayscale')

class_names = training_ds.class_indices

img_shape = (img_height, img_width, 3)
weight=r"E:\drive\pfe\accident\cnn_detection\accident_data\weights.h5"

base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights=weight)
len(base_model.layers)


for layer in base_model.layers[:130]:
  layer.trainable = False
for layer in base_model.layers[130:]:
  layer.trainable = True


input_ = tf.keras.layers.Input(shape = (48, 48, 1))
x = tf.keras.layers.Conv2D(32,3,  activation = 'relu', strides = 2)(input_)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(64,3,  activation = 'relu', strides = 2)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(128,3,  activation = 'relu', strides = 2)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation = "elu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(32, activation = "elu")(x)
x = tf.keras.layers.Dropout(0.6)(x)
output = tf.keras.layers.Dense(2, activation= 'softmax')(x)
m = tf.keras.models.Model(
    inputs = input_, 
    outputs = output
)
m.summary()


m.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001),loss='binary_crossentropy', metrics=['accuracy'])




history = m.fit(training_ds, validation_data = validation_ds, epochs = 160,
                batch_size = 64,
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 4, 
                                                     restore_best_weights = True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', 
                                                         patience = 3, factor = 0.1)
                ])






m.evaluate(training_ds)
m.evaluate(validation_ds)
m.evaluate(testing_ds)


# Save the trained model
model_dir = r"E:\drive\pfe\accident\cnn_detection\accident_data"
m.save(os.path.join(model_dir, "trained_model.h5"))

# Save the class names
class_names = {
    0: "No Accident",
    1: "Accident"
}

class_names_df = pd.DataFrame(class_names, index=[0])
class_names_df.to_csv(os.path.join(model_dir, "class_acc.csv"), index=False)





