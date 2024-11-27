# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:38:49 2024

@author: NISHAN
"""

# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

num_classes = 8

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_data = train_datagen.flow_from_directory('Training',
                                               target_size=(150, 150),
                                               batch_size=30,
                                               class_mode='categorical')

val_data = test_datagen.flow_from_directory('Testing',
                                            target_size=(150, 150),
                                            batch_size=30,
                                            class_mode='categorical')
test_data = test_datagen.flow_from_directory('Validation',
                                             target_size=(150, 150),
                                             batch_size=30,
                                             class_mode='categorical')

pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3), pooling='avg')

for layer in pre_trained.layers:
    layer.trainable = False

x = pre_trained.output

x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dropout(0.5)(x)
x = Dense(900, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=pre_trained.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
STEP_SIZE_TRAIN = train_data.n // train_data.batch_size
STEP_SIZE_VALID = val_data.n // val_data.batch_size

history = model.fit_generator(train_data,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=val_data,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=50,
                              verbose=1)

model_name = 'Cardamom_Nish_Inc.h5'
model.save(model_name, save_format='h5')

import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='training set')
plt.plot(history.history['val_loss'], label='test set')
plt.legend()

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='training set')
plt.plot(history.history['val_accuracy'], label='test set')
plt.legend()

class_map = train_data.class_indices
classes = []
for key in class_map.keys():
    classes.append(key)

#import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

from tensorflow.keras.models import load_model
model = load_model('Cardamom_Nish_Inc.h5')

def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(150, 150))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = np.argmax(prediction)
    plt.title("Prediction - {}".format(str(classes[index]).title()), size=18, color='red')
    plt.imshow(img_array)
    return index


def evaluate_model(test_dir, model):
    y_true = []
    y_pred = []

    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(test_dir, class_name)
        for filename in os.listdir(class_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(class_folder, filename)
                true_label = class_idx
                pred_label = predict_image(file_path, model)
                y_true.append(true_label)
                y_pred.append(pred_label)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print(classification_report(y_true, y_pred, target_names=classes))




# Assuming you have a folder named 'test_data' with subfolders for each class
evaluate_model('D:\Projrct Progs\Cardamom\Testing', model)


# changed here ??????????????????
# def predict_image(filename, model):
#     img_ = image.load_img(filename, target_size=(150, 150))
#     img_array = image.img_to_array(img_)
#     img_processed = np.expand_dims(img_array, axis=0)
#     img_processed /= 255.

#     prediction = model.predict(img_processed)

#     index = np.argmax(prediction)
#     plt.title("Prediction - {}".format(str(classes[index]).title()), size=18, color='red')
#     plt.imshow(img_array)


# predict_image('LC6_dl_44.jpg', model)


# confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# # Step 1: Make predictions on the validation/test set
# # Note: Here, val_data is used. You can change it to test_data if you want to use the test set.
# Y_pred = model.predict(val_data)
# y_pred = np.argmax(Y_pred, axis=1)

# # Step 2: Get the true labels
# y_true = val_data.classes

# # Step 3: Generate the confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# # Ensure the classes list is ordered correctly based on the class indices
# classes = [key for key in train_data.class_indices.keys()]

# # Step 4: Visualize the confusion matrix using seaborn
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.title('Confusion Matrix')
# plt.show()



# roc(receiver operating Charactersitics curve) auc(area under curve) in terms of rate 
# cohen kappa score
# ieee conference paper -> template
