# Melanona_Assignment_SairamK

**Melanoma Detection**
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

**Problem Statement:**
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

**Data Description:**
The data set contains the following categories:

Actinic keratosis 
Basal cell carcinoma 
Dermatofibroma 
Melanoma 
Nevus 
Pigmented benign keratosis 
Seborrheic keratosis 
Squamous cell carcinoma 
Vascular lesion

pip install tensorflow
Requirement already satisfied: tensorflow in c:\users\sairam\anaconda3\lib\site-packages (2.9.1)
Requirement already satisfied: protobuf<3.20,>=3.9.2 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (3.19.4)
Requirement already satisfied: google-pasta>=0.1.1 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (0.2.0)
Requirement already satisfied: wrapt>=1.11.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.12.1)
Requirement already satisfied: numpy>=1.20 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.20.3)
Requirement already satisfied: packaging in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (21.0)
Requirement already satisfied: six>=1.12.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.16.0)
Requirement already satisfied: tensorflow-estimator<2.10.0,>=2.9.0rc0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (2.9.0)
Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.1.2)
Requirement already satisfied: setuptools in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (58.0.4)
Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (3.10.0.2)
Requirement already satisfied: tensorboard<2.10,>=2.9 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (2.9.1)
Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (3.3.0)
Requirement already satisfied: libclang>=13.0.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (14.0.1)
Requirement already satisfied: keras<2.10.0,>=2.9.0rc0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (2.9.0)
Requirement already satisfied: h5py>=2.9.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (3.2.1)
Requirement already satisfied: absl-py>=1.0.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.1.0)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (0.26.0)
Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (0.4.0)
Requirement already satisfied: astunparse>=1.6.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers<2,>=1.12 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.12)
Requirement already satisfied: termcolor>=1.1.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.1.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\sairam\anaconda3\lib\site-packages (from tensorflow) (1.46.3)
Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\sairam\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.37.0)
Requirement already satisfied: markdown>=2.6.8 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (3.3.7)
Requirement already satisfied: requests<3,>=2.21.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.26.0)
Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.7.0)
Requirement already satisfied: werkzeug>=1.0.1 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.0.2)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (0.4.6)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (0.6.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\sairam\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (1.8.1)
Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\sairam\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (4.8)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\sairam\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (5.2.0)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\sairam\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (0.2.8)
Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\sairam\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in c:\users\sairam\anaconda3\lib\site-packages (from markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow) (4.8.1)
Requirement already satisfied: zipp>=0.5 in c:\users\sairam\anaconda3\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow) (3.6.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\sairam\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (0.4.8)
Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\sairam\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2.0.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\sairam\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (1.26.7)
Requirement already satisfied: idna<4,>=2.5 in c:\users\sairam\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (3.2)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\sairam\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2021.10.8)
Requirement already satisfied: oauthlib>=3.0.0 in c:\users\sairam\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow) (3.2.0)
Requirement already satisfied: pyparsing>=2.0.2 in c:\users\sairam\anaconda3\lib\site-packages (from packaging->tensorflow) (3.0.4)
Note: you may need to restart the kernel to use updated packages.

import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from sklearn.datasets import load_files
from keras.utils import np_utils

data_dir_train1 = pathlib.Path(r"C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train")
data_dir_test1 = pathlib.Path(r'C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Test')
image_count_train1 = len(list(data_dir_train1.glob('*/*.jpg')))
print("Train data consists of {} images.".format(image_count_train1))

image_count_test1 = len(list(data_dir_test1.glob('*/*.jpg')))
print("Test data consists of {} images.".format(image_count_test1))

Train data consists of 2239 images.
Test data consists of 118 images.

batch_size = 32
img_height = 180
img_width = 180
epochs = 20
seed = 123

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train1,
  seed=123,
  validation_split = 0.2,
  subset = "training",
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
Found 2239 files belonging to 9 classes.
Using 1792 files for training.

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train1,
  seed=123,
  validation_split = 0.2,
  subset = "validation",
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
Found 2239 files belonging to 9 classes.
Using 447 files for validation.

class_names = train_ds.class_names
print("As we can see that we have 9 category in our dataset. They are as follows:\n")
for i,j in enumerate(class_names):
    print(str(i+1) + ' - ' + j)
    
As we can see that we have 9 category in our dataset. They are as follows:

1 - actinic keratosis
2 - basal cell carcinoma
3 - dermatofibroma
4 - melanoma
5 - nevus
6 - pigmented benign keratosis
7 - seborrheic keratosis
8 - squamous cell carcinoma
9 - vascular lesion

import matplotlib.pyplot as plt

plt.figure(figsize = (10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
    
(32, 180, 180, 3)
(32,)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.summary()

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 180, 180, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 180, 180, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 90, 90, 16)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 90, 90, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 45, 45, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 45, 45, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 22, 22, 64)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 30976)             0         
                                                                 
 dense (Dense)               (None, 128)               3965056   
                                                                 
 dense_1 (Dense)             (None, 9)                 1161      
                                                                 
=================================================================
Total params: 3,989,801
Trainable params: 3,989,801
Non-trainable params: 0
_________________________________________________________________

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

Epoch 1/20
56/56 [==============================] - 24s 320ms/step - loss: 2.0271 - accuracy: 0.2277 - val_loss: 1.9458 - val_accuracy: 0.2036
Epoch 2/20
56/56 [==============================] - 17s 298ms/step - loss: 1.7457 - accuracy: 0.3739 - val_loss: 1.5122 - val_accuracy: 0.4698
Epoch 3/20
56/56 [==============================] - 17s 297ms/step - loss: 1.4727 - accuracy: 0.4955 - val_loss: 1.5122 - val_accuracy: 0.4765
Epoch 4/20
56/56 [==============================] - 17s 299ms/step - loss: 1.3870 - accuracy: 0.5223 - val_loss: 1.4612 - val_accuracy: 0.5101
Epoch 5/20
56/56 [==============================] - 17s 296ms/step - loss: 1.2421 - accuracy: 0.5698 - val_loss: 1.5520 - val_accuracy: 0.4743
Epoch 6/20
56/56 [==============================] - 17s 300ms/step - loss: 1.1571 - accuracy: 0.5993 - val_loss: 1.3996 - val_accuracy: 0.5414
Epoch 7/20
56/56 [==============================] - 17s 310ms/step - loss: 1.0608 - accuracy: 0.6300 - val_loss: 1.3759 - val_accuracy: 0.5526
Epoch 8/20
56/56 [==============================] - 17s 296ms/step - loss: 0.9572 - accuracy: 0.6691 - val_loss: 1.5650 - val_accuracy: 0.5481
Epoch 9/20
56/56 [==============================] - 17s 302ms/step - loss: 0.8622 - accuracy: 0.6763 - val_loss: 1.5532 - val_accuracy: 0.5123
Epoch 10/20
56/56 [==============================] - 17s 300ms/step - loss: 0.7876 - accuracy: 0.7227 - val_loss: 1.6125 - val_accuracy: 0.5235
Epoch 11/20
56/56 [==============================] - 17s 303ms/step - loss: 0.7489 - accuracy: 0.7199 - val_loss: 1.8547 - val_accuracy: 0.5034
Epoch 12/20
56/56 [==============================] - 17s 306ms/step - loss: 0.6534 - accuracy: 0.7522 - val_loss: 1.8446 - val_accuracy: 0.5280
Epoch 13/20
56/56 [==============================] - 17s 303ms/step - loss: 0.5267 - accuracy: 0.8080 - val_loss: 2.0707 - val_accuracy: 0.5302
Epoch 14/20
56/56 [==============================] - 18s 314ms/step - loss: 0.4782 - accuracy: 0.8265 - val_loss: 1.8991 - val_accuracy: 0.5190
Epoch 15/20
56/56 [==============================] - 18s 324ms/step - loss: 0.4817 - accuracy: 0.8304 - val_loss: 2.1430 - val_accuracy: 0.5168
Epoch 16/20
56/56 [==============================] - 17s 309ms/step - loss: 0.4021 - accuracy: 0.8493 - val_loss: 2.0480 - val_accuracy: 0.4832
Epoch 17/20
56/56 [==============================] - 17s 310ms/step - loss: 0.3538 - accuracy: 0.8549 - val_loss: 2.2528 - val_accuracy: 0.5145
Epoch 18/20
56/56 [==============================] - 18s 318ms/step - loss: 0.2781 - accuracy: 0.8862 - val_loss: 2.5560 - val_accuracy: 0.5302
Epoch 19/20
56/56 [==============================] - 18s 331ms/step - loss: 0.2569 - accuracy: 0.8996 - val_loss: 2.5970 - val_accuracy: 0.4810
Epoch 20/20
56/56 [==============================] - 18s 322ms/step - loss: 0.2432 - accuracy: 0.8984 - val_loss: 2.8290 - val_accuracy: 0.5213

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

**As we can see the training accuracy increases with each epochs and validation accuracy but is not increasing with each epochs. So,it can be observed that training accuracy is high compared to validation accuracy which means that the model has learned the training data and not generalised it.So, It is a clear sign of overfitting. The Overfit model has very low accuracy on unseen data.
**
So,, we have to overcome overfitting of data and re- build the model

Techniques to overcome overfitting are

1.Data Augumentation 
2.Dropout Regularization 
Hence, we will try to implement the same in the above model.

def load_data_raw (path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 9)
    
    return files, targets

train_filenames, train_targets = load_data_raw(data_dir_train1)
from collections import Counter
filenames_trimmed = [filename.split('\\')[-2] for filename in train_filenames]
classes_count = Counter(filenames_trimmed)

# Plot the classes
plt.bar(classes_count.keys(), classes_count.values())
plt.xticks(rotation = 90)
plt.show()

total = sum(classes_count.values())
for key,value in classes_count.items():
    classes_count[key] = (value/total)
print(pd.DataFrame(classes_count.items(),columns=['Label', 'Percentage']))
plt.bar(classes_count.keys(), classes_count.values())
plt.xticks(rotation = 90)
plt.show()
                        Label  Percentage
0                    melanoma    0.195623
1              dermatofibroma    0.042430
2           actinic keratosis    0.050916
3                       nevus    0.159446
4             vascular lesion    0.062081
5  pigmented benign keratosis    0.206342
6        basal cell carcinoma    0.167932
7     squamous cell carcinoma    0.080840
8        seborrheic keratosis    0.034390

pip install Augmentor

Collecting Augmentor

  Downloading Augmentor-0.2.10-py2.py3-none-any.whl (38 kB)
Requirement already satisfied: future>=0.16.0 in c:\users\sairam\anaconda3\lib\site-packages (from Augmentor) (0.18.2)
Requirement already satisfied: Pillow>=5.2.0 in c:\users\sairam\anaconda3\lib\site-packages (from Augmentor) (8.4.0)
Requirement already satisfied: tqdm>=4.9.0 in c:\users\sairam\anaconda3\lib\site-packages (from Augmentor) (4.62.3)
Requirement already satisfied: numpy>=1.11.0 in c:\users\sairam\anaconda3\lib\site-packages (from Augmentor) (1.20.3)
Requirement already satisfied: colorama in c:\users\sairam\anaconda3\lib\site-packages (from tqdm>=4.9.0->Augmentor) (0.4.4)
Installing collected packages: Augmentor
Successfully installed Augmentor-0.2.10
Note: you may need to restart the kernel to use updated packages.

import Augmentor

path_to_training_dataset= r"C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train"

for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset+str("/") + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)
    
Initialised with 114 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/actinic keratosis\output.
Processing <PIL.Image.Image image mode=RGB size=600x450 at 0x1C28BA606A0>: 100%|█| 500/500 [00:03<00:00, 143.02 Samples
Initialised with 376 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/basal cell carcinoma\output.
Processing <PIL.Image.Image image mode=RGB size=600x450 at 0x1C28882D280>: 100%|█| 500/500 [00:03<00:00, 148.63 Samples
Initialised with 95 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/dermatofibroma\output.
Processing <PIL.Image.Image image mode=RGB size=600x450 at 0x1C2888816A0>: 100%|█| 500/500 [00:03<00:00, 141.40 Samples
Initialised with 438 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/melanoma\output.
Processing <PIL.Image.Image image mode=RGB size=1024x768 at 0x1C280545B50>: 100%|█| 500/500 [00:19<00:00, 25.66 Samples
Initialised with 357 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/nevus\output.
Processing <PIL.Image.Image image mode=RGB size=767x576 at 0x1C2887EA2B0>: 100%|█| 500/500 [00:14<00:00, 34.54 Samples/
Initialised with 462 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/pigmented benign keratosis\output.
Processing <PIL.Image.Image image mode=RGB size=600x450 at 0x1C28054BDC0>: 100%|█| 500/500 [00:03<00:00, 152.25 Samples
Initialised with 77 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/seborrheic keratosis\output.
Processing <PIL.Image.Image image mode=RGB size=1024x768 at 0x1C2802958B0>: 100%|█| 500/500 [00:08<00:00, 60.92 Samples
Initialised with 181 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/squamous cell carcinoma\output.
Processing <PIL.Image.Image image mode=RGB size=600x450 at 0x1C288807B80>: 100%|█| 500/500 [00:03<00:00, 152.30 Samples
Initialised with 139 image(s) found.
Output directory set to C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train/vascular lesion\output.
Processing <PIL.Image.Image image mode=RGB size=600x450 at 0x1C280551B20>: 100%|█| 500/500 [00:03<00:00, 154.18 Samples

image_count_train1 = len(list(data_dir_train1.glob('*/output/*.jpg')))
print(image_count_train1)

4500

from glob import *

path_list_new = [x for x in glob(os.path.join(data_dir_train1, '*','output', '*.jpg'))]
path_list_new

['C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_00c7175f-da92-4676-8a2d-8bdf08cc06f1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_19411b0d-e52e-4294-b794-6acfb35e94c7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_233ec99e-6ab5-49c3-bb07-c4615f0ff50c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_473fa1eb-061e-45ee-a5df-59ca4be38182.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_4f26fa3a-0846-4d26-ade3-2106e85d324d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_57ed6d77-1e58-46a0-bebc-5111174f3ee5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_770d115a-8007-4c5c-95e7-4b2aacb8acb4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025780.jpg_d2340982-8e9d-472a-93c4-3e86570988a6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025803.jpg_02e62454-5a70-484f-bc61-98718022b77b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025803.jpg_1d4a38b8-6a6d-4373-85db-51c7ca922c6d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025803.jpg_37c20596-091c-4f09-babe-4ecef7fd2bec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025803.jpg_93477f6c-7fa7-4072-9fb2-79ca756d9617.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025803.jpg_d9a65a57-b38b-41cc-a7d8-65d13d1a188b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025825.jpg_06bb3f5c-8304-40d4-8806-8f66ce651f31.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025825.jpg_313386f2-8afd-403d-a85a-abfee108ed9a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025825.jpg_6f148b6c-2476-4868-a36b-930930947104.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025825.jpg_a7878c30-7f75-45a0-be55-f45b8c88fb42.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025825.jpg_f6d9172f-ccf7-46c3-a7cb-f75827ae1c79.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025825.jpg_fa186067-2466-4d1f-abea-2f743014a9df.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025953.jpg_33418cf2-15b4-4a0c-aea7-a6edba1b6c25.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025953.jpg_6ad42b74-9d7a-413e-b7bb-c955899abd33.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025953.jpg_ad2bc5ef-0f2c-4831-9b51-3e8c7fc0e5d6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025953.jpg_ba131861-24da-4b7b-b6c4-19ca7f9851d3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025953.jpg_d2766ecc-40d9-4e81-953d-f2c6fce1f6f8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025953.jpg_e7cff94c-37e0-40ca-943e-ed1887de4442.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025957.jpg_5ce4f404-c734-4739-aed8-82efbc3de481.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025957.jpg_ac4584c5-d0f5-453a-87b2-452f684a317e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025957.jpg_d6e55dd4-8178-4c02-9c98-96d9b68201ad.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025957.jpg_e8e7739d-3ff7-4a1b-91fd-32ea354a3cad.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025992.jpg_6a75d35b-8cbc-4913-a3fc-177134f46c2a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025992.jpg_86b77e8f-7464-4ac8-8b75-7fafab72f0c2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025992.jpg_9e530dc0-a327-4152-abce-40166e1c154d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025992.jpg_f1d69465-dafa-4199-99a4-2b2e2774bdce.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0025992.jpg_fb7e8c3d-47da-432b-b824-991bb15a297b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026040.jpg_73beea95-4eab-4bbf-905f-fe90d5673660.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026040.jpg_8ad3fcf4-e662-4bfa-92f1-82fd26660a31.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026040.jpg_9731414c-e93b-47c0-b14b-73dfc4632c2d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026149.jpg_0e9ab753-43f0-4585-beb7-6c8bd4e55583.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026149.jpg_336e28d6-30b6-4d46-892b-5021bac26517.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026149.jpg_34057ea1-1c04-464c-abc3-f8ed192f82e6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026149.jpg_7b309ccd-789f-47e2-8998-4a41f0848319.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026149.jpg_f2ef2aa2-9d16-4970-a9fa-ce0ef43f9c13.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026171.jpg_606f35a4-1721-447e-8840-93440f58f128.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026171.jpg_660e04fd-b7e0-4a96-b06c-9fdf89b68952.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026171.jpg_83ca961f-cfb7-4214-b25d-874aee72f289.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026171.jpg_f1513743-84fe-419e-9bbe-d39d0a566e38.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026194.jpg_4c26c92b-4f56-4d33-bb1a-c1ea8508e76c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026212.jpg_672fec15-b1a7-472e-8eca-7779205153cc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026212.jpg_ce306eff-d822-4f9e-9e2e-61cdc9ac6721.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026212.jpg_f05a6bf8-b340-414e-baa2-856d56fa0873.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026457.jpg_0541d970-2aab-4cf6-8e9c-7ea1e95cad91.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026457.jpg_53757ef0-0f59-4515-b94c-25d23fbf74b1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026457.jpg_553b82fc-436e-4c9e-b4df-10848aba6817.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026457.jpg_5b064a75-dac5-4d41-91f7-4a1016c5f25c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026457.jpg_6d1dca43-1b40-4a3d-83bf-6caebac52810.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026468.jpg_45273691-9854-4c19-bec1-e039c83a25df.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026468.jpg_852d14e9-36b1-4bd9-8b21-6bf274f21549.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026468.jpg_9211affc-27d1-4feb-b2a9-f2fb4b9602f4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026468.jpg_c3732b70-908e-492c-9521-1084cc22b00f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026525.jpg_7e187b4f-c32e-4caf-9aba-bc6f0bb144bc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026525.jpg_b0ef78c9-a2a0-40b7-a39a-f616c36d6613.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026525.jpg_b3f9e689-ba65-45bb-9788-5686764e4921.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026525.jpg_da105c6c-71f4-4390-9042-5fea4618d632.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_135686c3-865e-4722-b4fb-db955a99f747.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_2b65a2d2-0b9c-49f7-81f6-b3bd73b6ba52.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_54cfa67c-87ba-4f73-af0d-6c3bd431f2eb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_b0226af3-ffa2-4f24-8601-224cb65d491b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_b7896b3b-67a7-4462-9857-4c67bea3dc7a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_d4a51778-598c-47aa-9d90-ea1915b07b9f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_da36beba-771e-4008-8fe4-78c325fc9263.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026575.jpg_db804ab8-91aa-4507-967b-da03f6b7c8d4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026625.jpg_04569d8b-deaa-4175-bf16-2934c1abf4a4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026625.jpg_5f9e07db-52da-4f63-99c7-e61459052e8c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026625.jpg_67fd99a7-b607-41a7-b6bf-603d94672254.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026625.jpg_a3e9c19c-5f78-485d-ac46-1e801277a697.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026625.jpg_e5931af8-1eff-4186-8671-17216d6dc796.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026626.jpg_0050412f-deef-499a-9fe8-515285ccd3c6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026626.jpg_8dc04050-6a44-4056-9d69-eabba8a3fcf9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026626.jpg_f9f7c5c2-92d9-47b7-9aee-19d5e67d70b0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026650.jpg_2d9fe31e-1422-48df-b2a0-1ecf52c1b17c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026650.jpg_5541fe34-497a-4600-aa50-1f8a525fec71.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026650.jpg_64ec0213-f042-4fdc-aca0-879682fbbaad.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026650.jpg_cbe73f78-1bbc-4f4a-9099-40aef858418a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026650.jpg_d8a8f122-ffd8-4ecd-b1e5-a89caeb2a82f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026702.jpg_5fc93427-3587-4f0c-a787-59be67e04f55.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026702.jpg_8c4fdd22-3a07-47c7-9ea2-d93192103b05.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026702.jpg_cf5dd72b-1c0f-426e-8371-41de3261ffd9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026702.jpg_de94ad6e-db4d-4a93-bf71-13f1140a220c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026702.jpg_e3705d19-db01-494d-baa1-e313b748d48d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026709.jpg_4bfe2f13-4e64-4e9e-b32b-855ca45d5349.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026709.jpg_50b1c4a9-fe93-4309-91d5-a473cfe5b065.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026709.jpg_a25f164e-74c4-48e4-a9db-9b43e9676422.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026729.jpg_10ad92d6-0a71-4606-b4e4-a6708fbe6180.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026729.jpg_12daf1d6-18a2-4cd8-993d-7da2fb0173f6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026729.jpg_a2b9392e-9398-498b-8641-ce92e72b990f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026729.jpg_e9620a82-7e31-45e5-88b1-4fbd5ea6fb4c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026765.jpg_2c170c1e-8075-401a-a6f3-e66ba8d76dbc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026765.jpg_3b04f81b-09e4-4995-8370-74df78e4fa4a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026765.jpg_982360bd-8bed-43a7-9aef-1bba7fa6d323.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026765.jpg_b36f715d-9ac5-42cf-a841-911a2d05c24e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026765.jpg_e397fd2c-0653-4bb2-8ee0-54e79c182453.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026848.jpg_05923cb2-9724-4682-8ba7-2e2bc3d23927.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026848.jpg_333a3e0f-220f-4978-b625-c1f60851875f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026848.jpg_3e04ed51-6b33-459c-a156-0469bf7786f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026848.jpg_607ec9db-6211-4728-a036-337e00d1a089.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026848.jpg_ba466b41-15b5-453b-b40f-74f23eeb8e73.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026857.jpg_163aa855-e25a-4a84-848d-b3ea11281eea.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026857.jpg_6591b995-687b-4ae9-bc49-504d0bdc862e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026857.jpg_6a4a77d3-1346-4dbb-8375-691e7efce899.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026857.jpg_c82b19d7-fffa-4ecb-ba5a-a00a740da355.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_10bab5ec-acdc-4ef0-981a-233121dce9eb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_18068d15-c034-45d8-b804-1ea650e8e61f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_2be86b4e-07af-42fc-98b6-d73611ecc8f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_87b28a66-7700-460f-b442-816996868a68.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_ab1155e1-af25-44e4-a777-e4a6094046f2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_c5d4f85f-f07d-4353-89c6-6ec80924a369.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_dea99c82-ca73-4fc8-8135-1042edd6c8bd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026905.jpg_deb2fe33-849a-419b-b331-3363113b5bc3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026984.jpg_120cb784-0cbd-406e-b40b-35e04ebf55ed.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026984.jpg_82b7c4f1-7193-4486-b5aa-ccfe39c83093.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026984.jpg_c57aa128-f3f4-45d4-87ed-31bce0c96ed9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0026984.jpg_e7df7e87-99b3-4339-b2b6-33dcb933b58c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027172.jpg_020e6804-2cc2-4026-9a34-965521f1b938.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027172.jpg_232fdf04-2fe4-4a3a-bb43-96e21fc63b55.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027172.jpg_a3a5b99b-2361-4666-9b11-d14327d5f720.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027172.jpg_d0b39b47-b76d-4826-8c5e-664ee81a9dc6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027172.jpg_dc274793-0d78-49ea-aa0f-37f0ee6e9775.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_3d1f0a15-e7e0-4af1-afac-5c683be779de.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_3e05cf37-3261-40dc-bf46-3f5bf4c10d65.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_638b759d-ad94-4823-b043-eb5302ab797c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_b2121e4b-5e7a-425a-bae1-1ec5938ee48b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_b974b865-353d-41a8-b941-a9c979000ad0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_e4c257b0-ceac-46f9-8094-1f113a48a1e3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027254.jpg_fec59b71-1fb4-4c4d-a464-9d3e11298de2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027334.jpg_26b71009-9fba-4e89-9e06-70feda21484a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027334.jpg_3257a83f-5ffd-4c67-b03d-6680d81727fc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027334.jpg_5a682d19-7c8a-4558-901b-a70737ae8727.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027334.jpg_6c198fd4-db8c-463c-805c-84afba92085a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027334.jpg_93e61d67-7d8d-4a66-b904-3e717d5d0cb3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027334.jpg_94639897-cea4-4b2c-a0b6-c0eb25a9a1e9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027447.jpg_042d470a-5578-4a18-9f1a-7a6ab6d58f03.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027447.jpg_e2f36a8a-fc56-409c-966b-d33e7562985c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027447.jpg_f68827d2-357a-40ae-a07d-eef9cf7c9ace.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027447.jpg_ff6a5bcd-1115-462e-b269-3f2cdcfb54bc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_01446eb1-f36b-42b2-8b20-fd266f5e1057.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_0820d3fd-1e50-4807-99a6-68b8ce1d98e1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_138188a5-f01e-4d16-af23-bae3ce2fec1a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_1b6443ac-789b-44d4-8e13-4cc441f7ac15.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_1bcb01c7-b04a-45f8-8e38-bbad8c6a5e67.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_5142a2eb-3fb7-4b38-8c85-77ff521d2371.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_5936c203-83c6-4b2d-a063-57a911cb4fab.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_639254b2-9c7f-4f1f-b451-d583f1af14af.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027452.jpg_ece20f4a-21c1-4f0d-9f5c-af2418474927.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_264272b4-3726-4164-9911-aff2d732169b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_30d86789-6854-4be7-8ebc-27ad6e7f7bf0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_567e11be-2088-4a37-8bf9-d8aea5895ca0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_6ddf5877-4ced-4517-a8b1-21365288a7c2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_77de66ec-b79a-4d9b-a39b-15801d8c9e90.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_cb3bcba8-3a16-4b7a-9ea4-6f7000927673.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027536.jpg_f2e0e03b-839e-48c4-bab5-ebc5c9496e3c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027562.jpg_3dfebd1a-097c-4450-8afa-e64253f666ec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027562.jpg_8e7c71f2-22da-4ba3-8223-46a9020f5f3c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027562.jpg_986b5d33-b16c-4a35-b291-82e07a4c14e3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027562.jpg_d49f2771-a17f-4366-9b2c-c505055c373f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027580.jpg_5623446b-c438-437b-b176-0eed02b3543e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027580.jpg_6d0567bd-6630-4bfb-9aee-3cb7cca7a817.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027580.jpg_a920bd20-376e-45c0-8926-5a588721ef28.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027580.jpg_c6d97f48-c7fa-4bc5-a2fd-b742849c21f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027580.jpg_ef1d39b0-4362-4672-a1ca-7a182e5c4489.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_33dd7f2e-1c1d-4e02-97e1-a4018bb9c2de.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_4cfd1164-31ba-4081-ae77-6be40f29b09c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_8fc631f5-fcb2-471b-89e2-5ef6ff091364.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_9f3f341d-7cde-4d4e-91ce-466337315b58.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_b457aa02-070d-4eb8-8213-7d9676b10224.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_d5151ec6-8e70-4ed4-b82f-0dd572d66303.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027615.jpg_e34f2fc9-414a-47ef-a909-98eb874a1e24.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027650.jpg_2a9c5604-a981-4552-bb99-874e0b577756.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027650.jpg_5f1a1d64-dc17-47e2-b539-80b63e0f9f04.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027650.jpg_7aa5aa4d-68c8-4546-8282-507aaba66290.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027668.jpg_0a4509e5-a845-4785-9fe0-4b196d7d260f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027668.jpg_204aca19-bbd3-468a-bfee-24c5b71c36e9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027668.jpg_8bebdcd7-ab5d-4094-9139-1dde15408c44.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027668.jpg_904a73d7-65a2-4e56-a9e7-2e5755960ab3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027668.jpg_b3b38c4a-2d61-4bb7-a0d9-bc2e2d09d5fa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027802.jpg_44feae1b-a9f1-4fa9-8a77-93d1c51a9d20.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027802.jpg_93afa07d-52ac-426f-86e6-826cbf4a4aad.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027802.jpg_a0852969-2754-4006-b58f-e1ceaae79efb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027802.jpg_b7622a8a-7447-4eaa-b921-d337afce7fbe.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027829.jpg_20232f77-7c75-4c07-a653-9d84d318b22f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027829.jpg_2591f629-093f-49ef-9648-263f1390a033.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027829.jpg_6f9595ed-7b7d-4986-8e31-dfce2104619d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027829.jpg_856691cf-546c-4810-9720-da0257d86e88.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027829.jpg_98f5ba0f-1c84-4eea-89aa-f03b1f639c23.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027829.jpg_ea38376e-df98-4a98-b65f-bb4f4f3213cc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027884.jpg_5ae358ca-07af-420c-9d5c-3afd4c44e1b7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027884.jpg_73e233da-b9d3-475d-ad03-432eeadf8b1c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027884.jpg_a38a1c4c-74a8-40a2-95e3-6be4c69c5d64.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027884.jpg_ce0db1aa-6cd8-4764-bd8b-b0a6a25d09f3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027884.jpg_f343883a-da29-4148-9f2c-f1e99d54bb01.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027896.jpg_dad135e8-f3c8-4d7b-9805-d2215ac41c8f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027896.jpg_f0e40d34-77e1-4e05-9387-a69649b025ca.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027896.jpg_fb1f17cd-4fbf-4679-8882-530dfcdd54db.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027950.jpg_26358275-d55e-46c2-9103-99c65905d226.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027950.jpg_4cf46388-52ae-46af-988a-2afbf4ce42e0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027950.jpg_55f141fa-6fdd-4e07-b466-7f852c33cab3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027950.jpg_af55c5ad-0244-4727-8bd4-ccd92d11a181.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027950.jpg_c8b51da1-174a-4812-b318-31936f4ef69e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_03701485-16f2-404e-9625-bea11fdb3c1f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_0a05c286-fccf-4fa0-95e7-5f545f5eb2f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_0f75cbfa-7654-497b-bbab-a53f99a54b47.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_0feaf4c8-a515-486e-8a76-85d94b653099.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_13c2653b-6172-4739-8cf1-19556d10b8ec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_1ff760f6-b89a-4e1a-98ef-68a4f4aab93f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_319d4ec2-93d1-49e1-aa7d-b3c0e7a1b051.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_6919f7b1-29b0-4193-8f81-38750c505c44.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_703e3b55-e81f-4c70-b78c-18af09b5663b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_ce935562-c045-40fb-aec0-a2c5d74faa27.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_d6d609b2-4ae5-43ab-af03-de87ad15fd8b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0027958.jpg_e7173900-3e0e-4781-a7c7-3b7cbde6d469.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028063.jpg_092c357a-1f82-436a-a2c7-8d3d6c7f35a9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028063.jpg_2a19b886-e9dc-4095-8a20-20b49608d381.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028063.jpg_441bfeca-895d-4044-86c0-9d6a232915db.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028063.jpg_d1c73e35-dad2-4cab-ba03-0abe8ea5d766.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028063.jpg_d5aa5707-a862-4399-b091-e3baacbf6bae.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028063.jpg_f02728c7-d916-4351-8d9c-bd7df7d65f9d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028076.jpg_015a05f8-b3c6-4308-a8ec-26147d2d84e5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028076.jpg_2f32a294-42dd-402f-97b4-5102f2cf11ea.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028076.jpg_9b7ce2e0-9cb8-49eb-9b95-950eef951d25.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028076.jpg_a9643135-aa76-47e4-bccd-1ffd4489aefc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028076.jpg_b2df87f9-3f86-4fe8-ae8c-261f66125e08.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028076.jpg_b2e2192a-6752-4a06-bcd4-65563a6efc81.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028190.jpg_af77b2bf-ffc1-48d4-a978-f7102acbceae.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028314.jpg_c729163d-080e-4531-80b0-88a8e29f32a4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028314.jpg_c7580e18-224c-4c10-bf45-b6a74ee3483c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028314.jpg_ccff7687-006c-4b34-990b-9433f3b8d22a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028370.jpg_697b2cdc-f954-47e9-ab25-140643e2b082.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028370.jpg_77408d6e-5246-49cb-b7ef-a1b463b92ddc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028370.jpg_a512ed8e-c2d2-4b95-8213-618a754a365d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028370.jpg_bc89e19b-304a-4314-a3f4-ee90d38df36c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028370.jpg_efbf388a-fbd4-480e-a3c5-ec428c5ddb2a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_21fd69bc-5a8e-4274-9755-7f38957dd8f7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_2b6e932f-cdf4-4f5f-816a-dc4eb93d9b48.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_3e33db5a-2867-4f56-9510-b688ee6d4a38.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_7029f00f-f1b1-4abb-99f1-446acb50ae3f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_735d966e-cd14-4fcd-83a2-8466f00432af.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_af125655-d913-4cd6-9b79-c2619f876838.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_fb12a0e2-73f2-413c-a65b-b56aee0269e4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028393.jpg_fefa2a29-5d76-42e0-b606-2c59a89e538c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_14cce45d-09d8-463f-9251-582272b3a576.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_2dd6b746-3186-4cd6-ad1b-fa9ef76d8e91.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_5e49e4f8-7668-409b-803f-676d77ddd710.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_6c7fbe84-fa91-49da-9fbd-aab264212788.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_95b5e752-3a4f-4f1e-a387-2d0465de31cb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_a85cc16f-11b0-40ea-adbd-78e787ff8344.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028517.jpg_bd8624d0-6150-4082-b885-5943b5c0e737.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028558.jpg_2138da35-3a55-4363-9e17-cffb17181e00.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028558.jpg_a56751d8-c812-40fb-a239-484b7d706415.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028558.jpg_c0220683-c38a-4683-8808-aab3a2e8aec7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028558.jpg_c5c79d7e-6c09-4acd-bc30-4d31c4994d83.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028820.jpg_32844bc1-c137-42b7-826a-6ebfe4b568ad.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028820.jpg_44c2d134-3a4c-40d5-b790-362f4efd4a35.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028820.jpg_572b0c67-8357-4a9e-a02f-19ff4701f13e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028820.jpg_8529c66e-a5fd-4a6b-944f-ff4cfdef008c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028820.jpg_d60d2a1f-538e-44db-ac68-c41b6c1e1d89.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028854.jpg_a1e07b87-6474-4849-9e32-10a3ec85c77e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028941.jpg_348ccfdd-de2c-4a9e-9139-69c5a9fc9aed.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028941.jpg_8053af74-4cdb-4eb4-b088-556e6dae0b92.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028941.jpg_8b1cd8e6-0eff-42c8-a22e-1607eb33b6e6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028990.jpg_0ed0eacb-6418-483f-8518-ecb9e8891e69.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028990.jpg_665a0789-a6a1-440c-ac54-c5c620b6c4ec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028990.jpg_67c15825-7463-4493-ad5d-414baa234dbe.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028990.jpg_73e6a7fb-b187-4361-a4a8-11f033d7bea0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028990.jpg_db5a573f-76b9-4773-a709-c1cd20a1244d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0028990.jpg_ea18252f-cf89-440b-90a7-8e877e274d01.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029025.jpg_0ee571b1-db91-4b64-bdf3-13650050eb68.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029025.jpg_38ab6938-9432-49fe-a2f7-1f274d4f2f63.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029133.jpg_18f637fd-9aa6-47af-acaf-a29d7e7cf097.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029133.jpg_b277fae6-5319-4df8-87f3-9e34352b9320.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029133.jpg_bd0cf4d8-dfc9-4da8-875b-8429d6e8727d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029141.jpg_1f3a0d71-8f78-4193-beaa-570a3f2bd9f2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029141.jpg_22b03836-a11d-4344-9fa3-847fc7cbaf03.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029141.jpg_326adf37-7bc5-4324-aa87-770186269acf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029141.jpg_6fb4e0d8-a41a-49c1-b80a-2df41bdfc0b6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029141.jpg_907d6212-9916-4405-9ec7-8a41cc6bee01.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029210.jpg_b63527e4-f9e9-480d-a135-54e613ca4e04.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029309.jpg_6474e37f-6d4b-492e-b4ea-58558b220258.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029309.jpg_b17c7219-d309-4c4b-ab3f-b6b76a374a89.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029309.jpg_d2830f81-9bf8-46d9-8399-3f8277a0b263.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029460.jpg_242d4f4c-d3da-42f1-af82-8d307f22adbb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029460.jpg_64ef7318-bb77-468e-a29b-554287dcdcee.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029460.jpg_817ac0b0-b50a-4365-a784-6afceb02240a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029500.jpg_31493f5b-9f33-42e2-922c-1e090de53b0d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029500.jpg_39c159ff-daed-4018-92b6-26bd02a416e4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029500.jpg_5da2fe05-9817-43eb-ac3b-195f8b81fc96.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029500.jpg_f815d030-a35b-4f03-94da-4894a09f12ce.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029659.jpg_179d41fc-9646-4051-a871-a5c806a95cde.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029659.jpg_291774cd-cead-4e25-a58d-249f4943d46d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029659.jpg_57bfeee1-c1cf-4c4d-8192-de93e8a0f6d7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029713.jpg_15c8766b-9a66-472c-8d7d-6ed984e55bba.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029713.jpg_30654eb4-c958-4d00-8bfc-7afa933c7ef6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029713.jpg_c948eec7-870b-4879-8bee-607ca27767c8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029781.jpg_9abe7464-e3c9-453a-be26-69fad01ca51e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029827.jpg_1a2d685f-bced-4018-9920-59551edddbda.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029827.jpg_7e52f063-d47e-4ec9-bea4-2d7fc1ac3954.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029827.jpg_913d5fd1-7b62-49f2-96a2-f2b1d7f65887.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029827.jpg_f7c2bd23-091b-488c-b96a-ded117823ce3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029830.jpg_ad549bbe-f5b4-445d-b567-cf2e5ac07f62.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029830.jpg_add71105-915f-426d-a3f0-ce2b99f8b6e0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029830.jpg_eceb5720-1873-45e2-8daa-6143633ab6ae.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_2ac13dcb-9025-46ae-97ec-b389bd939910.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_52c71431-ca97-484c-b508-f5b273f95e3b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_74a64a80-741f-4d40-a617-b808bdd0d9f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_bfa06059-5be5-49c8-a631-15e1c1f92e8f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_d4d0c76a-4f47-40e7-8af8-6b1c4f6a0281.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_d5f33f26-49e7-4566-ab8d-e4411752c1e3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029840.jpg_fb02d34e-75e5-4f68-be88-d0a042a7b171.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029900.jpg_3f7e018d-1cc5-45ff-9617-c7f6f54133f6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029900.jpg_875daa11-826b-4747-b525-17f9197a35bc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029900.jpg_b95e081b-ae86-4af6-80af-f6aed0b23990.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029900.jpg_ce273c3d-de2d-4879-917c-0bcae974e528.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029900.jpg_d791f498-c826-4f64-9b5e-7539215b696f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029915.jpg_5bfc1daa-3064-44fd-945d-8614a410aae4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029915.jpg_6efcfee8-055c-46c7-a5aa-72ab24c1161c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029915.jpg_81febac3-4385-4d42-ba8f-cd98e4c932fa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029915.jpg_d4940c04-d044-4281-b016-1b65edb1c591.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029915.jpg_e9992122-641e-4f4c-b25c-d1c4a3ef6858.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029930.jpg_0b344a6d-d002-42c4-b0d7-208ce0565d98.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029930.jpg_246dd111-8cfb-4d58-ac35-6f6e1a2b2901.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029930.jpg_72005d10-2f00-49be-9287-23e817f9cd3d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0029930.jpg_da05093f-8b09-464d-9cfc-2a2ed8786fb0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030036.jpg_0b7821fa-4efd-45fa-8763-644148c3cbaf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030036.jpg_0c6137d8-fa1b-4374-ba42-514a2dd372d6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030036.jpg_20565416-5a8b-4ec8-898d-5ba4ed8bd080.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030036.jpg_4a63e5a6-4f2b-4da7-9381-80485cd825d8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030036.jpg_90da802c-a01c-44bf-a157-0889d57b229f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030036.jpg_da300e60-0c21-4714-abac-1802abfcb90d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_2ebd239f-5909-457e-8da1-f9731e852ba3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_3119f99a-4d62-497f-a064-5af7bf8ab484.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_33ec29a2-a820-4889-a2e1-5323a0c77370.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_45d08d80-d4a5-42ea-a3fb-984ede1c57a3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_6c9f4f97-b9ef-45e1-bd84-ed2adf8d901a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_6cea56b9-733f-45a6-b1ff-3addbba6c3c4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030133.jpg_85e2f515-54ca-4359-9e1e-1724966d8f7e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030142.jpg_33aabac9-9d83-4f61-ab7d-a62171d3011e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030142.jpg_4d20cbc4-0030-4a56-a401-e190b5a87841.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030142.jpg_59213438-7369-4d98-9f79-09cdb880ca2a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030142.jpg_86e8963b-51e8-4a8a-ab71-60b9ac8c9262.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030142.jpg_d99d10e9-76fe-4003-ad86-fd97a4b00985.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030242.jpg_9a5a555b-8b54-49e7-aeff-88167b1bfd36.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030242.jpg_cace736c-a047-4547-86b7-10284034190e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030242.jpg_d1c0a700-ebb0-49c2-87f9-befa1b77ae7f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_280c415d-83e5-4f86-a719-d75d37d4b3f9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_39222c14-a717-4025-9274-26891da81161.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_55dbc492-dbc1-4ce8-85bb-b7e6dfdc6a8f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_a078fcc7-03d5-44f9-80dc-a092f638838f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_ae9c2b4c-8409-4876-9213-c9944ad2e110.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_b6387fb2-cb89-43eb-b076-386c33d0e82e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030344.jpg_f3e5053e-42e7-42b9-93b3-4fd4c66049ca.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_2d8c6c69-9cbc-4f3b-8e42-5e264654515e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_4b0a5eb5-d21e-41c6-a67c-7454c29eac9a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_4da224e5-8740-4d9c-ad76-f37d3e18ae98.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_79a843ca-0a30-4ee6-ad1f-76fdbdcf810a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_9b3ca18a-ea22-41a9-a185-61a92c95d882.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_9b69eb2f-7116-4a5b-bfb6-fc6fa092d532.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_f337c6b9-95e1-4537-b3eb-c2a94d84b016.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030408.jpg_fd195a4d-8e86-4a28-b175-7be0659b642c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030463.jpg_78e22380-3bc1-4e60-99ac-06e4a3239c6a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030463.jpg_d6b58ad5-a912-4952-a3fd-a2173d5eed25.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030491.jpg_0dba15e4-5397-4b7c-969a-9cd4a17c80af.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030491.jpg_2b696e8b-7655-4cef-9879-c35ac2df0f73.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030491.jpg_4fd0d4f6-1412-47b9-bc31-98cd5a90f233.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030491.jpg_93122915-852d-4e15-8807-76b0884fe0ab.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030586.jpg_39ca8c28-43f8-47ff-b5a7-483fe10fef4e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030586.jpg_bc382ec7-e96d-4d5e-a35a-bf995ff382e1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030586.jpg_daa2677e-2c4d-47d9-bb42-ca664b4b0766.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030655.jpg_0833e7e2-946d-4196-8a57-ba2ec53a2442.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030655.jpg_1abfbba2-7c5d-46b1-9492-8f1605465ea0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030655.jpg_cc6326e6-2c48-45c1-8e04-ff3a4c189952.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030655.jpg_e59e2b5c-bd96-4c96-aae8-a31d0156e888.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030655.jpg_f1083c80-1861-41c0-8922-f28ce765af72.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030730.jpg_0e0bc1b6-360f-4768-965f-7c0cd8269f44.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030730.jpg_0fabe9fd-ba71-45e3-8ef1-327779084baf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030730.jpg_ed743b4e-a793-4057-805d-8d05a9fee39c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_36fcb344-31f4-406f-90c1-94733befa513.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_38cf932b-5c0a-4483-b998-75c0f5841da1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_408e84e3-3495-47f7-aed7-15c14dfc84db.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_72f3d022-8043-4945-aa7d-dd204c9f59a8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_b32a2cb8-7dae-4754-a562-79c4ad361133.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_c3227799-ce46-439b-8f45-4504286f3428.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030825.jpg_fe3d9b08-1c38-4f46-9f52-9aa2ac77997a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030826.jpg_10308067-4511-41a8-bb61-da3c135d20ae.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030826.jpg_b342eb11-fc3d-4e3e-a48c-f561900ffd34.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030877.jpg_9f90eecd-2ee4-4f6d-af09-7b7f56b2a8a1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030877.jpg_9fb727f3-5339-4cf3-9f19-5b108ee042cc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030877.jpg_cc228655-8b94-40c5-a9a2-d928c3868d68.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030877.jpg_d466671f-1757-48a6-91cb-7efef12073c4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0030877.jpg_e561272f-ab34-412f-9439-ffd15aafc6d8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031040.jpg_35a0515a-4f93-42c5-b640-41decb79aa90.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031040.jpg_75ab21fd-f8ec-47e2-b79c-618b4f0dd711.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031040.jpg_7fe44655-6704-4a0d-95f4-b34c0d6059b1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031040.jpg_f04bdd2f-ebec-4448-8385-3d5707178414.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031108.jpg_2ce6653f-c0c9-4ba8-b3b7-2588a03adfd5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031108.jpg_8a66cfe8-0be7-4ac9-b8be-6420317dc06e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031108.jpg_8bf22e3c-c2f7-49a8-9dfa-14a5b9ca2b76.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031108.jpg_a1f909f8-9ef1-4112-8ae9-d12811a8b72a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031108.jpg_fa7090c8-55cd-4958-9f2a-b24f1919d575.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031228.jpg_68e01908-42a7-4c51-adb4-83e9e5ea41f2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031228.jpg_9ae025f5-9a6d-4de1-ab0f-2fba0a88bdaa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031228.jpg_f69c3b72-1576-4d66-92c6-53bd57a4c5ea.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031292.jpg_8fb29893-b752-4326-b537-7f379ac8f473.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031292.jpg_a5d9bb3c-5391-4723-be63-94d93148dca5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031292.jpg_ed25ef71-e179-4770-9a15-c373eeb9b296.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031335.jpg_58d4d2c2-fb9b-47a4-a2a2-a33173026f5c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031335.jpg_6f14dbe7-872a-4e71-845e-40b0e091e523.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031381.jpg_448a88d1-0fe9-4f12-a670-57f7a46d3a0d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031381.jpg_ac42797c-06e9-48a8-a9ff-b6998d035260.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031381.jpg_d271c0d7-f718-4ce3-a3ef-4bd057905985.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031430.jpg_d6b57c2d-3d26-44ea-8361-628df7a2294b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031430.jpg_feed8d3e-22cc-4c57-9944-87a4f3b47c99.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031506.jpg_45360ece-6242-489d-a83c-77db71bd6ff3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031506.jpg_4550b5b6-f2d1-4143-b084-76c1e807f34a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031506.jpg_a3f78d38-9f51-43b5-a715-7822edb16dcc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031506.jpg_e79b4348-6b12-4d55-a2c5-a76d1a602eb0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031609.jpg_4991d49c-9297-409c-a48e-5edeaf4f0e45.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031609.jpg_9e063050-8e19-4202-8021-cfb8639637fe.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031609.jpg_f8d4b6ba-431e-4f9f-9ddd-176a6c766c12.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031823.jpg_e4b57a96-5bc3-406a-b831-206590b9443f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031823.jpg_e6237f7d-367a-4392-8167-22f25bf6d525.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031922.jpg_283622bb-b37b-4a4a-8188-0489abb881b1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031922.jpg_9327c3cf-9a5d-402f-96d4-4cbdd9bad1b2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031922.jpg_da771fdb-4ab2-46fa-93c6-287784a5d363.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031922.jpg_dc4c48b4-1a11-416b-ad7a-124fc56d66ed.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031993.jpg_1da7cd87-6f54-4b58-8cad-e820e9a0b01d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0031993.jpg_b9f01356-95fa-4355-8236-60a2a4cca08e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032135.jpg_2d6ccf09-af4a-49bf-8e9e-f484e1215baf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032135.jpg_344423e0-ca94-4603-be88-35c16083ee3a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032135.jpg_ee453fbf-7586-44a5-8809-03cf78363ad7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032135.jpg_f8e767c8-ced1-4f65-8382-d5e1eaa26339.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032199.jpg_217e8c28-3a65-451b-a7a0-27316be9663b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032199.jpg_40e80080-17d9-4bc3-a57f-5baef5f232aa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032199.jpg_5ea92bcb-e92c-4b41-a8bb-06b3e6a7dfda.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032199.jpg_cc558a23-7426-45af-97c6-b81ea51425c7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032206.jpg_1e6fca24-8e74-4907-b8ed-9c60d7527ebc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032206.jpg_6d53662e-806c-4b83-ad2c-efde530278d8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032206.jpg_9ba38cf2-cf43-4bb2-a135-7cd1a00c7d9d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032206.jpg_ff463b62-4414-4386-8f19-8241334c780d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032404.jpg_5ba50533-eddd-4607-8652-e92c24e0f647.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032404.jpg_629751db-8cea-4b63-8985-3e32518b3039.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032404.jpg_6d2359a3-c599-4477-af61-c760d6aa142a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032404.jpg_c26ebfc2-052a-4f7c-a7e8-4f691a74dbb0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032404.jpg_c9cbe506-3adf-42c2-a7b5-d01bec33d8f7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032422.jpg_611954ce-e68b-4b4a-8a53-d261c3bbc304.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032422.jpg_6b1e4efa-e560-40ec-b9aa-b22a04f03773.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032422.jpg_6fbccc2c-98a9-4cf9-8d94-28d98a94eacd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032437.jpg_03d74161-79dc-4d6f-9a12-f39e1b46f59c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032437.jpg_2f4e611e-c95c-4bc5-852d-98885ace9cd0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032437.jpg_4bb0fc72-a7b0-4b82-9851-17af829c244f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032437.jpg_7e02c2ac-7659-4e93-8394-9e3a3b787203.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032437.jpg_a4abdfd6-2955-46c0-9fde-7e7e37834f4a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032437.jpg_bde959e5-cd36-4f01-9ebb-5b173f104933.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032854.jpg_86e2c823-dabf-4623-9bcc-6de68878869a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032854.jpg_96774f3f-3c3f-44ed-98aa-46d1b8026b7f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032854.jpg_a048bcd9-83e1-4d4d-9f2a-938838f9c5f9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0032854.jpg_c3e2ed4d-37b0-467d-ad91-54838e88ac02.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033151.jpg_9b3f7cb6-61a3-48a2-907c-20092d299349.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033151.jpg_9e84518d-7cc5-49ef-a1b7-be22c5691083.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033358.jpg_04a663a4-bf9b-4384-b094-832635bfad84.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033358.jpg_8566cf5c-0be1-44ac-a29b-0b918fbfd83f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033358.jpg_8ad67284-a122-468c-aaea-b99762bf4bdb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033358.jpg_c7d2106a-89ee-4413-9b02-d65ae97bfe4a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033413.jpg_500a00cf-1e53-49b8-b7c6-d7fc22fa4adc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033413.jpg_a8927550-d88f-480d-a3ef-1b1cb4ba1ce3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033413.jpg_b9d5786a-6b8e-4ea3-a02d-b2e722756db6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_0b8ca696-a580-4523-9dc2-c1a7f0bdb8e7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_0ec22894-6a91-4c18-94c0-b36b7c174ed2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_1b385e6b-5a7f-415b-9ef3-44ab87828520.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_4f5268e5-61e9-495f-973a-d64956308567.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_58b87167-35d7-42c0-b284-1fe442bc4914.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_adc98dcf-f515-4687-a016-cbde4f3f1af5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033456.jpg_be0924d6-51cc-4750-9fb0-a5d09abffcf9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033494.jpg_0b99a9ac-15db-4be1-84fa-c6ea1f335cdf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033494.jpg_31be9113-41b2-4e31-b269-f12afe0a88f3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033494.jpg_827cc4c7-932f-4a7d-9a02-b7a67b0e5bb3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033494.jpg_a450c7a4-5952-491f-8c56-cf93412f6eb0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033494.jpg_c13d011a-f358-4cf3-8d4c-8ea8dfded17f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033494.jpg_ce1f0a3c-1179-426d-9805-749a6c02333b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033705.jpg_0ec8ace5-e985-4b70-9c5d-a0e0e2a9f520.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033705.jpg_503d08e4-6e30-45f1-88cf-c36509c09d3b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033705.jpg_b5447e91-c7af-4efe-88aa-15a79b2cd12d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033705.jpg_bac0335f-21d0-40d1-8bf3-a61895587403.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033811.jpg_1532a7bc-40ad-4a18-b45d-160003ab7a3b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033811.jpg_33a3cff7-7802-4152-84cd-3e72b179d40c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033811.jpg_79dee218-5cfe-4b79-9c14-1c44bda17b06.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033811.jpg_83934284-817b-4652-b2b6-6bedb4840599.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033866.jpg_2bd81245-562c-4b8c-b121-b0f29f977953.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033866.jpg_3468d288-6719-4434-ac3f-b501c5066100.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033866.jpg_8e37cdbc-9913-4577-8bb4-c0c36d9da7a4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033866.jpg_be62fe6b-8f66-402d-a726-2ca0a1277411.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033866.jpg_fbb93ac6-90b2-424e-990a-6292b009c46a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\actinic keratosis\\output\\actinic keratosis_original_ISIC_0033866.jpg_fe9d94e5-03fc-498c-935b-0da00aae1ebd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024550.jpg_0cb8ad29-f964-4bd7-bda0-894e94b818f7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024564.jpg_b9b7874e-2145-408e-9fc2-32d0747b7d18.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024572.jpg_2b8c68ae-9714-456d-a25f-b9f95c732ff9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024573.jpg_97434e6b-31d4-4fa9-a681-3841f2ade3de.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024582.jpg_5db4e686-24fe-4296-aacd-4bf7e30131b7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024582.jpg_a67f51bc-90fb-4e4d-b8f2-2718c806e9f4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024590.jpg_f991daed-e8d0-4274-9d14-c7779a6530ee.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024595.jpg_0c9d3348-9666-46e3-ab61-f09895ad361d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024634.jpg_24806836-743e-43df-9fee-dc30289b9bdd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024665.jpg_89c712a9-390b-4891-bac8-86df2814a214.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024666.jpg_db47ecf2-b60b-4118-90fd-acd6cddf163d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024673.jpg_d2ae417a-aaf5-478a-aeb6-aae283982298.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024743.jpg_19673a44-edfd-436a-bb9a-0abc82e23bfe.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024787.jpg_f79ecb5d-5cde-462e-9874-a577f7fea778.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024795.jpg_0c401d0d-af00-4ccb-bc03-3c7ec15e7395.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024799.jpg_f15dafb0-b91e-4240-b7b3-fd16169a4273.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024829.jpg_e15ba5e0-5d17-44e9-9650-efde799e0307.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024833.jpg_29f47e73-8065-43eb-86ca-ed2e1cd51da5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024848.jpg_621572ee-9270-432e-b3cc-d4cab8430b9a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024848.jpg_79b2d665-dc08-47b0-b9e7-fbbef20d0aab.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024848.jpg_cc456939-b5f1-4ccf-b8e2-56a315bd2315.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024897.jpg_03ec0d47-ae31-4525-b326-b92aa1885192.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024897.jpg_0d80c858-1da8-481d-b232-8fdd318a438f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024897.jpg_2fa5289f-468c-4c4a-88ae-9ecccafe7152.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024931.jpg_645dcbba-e497-4853-b662-c6cf9a6565b7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024949.jpg_c0ea9ccd-710f-4cba-910e-7219cb39b8e7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0024984.jpg_452449e2-e02e-4491-87af-79384e232090.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025019.jpg_63588fdc-8df7-47f0-a2bc-70bbfcfdfe72.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025019.jpg_d5f9fbb4-b3c9-42e6-9219-79e9126a10da.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025031.jpg_00fabe62-1fc8-47b1-be70-f18d586f3aeb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025031.jpg_245dfd53-0e95-4f73-9fd4-2d89de0a40b1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025031.jpg_7669e24b-f794-49a7-a938-5f8d7b36d846.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025102.jpg_65ef4597-9981-4161-b24a-b06fc8720843.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025144.jpg_19b4337f-040d-46d4-a0f9-f942b5b80c63.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025144.jpg_bc5ed53b-8293-4210-a747-fb1e2b49f1c7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025266.jpg_2932f7f8-60ba-4f38-be45-e32281ce2f21.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025266.jpg_fbc4320a-254f-4a44-9a85-6455cfd23eea.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025285.jpg_0e0a2131-3c28-4754-b95b-eabf98b80036.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025285.jpg_1b1d66fb-6851-499e-9fa2-010c9f3b5dfc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025285.jpg_5160bbbe-c8c3-4279-b740-27e163280164.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025299.jpg_373dc1b1-1ad6-49f6-bb07-c7206760699c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025299.jpg_d14cb71d-d6a3-4b22-8245-054ce9fa0951.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025322.jpg_d63150c8-d144-4b95-a219-cb2942b3dc8b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025322.jpg_f93ceaef-f941-4447-b341-28aab42ea9b2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025362.jpg_2015bcbc-37ce-4592-ba21-430f6f626a27.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025417.jpg_089087bd-1a0a-4676-b56e-dd529f32b84a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025417.jpg_4c45bbdd-93b4-4cfd-a43d-95b85f0ca979.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025417.jpg_64f22e84-0710-4d19-9cbb-d479b1c01708.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025417.jpg_6917d870-1b48-4971-be04-e2311a6cda92.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025417.jpg_7494ad3a-c7d2-4d58-8061-019e3fe53aff.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025417.jpg_8c2c78f5-afe3-4284-921b-aae76284e9fe.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025433.jpg_48fc142c-a85b-41db-b034-ee369e5548bc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025433.jpg_fdb0154a-82b3-46da-8e9d-b79749dc4c3c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025467.jpg_27605939-3a3c-473b-b865-f963c25483d9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025467.jpg_eadd04bb-6672-42f8-8194-b58134807e3c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025509.jpg_6e76557e-8435-4049-bfe9-6038cc769d96.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025509.jpg_ab0d660d-4ac4-4656-87fb-87bc0baa86df.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025513.jpg_8bc1e8b8-31c9-4cc3-a908-bc0dbcc44258.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025530.jpg_826cb0a3-ff5a-4066-b018-6f3668853728.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025530.jpg_a13110f7-7139-46f7-b20b-6e6169f29a0b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025530.jpg_c2f516ff-07dd-4548-a174-8f43c5c9bc35.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025530.jpg_e0817e1d-3f55-4f40-8cf7-071529c18eb8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025557.jpg_06d3d738-d265-4533-9c06-5f1f421aec0a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025557.jpg_4c45017e-b48e-44af-9fec-e96b745902b8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025557.jpg_f844fee9-4e91-49e0-aaaf-a3017327a9f1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025576.jpg_66f700e2-8eea-4d5e-a67f-78f5175ad806.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025601.jpg_56728668-1c3c-49a2-abe5-dc428c0f5aef.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025601.jpg_e84a62a1-bbff-49ec-b897-18922a414689.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025630.jpg_0278b996-97bb-49f3-bb7f-7defa34e5616.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025644.jpg_3788b8b6-d97a-4110-9b32-be37c617dd28.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025644.jpg_cd4df5e1-421e-4059-b467-520178fba172.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025644.jpg_ea161105-55b7-482b-8135-d7debde79864.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025650.jpg_56b2c450-7d7f-4da8-aa00-96ad1236ef37.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025691.jpg_a5ff3cda-dbcf-431b-9a7f-b252690ff4e9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025711.jpg_8d758992-4b4f-4cec-8898-1e16d2072124.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025718.jpg_5cf9beed-1a29-4b50-a816-188182695e7d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025718.jpg_65ce4ce8-f7c4-4d56-8062-2bd0b3aeae5b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025731.jpg_b68ecaa3-561c-4d54-a84e-e151e5eae98f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025731.jpg_e4f6c90d-e093-42d3-a96c-22cf4d58272f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025752.jpg_d009188f-03da-44b9-8759-6d67ddcfd25f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025793.jpg_6f259bbf-5362-4a6e-9c93-64e31007e5eb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025793.jpg_8900daee-43e7-4f1c-b0a8-5938b70f7743.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025818.jpg_eed76dd3-cc5d-4af4-a90f-60ee3909a52c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025824.jpg_d529c237-5eae-4556-97f8-85a6cffb77ad.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025826.jpg_1df5e59f-6c3a-41ab-b379-d1f9aace30cd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025826.jpg_3c0b515a-d75e-454b-b37c-deb083ead86f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025940.jpg_be52c548-9c51-4356-8741-b36c1fd0a745.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025971.jpg_358bcb3c-0450-4e1e-9c19-b0a2695f4f72.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025971.jpg_884779e2-2239-4edf-9608-78236d594a11.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025971.jpg_9cdf1cd5-6e0e-49c4-b8cc-b41021eb2f6f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0025975.jpg_2e71b03b-86ec-42e6-9c8d-c2884e7a25a5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026074.jpg_bb1b00aa-af63-4e5b-bcd9-f60ef2df83a8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026090.jpg_e2c85859-03b5-4f70-9a24-4dd203a238b1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026117.jpg_86f02379-5f7c-4cf8-8377-e3c1511c597f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026117.jpg_d1e74aee-97ec-4f42-8aa8-5bdd9fd7007f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026118.jpg_8c00fade-8b2a-421a-853c-8280218cdef9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026118.jpg_c72eea70-9231-46c3-8cb9-83cdd31e18a0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026154.jpg_2a4cd687-7f0e-43e5-9576-1b87a5d6dd0c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026154.jpg_7d84a3ee-0317-45ea-86d3-94d108118d52.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026154.jpg_dcd7fd72-8129-47cd-9d42-a3ec0a28cc33.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026154.jpg_deb23b32-c6f4-407c-8cc9-053539cf8a8d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026192.jpg_5ac6e379-4c63-4d2d-8c0b-0a188f80e485.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026213.jpg_10b6deac-0a3e-4b40-bee6-8b387f3b5d13.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026230.jpg_07dc6df8-a727-4803-865e-52d39a43b954.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026230.jpg_7d4a789f-d0af-4768-8ade-31f2dac8018a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026275.jpg_9e0098f2-c166-44db-a7d5-fb3c651568bd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026282.jpg_109575c8-192b-4dda-9cc1-89d2c30af0bc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026282.jpg_2fbf9f93-1fe5-4563-a32b-660d4433e0a7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026282.jpg_a86ceb24-426e-4645-8171-8c2bf9e77e20.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026321.jpg_5f5211b7-3d19-47dd-bc1d-e5744a8f8dac.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026321.jpg_d70b822a-c100-48b3-9577-5111bba4883b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026343.jpg_5ce3d8c5-4068-4b79-bdb2-4b0abe9fc52d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026343.jpg_78c30d97-e0d3-4bca-bf2d-4a0896b7e827.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026350.jpg_40354f12-3706-443a-a35f-20e96c79aced.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026433.jpg_0ab56260-0ba2-4bcd-986d-8d819034e205.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026433.jpg_5174bcbc-8a61-4ab0-bf28-39bd2f35b5a4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026433.jpg_b104bf6d-5303-41f2-af85-d11ecc1ce1e4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026439.jpg_3d293b22-cc1f-40b3-bd50-336c67550f8e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026439.jpg_4e9b93bd-6fb5-4c7a-a39e-7cade806c90d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026439.jpg_67562575-3c65-45dc-9e15-0b2b19b8664a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026442.jpg_3aa207dd-723b-4cfc-a6d3-af80d9fb7574.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026442.jpg_3d78af1a-7f4e-4e7a-add1-338199663d13.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026453.jpg_fa1f1680-a0fd-472d-8b3c-87a69b98b05b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026496.jpg_544204a4-e6dc-464f-9972-907b2da5557c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026528.jpg_074adc79-6b31-4697-b71a-1412bcb73949.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026528.jpg_841bdfb5-542a-412c-8b5d-c546d8c315bc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026574.jpg_52619bc2-14ef-4552-ae25-3e29a0d7ae25.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026574.jpg_8fbf9df8-fe13-4572-8a2b-2ccb77c4bccb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026593.jpg_064d0bf4-5127-4073-9fb5-dda88ab7e3e4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026643.jpg_95e2f27e-c20d-4cd9-a637-68c1e2840bd8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026643.jpg_e08e0e97-46b4-4e5c-a05e-0e0e8a0c92eb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026668.jpg_6b99c6eb-3b78-44cc-bd94-263a10578001.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026760.jpg_bd0e82a9-58ed-4b8b-a679-1b71f9da8226.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026766.jpg_b335822b-aab5-4229-ab0c-5f261ef2df4f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026798.jpg_ac3b91ea-e025-4879-96b5-db63c3bd33bd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026845.jpg_2fe6dd6c-b543-46fb-bbb4-21de9851433d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026855.jpg_7cda2230-787f-4cd4-9ab8-bfc50bc9fc54.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026855.jpg_c5608061-0588-4ffa-babc-f181274dadf9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026902.jpg_6854772a-2b7b-4031-80eb-6608bc350438.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026902.jpg_bbadfdce-7b57-4d68-a781-0be3958863f4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026902.jpg_f1928df8-e71d-4ec3-b57f-3e0fd2dcad35.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026940.jpg_1140f2c0-9824-40ba-b5c2-a34842975b7f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026968.jpg_68a5d3eb-28a9-41c1-a5c0-238be8a5afd4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026970.jpg_ab85c306-b6a1-465b-a74a-0f61cbd00d11.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026978.jpg_6c6090d0-cf59-4a6f-8bea-2875bd3c35aa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026995.jpg_793bab24-bdea-4297-8997-af52e0ee1bdd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026995.jpg_8e9a1517-8b37-4c6f-96e8-a6cabe0fad19.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026995.jpg_9eb9be17-5930-418f-9b6f-46b9ef7e6ab3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0026995.jpg_fd706caa-16c6-4f80-9212-8da5570973fc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027004.jpg_b23120cc-bf6a-44e4-879d-c8113c9efe32.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027004.jpg_b7d9d6b8-c0d5-4a8a-b6b8-8cb58913fe9c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027038.jpg_280004ba-4e82-46c1-9be8-52f01f537078.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027046.jpg_53940475-fd35-4050-a1f8-239d1b8c8297.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027046.jpg_a19ef398-648f-47ca-9d03-2c0f52a0c6f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027057.jpg_8cfd7f33-530b-4b20-9940-13ebf16677d4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027058.jpg_8979b48a-e10d-4cf2-96f1-c4814e948810.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027090.jpg_22e7d83f-dd55-408d-a9c0-e3fcd47ab451.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027090.jpg_b6072d3b-1efa-4df6-ae43-c308cdbffe86.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027093.jpg_bb524654-9cea-4328-896b-615291af3bb5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027093.jpg_d2f8971e-d067-4784-af3a-acd23658e4b4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027120.jpg_d2cfebb5-7260-4209-8e71-a9a39b86715b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027135.jpg_5c7a0d56-5013-4125-9432-c89549b4d487.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027146.jpg_129f9c63-211e-4314-95de-ac90fdd08484.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027146.jpg_a30389bb-3cb7-4a84-b458-8992ed7d32b2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027146.jpg_c8a13b2d-ff7a-40d0-91cb-d54c73755ca0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027146.jpg_d312630d-5370-472a-ac0c-42e8fc11499f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027175.jpg_12166310-8169-4c18-af6a-d05083cdde5e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027175.jpg_8385f2ac-e376-415b-9d6a-c2f46aa47fb7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027175.jpg_b5c98563-e907-44ec-9df2-388029faccaa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027189.jpg_67f2eefa-0630-428a-ac1b-d30c3cda9936.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027229.jpg_7c9d3497-a6cc-4b2c-9cf2-e52dbaa008a6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027281.jpg_b120f110-5984-46e6-bb43-e6daf3578fe7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027281.jpg_c1dbd768-3672-40fc-914b-25e37852c510.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027281.jpg_d40b1f34-9d0b-47b4-8b69-10e74b257620.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027291.jpg_86fe0e99-2485-485e-9649-7adcd4c66c4b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027337.jpg_059618ac-3e13-467a-b279-6addab008a58.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027337.jpg_7f2d288e-41f9-41a4-8973-656c0623a53c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027337.jpg_934ec954-d854-4b23-bbdb-ebea6813d207.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027337.jpg_e589298b-3649-4e8e-a832-13ffdb3e14ff.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027371.jpg_8e20a730-6bf4-4c72-851f-46bf81e552f6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027371.jpg_b102201a-6716-4d41-856e-056b8ac17ce5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027371.jpg_b5835942-bdd2-4287-9a2e-98f38bb716eb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027425.jpg_7624e90f-58bf-404f-a4d4-0469249cc9eb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027473.jpg_2d417325-37b6-470f-946e-db0f971fe80b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027525.jpg_3968459a-53a8-4696-9900-9208260cf4ca.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027525.jpg_c75a4383-453a-4db5-b27e-371de27320e2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027526.jpg_0244553b-3133-44f4-9f1a-07b11ba4ef45.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027526.jpg_37e2034b-55ca-4bae-bdbc-3b95a4d694cd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027526.jpg_b99f1bfb-b8ce-4074-9bcf-91142bbc6b8a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027531.jpg_3abe4018-27e3-4411-bf2c-982a3c74f19e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027531.jpg_f930f5dd-c45e-412d-8374-35d4afaf7841.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027546.jpg_4c34c88c-e685-4aaa-86d3-c6669b510b40.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027546.jpg_674736cc-30cb-40c4-b7c9-13bf44f7d744.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027546.jpg_c820e85c-74ee-42d5-987d-ea0aad8e794e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027546.jpg_e840f5f6-d38a-410a-becc-7b3101306fa1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027595.jpg_9cd2e956-76af-47f2-ba0c-6814c0ffaa93.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027609.jpg_4ef43916-9d1b-4938-a725-46498719c2cd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027609.jpg_5c985d62-62db-4a74-adc8-09e4c12ec516.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027609.jpg_a9f4210a-7cbe-4417-973e-988c07481bcf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027609.jpg_f50d2799-5e99-46cd-b4e7-91fc62b1db83.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027629.jpg_4e621227-e90f-4cac-bc30-b47977b81738.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027629.jpg_8a683037-4d0d-4180-bd69-91819a83574d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027629.jpg_ab32697e-8f1e-4084-8f12-70a8b99eea7b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027631.jpg_49d5fff1-b3d2-47d2-a8e5-8e14e26999cc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027631.jpg_5f013fb6-71fc-4094-85e8-6a887c3a5fe9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027631.jpg_f276eda2-759c-4811-844f-b168b5a2f8ab.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027631.jpg_ff8d6a32-d032-47d4-9da4-9d75afc62b33.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027675.jpg_cd0f6343-6d16-4593-9a6b-c75cbc86c273.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027677.jpg_a4b99c7c-c047-4a10-9168-a0960dcd950f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027704.jpg_72a1f44c-fb09-4f3d-bef0-448ee9a8e72d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027704.jpg_876bdd6d-a0aa-4c17-9197-576871cf4ab2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027759.jpg_3b4f3cf7-6303-4c20-b9f0-c5d072abd2f6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027759.jpg_6fbf278a-4a8e-42e3-87ed-a04c73c8d1ec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027759.jpg_87f2862b-8760-41a1-b7aa-256a9a94cdc2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027774.jpg_350e022d-42e5-49c0-ba2b-86a41df1a776.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027774.jpg_953ed3a4-a23f-41ed-95c6-eaeb9e3da382.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027786.jpg_cc9a9b2a-2407-4598-9f04-425827162d35.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027788.jpg_012ecc97-0d4e-4bce-83a1-1181ec1289f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027788.jpg_0fddef2b-d1ce-4c7a-b9cd-a5453c87e9a9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027814.jpg_53d0936e-19aa-4082-a097-fb1e78c38c09.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027814.jpg_e9080647-83aa-43c3-aaef-eaa5dd563cec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027819.jpg_de4dc7c2-f336-4310-bc32-0379c44c7599.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027825.jpg_7304b8c3-fb0e-4f8f-8b7e-b6789512b5fc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027858.jpg_4d16d8e1-e744-44ea-817a-c476a783e911.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027858.jpg_b7962434-8356-4cad-924b-01585503e8a7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027915.jpg_9839f210-b070-4267-a86b-0bfe8f1a01fc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027920.jpg_3fbd1b34-cd7b-4a28-9bcc-dfde0d4474e5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027920.jpg_a974dabb-bbb8-46f4-8271-2996e0785e46.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027966.jpg_35c6aa06-e1fa-4e08-8b2a-23b1a9c481e5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027966.jpg_fdf830c6-735a-4555-9157-9e2a300a3e20.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027976.jpg_14d909a8-580b-4ac3-88de-8607a7904b91.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027998.jpg_2fe3b403-58a1-428f-944e-40f3a60d1819.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0027998.jpg_8fef771a-2333-4ded-9feb-e0f9be85d4d6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028059.jpg_e451c99b-ac0f-48f3-af23-42e22c4873dd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028079.jpg_155b71a4-bb22-4aac-a4cc-d6e4bb4158b7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028079.jpg_2b7a8377-97a9-4411-a0ed-41e800cd9e56.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028079.jpg_6d97d06b-af6a-495a-a1ca-4dfa0bb3d9bb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028079.jpg_6edee253-8d5b-4692-a629-b84cac5e1dcc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028084.jpg_62772946-01fa-440f-8880-105742cf4aec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028085.jpg_29f60e2d-6612-4071-b6c4-ae71be4589a0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028085.jpg_431c9065-88b1-4e29-bd74-ce9b1373155f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028085.jpg_abd1fe8d-a98e-4289-922f-e6ad64fd296e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028095.jpg_019f91ec-e04a-4509-a5f6-0f51c595b2c9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028095.jpg_385a6133-f98c-470f-8ce3-3336e409d5aa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028107.jpg_296d75de-f10f-4a3c-8e3f-565d32698bc6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028107.jpg_5e1a67d0-ee38-4a36-aa04-386f14b8051a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028122.jpg_a754f43e-2117-4fb4-a3ae-36c1fdd726ae.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028147.jpg_15d16c98-c764-425d-98bc-337f30a897a5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028147.jpg_c1403550-8e50-4a7c-a4b7-39d53520a27d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028197.jpg_2cadacee-438e-4f3b-8633-17727743fb3c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028238.jpg_01c21031-d1f2-4c1b-9830-7711760354bf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028303.jpg_56cce7d3-ee16-4d3b-86b2-f4ddc75c11e2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028303.jpg_7ccd95de-37db-4bb0-a863-f047d2f94b79.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028303.jpg_e9c3d134-509f-42a8-85c9-70f904153e5e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028303.jpg_fd5cb0ca-7065-4594-af01-7b6170f28a87.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028316.jpg_3ce66be8-706c-4258-b300-6399001c8616.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028316.jpg_5981e258-eae0-46b3-827b-805c884bc3bb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028323.jpg_66dda3f3-ee37-4f40-9512-35c103ca574e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028323.jpg_8f42801e-8614-4c79-8b70-49244c0ef33b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028323.jpg_c22fa654-c80c-4d3e-ad1b-270db3f4333a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028329.jpg_0aa51f7e-50ce-4612-bba2-7bf8c2c77d1c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028329.jpg_ef50cfac-5f61-4659-af28-630b6d5f8c8e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028350.jpg_318701b8-2479-4994-8b36-3c94f353c338.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028350.jpg_a50facd6-6ff4-4b5f-bfef-a7b14368bd3f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028419.jpg_37998c9b-d7be-4cdd-9d20-286de4a217ee.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028419.jpg_a0690ede-d725-4c48-a935-80ed2088934f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028486.jpg_1d12f150-109f-4ab2-9d1a-dd1289560b97.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028486.jpg_ecaccb1b-b6aa-41f5-941e-ab7f8e5af30f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028495.jpg_8f49791d-4527-4530-a731-63947b42b7fc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028542.jpg_175c4da7-454f-4a1b-a86c-ce9caf15aa97.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028542.jpg_9cea01e2-5f30-4c91-8a6a-ba6636604b9f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028568.jpg_bcf2a3f3-56fb-47f7-aeb8-b27621ed1e97.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028570.jpg_154992eb-128b-4f75-8b46-f1252aa9c5a9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028570.jpg_1a7c2142-b5eb-46a6-b2a5-1a1e795a4ef4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028570.jpg_66c57304-74b9-422c-84a6-0a78f4acaf01.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028570.jpg_91729959-5ba7-41e6-a435-92565e29d89a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028570.jpg_de16bf27-07cb-49bf-9504-896c67242ab2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028577.jpg_a3b7df86-a492-4393-9e9d-e7918411d071.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028577.jpg_f0ffb7b4-ff52-45c1-abd8-e20f7c54204c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028583.jpg_1ab07384-7af7-42ac-b68a-9647c93666d6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028583.jpg_2a5524c7-0a26-45e2-aaac-b399dca31e40.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028583.jpg_46edbbeb-41a2-45fd-8641-c9542e75b865.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028652.jpg_5541766a-d4e1-48e1-9768-93da128551b0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028652.jpg_61b786d4-05bf-4f51-a8e0-5dabd1e0385f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028653.jpg_89c62e1e-1a56-403c-afba-f735a3113cfc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028670.jpg_0412466e-91d8-4e7c-a000-65a829340fb9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028671.jpg_5ea686f0-af77-412f-82a6-6cf49ba6b26e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028671.jpg_c514e2a8-d3b2-4c6e-a6ef-191377a9466d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028671.jpg_ea5892d1-3dd5-46a7-911e-3996700f61cc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028677.jpg_59800b6a-76dd-41e0-97a4-22ab31694973.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028687.jpg_db36e387-5ba0-458c-8c20-4ef197a371b4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028687.jpg_f6969be9-57d9-4a9a-99d9-83acb2079914.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028688.jpg_52b41428-2592-4451-b1e1-fc2c10940531.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028688.jpg_fc46f020-c818-40d3-be51-4fb769db0dd1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028693.jpg_7daba169-b3c0-4f03-b672-a812e226db0a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028719.jpg_119fa2f9-f2aa-4665-a4f8-9f5bdb054048.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028719.jpg_3c3f0727-4ccf-4697-bc90-3a426762040d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028728.jpg_bab9330c-4ba3-4098-bed0-69b6a1ff7052.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028739.jpg_39b46d8f-945e-4b27-b6a0-b2b3dd17162e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028747.jpg_6a3238c4-a74c-42dd-bc5a-6626f225330b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028765.jpg_32392b94-fb23-4cd9-883f-b1754eafd87b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028765.jpg_ec793d9f-93bc-4e4f-b48b-23678b2934f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028807.jpg_9d718dc8-bf91-4367-8ad0-8d6316a53301.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028814.jpg_16dffcc2-128c-4e0f-8edb-aa9afbb0d267.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028814.jpg_5bdfea19-ae08-46b2-baad-fd157593d730.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028814.jpg_aa59db36-fdc8-44f1-8fcb-150c6c9c151a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028815.jpg_43eaccd2-bd1c-4192-8225-3cf474098c11.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028818.jpg_60957210-ef4e-4237-8f4a-f502675e0bc2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028818.jpg_6f9893ac-f35c-4bc2-81c0-99bd6d85949d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028858.jpg_661feba3-a373-483c-ab41-157dbec25fde.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028890.jpg_dc06005e-3222-459e-a125-303fdb536620.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028928.jpg_1e8feb04-b71f-4ccc-9ca4-bc30f0bd28f9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028928.jpg_74bb787c-4211-4e4f-8535-a1252e700738.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028937.jpg_7918b6ac-7d98-488c-9e2d-c81ac98f1d0d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028937.jpg_dc31f91e-a720-4782-aa9c-ca63eee20422.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028973.jpg_08d8f6f6-fd93-4b40-aa5a-f1c14deb06d6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028973.jpg_dcab3649-6092-405b-9827-67bfaddf3bc4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028973.jpg_f14d6c2c-9cf6-4113-a9b3-4118fafc9c5d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028978.jpg_cb486e10-8e46-4b87-bb5a-a1f915ca2de3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028980.jpg_5df45e70-b5ce-4c31-a952-700bcff0d56f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028994.jpg_3e28f1e6-4735-48e7-b520-f9ccc20569df.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028994.jpg_a390327f-95ca-4035-a30c-9865c34026ce.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0028994.jpg_ebfd2135-7ea5-458e-9d5f-87fc543a7ddd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029034.jpg_265a705b-d9be-46cd-a5ce-4e6834aefdee.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029034.jpg_f4c4a491-1135-4738-a344-42a180d5273e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029035.jpg_20916e81-4632-4f8f-9864-22d98ff4efe7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029053.jpg_9c9f68de-4d87-44d4-9026-6a78f859d47c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029053.jpg_b5ba0cec-6af5-401c-b4e8-ab42177ae2f8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029082.jpg_4783f605-0165-4ef8-9739-8473d22a6bcc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029082.jpg_a019fbcc-daa9-4df9-9705-149bd88d0e90.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029083.jpg_2697cdd6-bc62-43f9-a271-8b735ac68adb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029083.jpg_c011ec2c-cb4b-4388-8eb0-93837e2293bd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029083.jpg_db90fe2b-951b-4d22-ac67-6c3b6dde92c4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029083.jpg_f1c3eaf9-47a3-4b5b-9bf4-fd2030d2b882.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029083.jpg_f2df311e-f02b-4c66-9f05-fd3e418b6bbd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029123.jpg_745d4b2c-bc0f-4a91-8f89-e86a2043f2b3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029193.jpg_5200a559-506c-4ff4-bec8-68fc104682c1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029201.jpg_245dd8b5-c6a5-4c67-95ac-94f747454436.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029220.jpg_294e3c9d-f0e9-486d-a749-887789c8a7d9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029230.jpg_1ab7895d-3ea3-4473-b319-70c322f82214.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029230.jpg_7cc0f957-e32e-46f3-a7fb-69559665142e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029230.jpg_cfe4c8ca-3df8-49fc-9afd-ee3af1ef7c6a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029263.jpg_cfaaa5f4-79f0-46b5-ae9e-205dd26c1b58.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029278.jpg_40c6ee8a-fc60-42d3-8d71-7204d4a3d620.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029278.jpg_842e0672-d530-46ef-9aa1-9294bc493847.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029278.jpg_d7f6cbf4-1561-4863-9814-b28646c88683.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029323.jpg_2e372f38-8b91-44be-baf7-16db2435a284.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029323.jpg_f7a11fc0-3aca-4219-a27d-e90800ed12cd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029337.jpg_64a04536-eb4e-404f-8dfd-059c1c4a65cd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029341.jpg_5c77d27a-7026-46b7-8dcd-33d857bce2aa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029341.jpg_c11d71d0-5bf0-48cf-91d7-b558894f1255.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029352.jpg_199688f7-e1d9-41ee-8700-b84401f31394.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029372.jpg_cf9d76f4-027c-4708-ba69-118f6b8cfac4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029391.jpg_9a41bee4-e99d-499a-9ba3-645dfd544f80.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029466.jpg_29dc7798-cfdb-4d12-baf0-fb6aad1eba8b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029489.jpg_88e73703-128d-4dcf-a56f-50397741a3f1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029489.jpg_ce42fec1-3006-4bd4-a8b4-e0c372669cf6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029501.jpg_adb1e51f-54cc-402a-914a-9d712a42dd41.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029515.jpg_93239a31-8cc5-485a-9d46-7301e289b0d7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029524.jpg_ed2c5cbf-5ece-4c8f-903b-d2a85f4762dc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029539.jpg_c85d9d5b-559d-470b-afc1-aefb816d643d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029545.jpg_838e3a04-358c-48ca-92ba-57e35af71df2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029545.jpg_f41a10a0-7d87-415d-ac70-e8a99789d7ec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029546.jpg_ce9bbacc-6003-4895-bee2-49042ca0d27c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029546.jpg_ff39ca3b-56a1-467c-b64b-e219b3d55bbd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029564.jpg_09bfcfc3-e5fc-4b42-a3ef-74608be08c89.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029564.jpg_f4917344-4d6c-4002-94f4-1815efbe5c1c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029655.jpg_ec5603ac-b477-4250-a482-fa5779429fae.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029669.jpg_eaa37847-3df7-439a-93e0-b454b41f2087.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029680.jpg_076245e5-da91-4590-aec2-32cb2327b183.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029745.jpg_4d0dfda8-67bc-47c8-b801-350bba802f36.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029745.jpg_92c0feef-8156-4c20-aaf4-e4a930d6535e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029755.jpg_384b8e40-3c74-4acd-ae88-1171a59131d6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029805.jpg_6472e7bb-3401-4b5d-83de-5881704741e1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029847.jpg_2680813f-57ba-4db7-9a9b-f2c557902cf3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029856.jpg_deafd6cc-3f25-405a-a7f6-5673720396be.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029857.jpg_3417d19b-6dd5-452c-8464-9c9a1dc184e4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029919.jpg_b2312df0-0f3c-4d14-96b0-59c278c2f708.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029951.jpg_0b3b1ef6-001c-40f6-ab5e-1323897c61e0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0029974.jpg_86434ec6-b60f-4259-bc4f-efd7e6d63609.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030010.jpg_1b3d5bc0-4643-48e8-ba57-b8c3cc98b382.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030094.jpg_4cb5633e-f04d-4def-8355-5275b89866e1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030094.jpg_a7292ccd-404e-4e44-bb3a-8e17773ab5aa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030114.jpg_02322f57-8add-478e-92d0-425055b9716f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030114.jpg_78917f82-743d-4354-9ce5-b68ac39e1fda.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030145.jpg_b8f2cae8-e548-48f4-ad91-98c6900db292.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030181.jpg_a693beca-41ba-4161-aa8f-362d0ed98e15.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030197.jpg_3a374198-a091-4a26-892c-fb777c3a9658.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030197.jpg_4f85ce4a-939d-4644-bf57-1994fde3dc04.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030197.jpg_defd08b9-6401-47b5-9f46-0f38a92eb302.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030230.jpg_2f908b51-cf53-4806-876b-5d09b3f7ce3e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030230.jpg_3f6f9c52-2398-4a33-a3b8-52000e643a37.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030230.jpg_4168e785-a7b6-4b1e-8f3b-925b2804f1e2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030230.jpg_c7e9e6b4-0828-4e57-ad45-1feb82d32a0b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030233.jpg_7ae487ae-2c99-407b-90cb-82e366078209.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030249.jpg_900f6548-51c4-4c6b-85ac-c2ee984d8bbc.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030249.jpg_9dcb2ad5-4f4f-49a3-9e46-9c80f2983315.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030271.jpg_38031646-a45b-402b-b913-edff3b481a78.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030314.jpg_a75b699c-9e3f-4c6d-be09-e97c854c4e75.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030314.jpg_c7fe603a-df92-4560-8fa2-f8b5bb03f08d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030335.jpg_e665ebe6-ba5e-49f8-8e2d-c586415fb3f4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030339.jpg_328a2333-b825-4b0f-894d-01440d79c150.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030339.jpg_d7f81cd5-62e2-47c1-a66a-6cc823f00f96.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030349.jpg_942e116a-1412-48f2-b28c-f56d4f389ee2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030349.jpg_a2936b19-739c-4a08-ab09-794c53e35da2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030349.jpg_b6c134f9-e934-43f5-9340-778baa9f42e4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030349.jpg_ff7f6caa-af45-4656-b5fd-2497e8d5f87e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030352.jpg_a7052b6e-7fed-44d6-bdd3-0d171f663472.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030352.jpg_ec933428-b35a-448f-9661-c29322d08075.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030452.jpg_bbdd8a3d-4016-44f7-af98-8b99533d3a4b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030511.jpg_13a12525-e301-4a4a-b954-ee72f76a211a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030511.jpg_54b1cdc5-0529-41c4-b712-01a055121b5c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030511.jpg_666b19c6-b358-4484-836b-53da332611d0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030526.jpg_734a8c8b-e426-4aaa-9537-13c671374474.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030526.jpg_8ee5b9c2-198e-4e3a-8948-fe1b0ff7041a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030594.jpg_dca70fbf-2b9b-414d-ba87-05107feb8baa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030594.jpg_e09190dc-4fee-4897-9478-84b0ec177e99.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030659.jpg_7e4c05df-a002-4f34-9384-40e7459adccb.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030659.jpg_83d537e0-e700-4514-8d33-d27a79ad021e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030690.jpg_c5bc0026-644d-4a56-b88d-655fce130116.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030712.jpg_e2ddf967-c075-4674-b9ec-6bb2423f85ff.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030737.jpg_574d296e-72cc-4f76-a0ee-b68b2b064ecd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030755.jpg_24b0e7ed-0b91-44b7-a275-0d2fd69a8dc0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030755.jpg_368f057b-852d-4cb4-af9a-6c025c942bdf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030755.jpg_b9e7159e-3c1b-4961-afd9-fe83e80e4b0e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030778.jpg_805fb084-cae4-4f22-8691-645c2bffc8d4.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030782.jpg_dc5da5c8-8005-4946-9f5f-9a8da511de79.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030782.jpg_dea1ce7e-3ba5-42eb-8643-3059754fdef0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030800.jpg_13491f6b-3126-4ac2-af96-2b568fd34656.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030813.jpg_2fce374b-cbd5-4c15-a154-f20754dae1c0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030813.jpg_c72e5bf3-3a30-4912-9053-6a47fd07e5e6.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030868.jpg_5c64792b-c549-4246-b5c9-d0acdef4f327.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030893.jpg_1b18f53c-6e38-43c5-955b-4a07e5513cee.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030893.jpg_4deaab69-7843-4e96-9f7d-bb3dd0d7aafa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030893.jpg_6357e488-5e30-4221-bc10-ec21503636aa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030915.jpg_9f3a28a1-9617-45d4-9e34-a9026e790bb8.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030915.jpg_a9f26f97-206b-412b-8a52-2efb3769603a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030964.jpg_19fb3dcc-2d79-4be2-ba7b-7a58d9236c32.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0030964.jpg_d65d3e89-69c7-46f5-9a97-42abb477569c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031007.jpg_77beb1f2-59a6-44a0-b6b7-bf711a373abd.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031007.jpg_8c5e5f40-e924-4632-b7cb-bfca7716e171.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031007.jpg_94b54a20-99bc-4415-9651-f5d160a97498.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031009.jpg_3731c370-1743-4c2b-8de4-0891369432a0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031026.jpg_e8f393da-f0d0-4005-bff6-3aa4a8d1d417.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031041.jpg_09c8a9bc-ac4e-4e21-a218-e928f2089cd7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031041.jpg_ae0404e8-bdd4-4e32-84dc-f0598339e9ef.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031041.jpg_bd190c5a-7362-49e1-b454-5483c5fcfb29.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031056.jpg_36bc2bfe-3d1a-421e-ba5a-59600bf5512c.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031056.jpg_916ebe2e-9a1e-4726-a8a7-2082730d38d2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031062.jpg_15349665-81d3-4a6d-b01d-97e1c1b5120f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031063.jpg_54917359-7490-4fb1-8e21-de334e5c150d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031063.jpg_c0db8a6e-ac5f-49c4-af56-2ec23869f395.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031122.jpg_c9e3891e-05d1-4994-a0a5-3adfb150bd6e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031140.jpg_716b8c74-4b2e-4ac0-a623-343aa413e73f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031154.jpg_3987c710-1bd3-4535-9cd9-2b1b62ac7d4a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031154.jpg_5efd2f92-8b87-48df-8138-50dd74bd474d.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031154.jpg_76538544-bff0-483f-b1bc-aa42c7a2d36f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031154.jpg_8b17a345-99cd-4df2-b0f6-0ac5b6b97156.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031166.jpg_4cfa4abc-128d-46d2-bbe5-b2207155f98e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031166.jpg_7b9f5f8c-fe0d-44fc-8e69-f8edc6b01c50.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031166.jpg_ea3e6a51-b9c3-48eb-93d1-7a7bff194235.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031169.jpg_9c7069a6-4bff-4f84-86b5-1eadd187aef5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031169.jpg_f1afc8ec-e4ae-402b-a56f-699e7bb3faba.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031171.jpg_406d0100-b092-4f6c-9982-dc4864fed33e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031175.jpg_29321fbc-d08d-4b16-b105-581cb64650c7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031175.jpg_88f4f5aa-73fd-4d07-8e4b-c6c00afd947e.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031243.jpg_ecad0e14-53c0-44ba-b1e0-c82ee0554b6b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031245.jpg_4e6710f2-232c-4f33-acb9-c4e0a56e65b2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031258.jpg_220293f0-eb8c-4ac7-850c-049d2422b13a.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031258.jpg_709fb4e7-8770-40f5-a500-d714f706becf.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031258.jpg_ccc2d2b3-a479-4378-ad44-650c18469891.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031258.jpg_e35d55eb-970a-4938-9fd4-9e9fee71d7a5.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031284.jpg_8d84ba60-253c-4f9c-b447-fe3400de172b.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031294.jpg_4bd8e575-a9cb-402f-9b8a-94c1a7c09126.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031298.jpg_f09d04c8-e0b9-47ad-bba8-6163f71f4116.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031318.jpg_7c416513-5177-4bf8-8780-0f4af9bcb178.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031318.jpg_fe9e6fd4-40ec-4230-984a-ebe496533357.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031351.jpg_04e4efbd-041c-4875-b39d-8995a65bcdb7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031378.jpg_19e7951f-c730-42a9-acf1-a80c3ec506b9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031378.jpg_7b41a48e-e97b-44df-ad75-2f9de1ca7bc3.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031384.jpg_41c84d9d-3eca-47de-8a7d-31b73a73f73f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031384.jpg_4a633fd1-de6d-4c63-add8-ebc6d8e78b69.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031384.jpg_6ad062e5-9bd5-4d51-b73d-92d95c7ba906.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031384.jpg_ffecbc4b-5eef-44fa-916d-67485fe76dd1.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031400.jpg_b9eae17e-43ee-4b3b-ae5b-8d84c3bb4243.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031400.jpg_e03632b7-a94f-4203-a278-954e1178bf63.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031442.jpg_c33fb8dc-0ac3-43fc-bc86-c908cd2b57f0.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031442.jpg_f514e2fb-f4c2-4e57-8696-7e80508291c7.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031470.jpg_11f9228d-48c5-417f-a518-fff154f7f9f9.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031470.jpg_c31391f5-56cc-4f1b-b786-439557839c35.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031489.jpg_42294a14-5665-48ce-9dd6-02c1dc4e3c90.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031513.jpg_dbc69e80-b3fb-4d13-a95c-445034010031.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031513.jpg_ddd86ccc-47b1-47c1-87ce-caf0439937ec.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031520.jpg_17d36893-99c9-4a29-8796-29e6c99b9e8f.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031539.jpg_2e755671-3aa8-4efb-9f54-129b27c1b6fa.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031552.jpg_4d5f89c3-2a86-4a4f-a06d-0388318fd355.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031569.jpg_0ee1d4cb-7667-410e-8a28-bdb447e24823.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031569.jpg_d38110de-749d-4cc1-891b-4ec0d37419f2.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031585.jpg_679210e1-78e2-4966-be82-6c726495a620.jpg',
 'C:\\Users\\Sairam\\Downloads\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train\\basal cell carcinoma\\output\\basal cell carcinoma_original_ISIC_0031597.jpg_bf238bc5-19bf-496e-b72c-9a5d7d262e64.jpg',
 ...]
 
lesion_list_new = [os.path.basename(os.path.dirname(os.path.dirname(y))) for y in glob(os.path.join(data_dir_train1, '*','output', '*.jpg'))]
lesion_list_new

['actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'actinic keratosis',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 'basal cell carcinoma',
 ...]
data_dir_train1 = pathlib.Path(r"C:\Users\Sairam\Downloads\Skin cancer ISIC The International Skin Imaging Collaboration\Train")
train_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train1,
  seed=123,
  validation_split = 0.2,
  subset = "training",
  image_size = (img_height, img_width),
  batch_size=batch_size)
Found 6739 files belonging to 9 classes.
Using 5392 files for training.
val_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train1,
  seed=123,
  validation_split = 0.2,
  subset = "validation",
  image_size=(img_height, img_width),
  batch_size=batch_size)
Found 6739 files belonging to 9 classes.
Using 1347 files for validation.
class_names = train_ds1.class_names
print("As we can see that we have 9 category in our dataset. They are as follows:\n")
for i,j in enumerate(class_names):
    print(str(i+1) + ' - ' + j)
As we can see that we have 9 category in our dataset. They are as follows:

1 - actinic keratosis
2 - basal cell carcinoma
3 - dermatofibroma
4 - melanoma
5 - nevus
6 - pigmented benign keratosis
7 - seborrheic keratosis
8 - squamous cell carcinoma
9 - vascular lesion
Now we can see 9 category in the dataset.
1 - actinic keratosis

2 - basal cell carcinoma

3 - dermatofibroma

4 - melanoma

5 - nevus

6 - pigmented benign keratosis

7 - seborrheic keratosis

8 - squamous cell carcinoma

9 - vascular lesion

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),## normalization
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.summary()

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling_1 (Rescaling)     (None, 180, 180, 3)       0         
                                                                 
 conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 22, 22, 64)        36928     
                                                                 
 max_pooling2d_6 (MaxPooling  (None, 11, 11, 64)       0         
 2D)                                                             
                                                                 
 dropout (Dropout)           (None, 11, 11, 64)        0         
                                                                 
 flatten_1 (Flatten)         (None, 7744)              0         
                                                                 
 dense_2 (Dense)             (None, 128)               991360    
                                                                 
 dense_3 (Dense)             (None, 9)                 1161      
                                                                 
=================================================================
Total params: 1,053,033
Trainable params: 1,053,033
Non-trainable params: 0
_________________________________________________________________

epochs=30
history = model.fit(
  train_ds1,
  validation_data=val_ds1,
  epochs=epochs
)

Epoch 1/30
169/169 [==============================] - 60s 349ms/step - loss: 1.7390 - accuracy: 0.3359 - val_loss: 1.4934 - val_accuracy: 0.4321
Epoch 2/30
169/169 [==============================] - 57s 334ms/step - loss: 1.4405 - accuracy: 0.4509 - val_loss: 1.4459 - val_accuracy: 0.4165
Epoch 3/30
169/169 [==============================] - 58s 339ms/step - loss: 1.3105 - accuracy: 0.4967 - val_loss: 1.3051 - val_accuracy: 0.5033
Epoch 4/30
169/169 [==============================] - 61s 359ms/step - loss: 1.2120 - accuracy: 0.5373 - val_loss: 1.1650 - val_accuracy: 0.5553
Epoch 5/30
169/169 [==============================] - 76s 447ms/step - loss: 1.0461 - accuracy: 0.6042 - val_loss: 1.0613 - val_accuracy: 0.6065
Epoch 6/30
169/169 [==============================] - 77s 452ms/step - loss: 0.9216 - accuracy: 0.6630 - val_loss: 0.9479 - val_accuracy: 0.6474
Epoch 7/30
169/169 [==============================] - 74s 433ms/step - loss: 0.7801 - accuracy: 0.7131 - val_loss: 0.8817 - val_accuracy: 0.6652
Epoch 8/30
169/169 [==============================] - 68s 398ms/step - loss: 0.6498 - accuracy: 0.7574 - val_loss: 0.8653 - val_accuracy: 0.6904
Epoch 9/30
169/169 [==============================] - 69s 409ms/step - loss: 0.5728 - accuracy: 0.7847 - val_loss: 0.7532 - val_accuracy: 0.7194
Epoch 10/30
169/169 [==============================] - 67s 395ms/step - loss: 0.4924 - accuracy: 0.8227 - val_loss: 0.7082 - val_accuracy: 0.7602
Epoch 11/30
169/169 [==============================] - 71s 420ms/step - loss: 0.4113 - accuracy: 0.8466 - val_loss: 0.7147 - val_accuracy: 0.7476
Epoch 12/30
169/169 [==============================] - 73s 427ms/step - loss: 0.3605 - accuracy: 0.8624 - val_loss: 0.6314 - val_accuracy: 0.7966
Epoch 13/30
169/169 [==============================] - 62s 362ms/step - loss: 0.3136 - accuracy: 0.8804 - val_loss: 0.6200 - val_accuracy: 0.7929
Epoch 14/30
169/169 [==============================] - 62s 362ms/step - loss: 0.2881 - accuracy: 0.8893 - val_loss: 0.6353 - val_accuracy: 0.7981
Epoch 15/30
169/169 [==============================] - 62s 366ms/step - loss: 0.2358 - accuracy: 0.9106 - val_loss: 0.7411 - val_accuracy: 0.7728
Epoch 16/30
169/169 [==============================] - 63s 370ms/step - loss: 0.2441 - accuracy: 0.9069 - val_loss: 0.6525 - val_accuracy: 0.8144
Epoch 17/30
169/169 [==============================] - 62s 368ms/step - loss: 0.2015 - accuracy: 0.9223 - val_loss: 0.6369 - val_accuracy: 0.8033
Epoch 18/30
169/169 [==============================] - 62s 367ms/step - loss: 0.2189 - accuracy: 0.9164 - val_loss: 0.7519 - val_accuracy: 0.7706
Epoch 19/30
169/169 [==============================] - 63s 369ms/step - loss: 0.2642 - accuracy: 0.9037 - val_loss: 0.5347 - val_accuracy: 0.8359
Epoch 20/30
169/169 [==============================] - 63s 369ms/step - loss: 0.2046 - accuracy: 0.9190 - val_loss: 0.6386 - val_accuracy: 0.7981
Epoch 21/30
169/169 [==============================] - 62s 367ms/step - loss: 0.1738 - accuracy: 0.9323 - val_loss: 0.6379 - val_accuracy: 0.8151
Epoch 22/30
169/169 [==============================] - 58s 343ms/step - loss: 0.1972 - accuracy: 0.9227 - val_loss: 0.7893 - val_accuracy: 0.7773
Epoch 23/30
169/169 [==============================] - 56s 331ms/step - loss: 0.1732 - accuracy: 0.9362 - val_loss: 0.6381 - val_accuracy: 0.8174
Epoch 24/30
169/169 [==============================] - 52s 305ms/step - loss: 0.1437 - accuracy: 0.9412 - val_loss: 0.5872 - val_accuracy: 0.8359
Epoch 25/30
169/169 [==============================] - 55s 322ms/step - loss: 0.1304 - accuracy: 0.9449 - val_loss: 0.6280 - val_accuracy: 0.8211
Epoch 26/30
169/169 [==============================] - 52s 306ms/step - loss: 0.1602 - accuracy: 0.9355 - val_loss: 0.6506 - val_accuracy: 0.8070
Epoch 27/30
169/169 [==============================] - 53s 309ms/step - loss: 0.1484 - accuracy: 0.9382 - val_loss: 0.6998 - val_accuracy: 0.8278
Epoch 28/30
169/169 [==============================] - 53s 311ms/step - loss: 0.1271 - accuracy: 0.9468 - val_loss: 0.6170 - val_accuracy: 0.8196
Epoch 29/30
169/169 [==============================] - 53s 313ms/step - loss: 0.2073 - accuracy: 0.9227 - val_loss: 0.6666 - val_accuracy: 0.8515
Epoch 30/30
169/169 [==============================] - 57s 338ms/step - loss: 0.1572 - accuracy: 0.9358 - val_loss: 0.9371 - val_accuracy: 0.7996

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
Text(0.5, 1.0, 'Training and Validation Loss')

**We can see that we have reduced the overfitting as well as increased the training accuracy and validation accuracy with the help of sampling method.
Conclusion
First of all we build a simple CNN model, got an training accuracy of 89% and validation accuracy of around 52%.So, It is a clear sign of overfitting. We can see with the help of data augmented method and dropout regularization method, overfitting of the data can be reduced. In this case it has been reduced the overfitting, but decreased our training accuracy(approx. 63%) and validation accuracy(approx. 57%). To increase the accuracy of both training and validation with overfitting in check, we used augmented sampling technique with dropout regularization. So, the training accuracy has increased from around 63% to 93% approx. and validation accuracy from around 57% to 79% approx. with removal of overfitting.
**
 
