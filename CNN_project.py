import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #vizualizari distributie date 
import matplotlib.pyplot as plt #data plotting
import warnings
!pip install -q tensorflow 
# filter warnings
warnings.filterwarnings('ignore')

import os # interactiune OS -> interactiune fisiere
import tensorflow as tf
from tensorflow import keras # API tensorflow pt crearea modelului
from keras.preprocessing.image import load_img # functie pt a incarca imaginile din fisier
from keras.preprocessing.image import img_to_array # transformarea unei imagini intr-un array

# defining global variable path
image_path = "../input/ai-unibuc-24-22-2021"

# functie prin care incarcam imaginile dintr-un folder intr-o lista de fisiere 
def loadImages(path, dataType):
    # Punem fisierele intr-o lista si le sortam 
    image_files = sorted([os.path.join(path,dataType, file) for file in os.listdir(path + "/" + dataType) if file.endswith('.png')])
    return image_files

# Preluarea fisierelor din foldere
train_data = loadImages(image_path, "train")
test_data = loadImages(image_path, "test")
validation_data = loadImages(image_path, "validation")
print(validation_data)

# Prelucrarea imaginilor
train_images = list()
for filename in train_data:
	# incarcam imaginea
    img_data = load_img(filename)
	# salvam imaginea intr-un array
    img_array = img_to_array(img_data)
    train_images.append(img_array)
    if img_array.shape != (32, 32, 3):
        print("Verificare in caz ca nu au toate dimensiunile la fel")
    # print('loaded %s %s' % (filename, img_array.shape))
# Transformam imaginile intr-un np array, din care eliminam datele aditionale pentru a se incadra in model
train_images = np.asarray(train_images)[:,:,:,:1]
print(train_images.shape)
# Prelucram datele din imagine, facandu-le de tip float32 si impartindu-le la 255, normalizez setul de dat pentru cnn_kaggle_datagen
train_images = train_images.astype('float32')
train_images /= 255

test_images = list()
for filename in test_data[:]:
	# load image
    img_data = load_img(filename)
	# store loaded image
    img_array = img_to_array(img_data)
    test_images.append(img_array)
    if img_array.shape != (32, 32, 3):
        print("Verificare in caz ca nu au toate dimensiunile la fel")
#     print('> loaded %s %s' % (filename, img_array.shape))
test_images = np.asarray(test_images)[:,:,:,:1]
print(test_images.shape)
test_images = test_images.astype('float32')
test_images /= 255

validation_images = list()
for filename in validation_data[:]:
	# load image
    img_data = load_img(filename)
	# store loaded image
    img_array = img_to_array(img_data)
    validation_images.append(img_array)
    if img_array.shape != (32, 32, 3):
        print("Verificare in caz ca nu au toate dimensiunile la fel")
#     print('> loaded %s %s' % (filename, img_array.shape))
validation_images = np.asarray(test_images)[:,:,:,:1]
print(validation_images.shape)
validation_images = validation_images.astype('float32')
validation_images /= 255

# Incarc label-urile pentru datele de training si iau doar categoriile in train_cat_labels, verific distributia lor
train_labels = pd.read_csv('../input/ai-unibuc-24-22-2021/train.txt', header=None)
train_cat_labels = train_labels[1]
train_images_labels = train_labels.drop(1,axis = 1) 
print(train_images_labels)
print('#############')
print(train_cat_labels)
g = sns.countplot(train_cat_labels)
train_cat_labels.value_counts()

test_labels = pd.read_csv('../input/ai-unibuc-24-22-2021/test.txt', header=None)
validation_labels = pd.read_csv('../input/ai-unibuc-24-22-2021/validation.txt', header=None)
Y_validation = validation_labels[1]
print(Y_validation)
X_validation = validation_labels.drop(1, axis = 1)
print(validation_labels)
g = sns.countplot(Y_validation)
Y_validation.value_counts()

# Verific daca am incarcat date invalide
train_cat_labels.isnull().any()
print(train_cat_labels[0])

test_labels.isnull().any()

print("x_train shape: ",train_images.shape)
from keras.utils.np_utils import to_categorical # convertim datele in binary, pentru keras
train_cat_labels = to_categorical(train_cat_labels, num_classes = 9)

# Facem split pe datele training pentru a antrena/verifica acuratetea modelului
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_cat_labels, test_size = 0.2, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

# Testam imaginea
plt.imshow(X_train[2][:,:,0],cmap='gray')
plt.show()

from sklearn.metrics import confusion_matrix
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Folosesc arhitectura CNN, cu Keras Sequential API, prin care adaug un layer pas cu pas
# Creez 2 layere pentru modelul CNN, iar apoi conectez layerele prin Flatten pentru a lega toate featurile gasite
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# conectez layerele
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(9, activation = "softmax"))

# Optimizer de date, cu parametrii standard
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# O alta incercare de optimizer, luat din documentatia Keras
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Folosesc un learning rate optimizer pentru a ajunge la minimul pt loss function mai eficient
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

# Compilez modelul
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 20  # numarul de epoci pentru care vreau sa antrenez modelul
batch_size = 250

# data augmentation pentru a evita problema de overfitting, prin generari random ale setului de imagini
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range=5, 
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

datagen.fit(X_train)

# Model fitting
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size

# Cream graficul pentru loss function
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix
import seaborn as sns
# Prezicem datele din setul de date de training
Y_pred = model.predict(X_val)
# Convertim clasele de predictie intr-un vector reprezentat binar
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convertim datele de validare (labeluri) intr-un vector reprezentat binar
Y_true = np.argmax(Y_val,axis = 1) 
# computam confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plottam confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# prezicem rezultate
results = model.predict(test_images)

# selectam indicele cu probabilitatea cea mai mare
results = np.argmax(results,axis = 1)

# scriem datele intr-un fisier

results = pd.Series(results,name="label")

list = []
for x in range(5000):
    list.append('0' + str(35001 + x) + '.png')
# print(list)
submission = pd.concat([pd.Series(list,name = "id"),results],axis = 1)

submission.to_csv("cnn_kaggle_datagen.csv",index=False)