import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
import timeit
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())




################ Load training data #################
#####################################################
DATADIR = "FaceImages"
CATEGORIES = ["faces", "non_faces"]
IMG_SIZE = 200
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do faces and non_faces

        path = os.path.join(DATADIR, category)  # create path to faces and non_faces
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=faces 1=non_faces

        for img in tqdm(os.listdir(path)):  # iterate over each image per faces and non_faces
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

'''Pickle for saving data'''
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)




################# CNN Training part ################
####################################################

NAME = "Face_Detection-CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))  # tensorboard to show the training process
X = X / 255.0
dense_layers = [1]
layer_sizes = [32]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (7, 7), input_shape=X.shape[1:]))  # 32 7x7 filters
            model.add(Activation('relu'))  # rectified linear unit
            model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling operation

            for a in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (7, 7)))  # 32 7x7 filters
                model.add(Activation('relu'))  # rectified linear unit
                model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling operation

            model.add(Flatten())

            for b in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )
            train_time = timeit.Timer(stmt='model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])', \
                                      setup='from __main__ import model, X, y, tensorboard')
            print(' ')
            print("Training time: ", train_time.timeit(number=1), "seconds")
            print(' ')

            #model.fit(X, y,
            #          batch_size=32,
            #          epochs=10,
            #          validation_split=0.3,
            #          callbacks=[tensorboard])
            # validation_split: out-of-sample data

model.save('FaceDetection-CNN.model')




############### Testing Part #################
##############################################
model = tf.keras.models.load_model("FaceDetection-CNN.model")
test2 = cv2.imread("Img1.jpg", cv2.IMREAD_GRAYSCALE)

#plt.imshow(test2, cmap="gray")
#plt.show()

test2 = tf.reshape(test2, [1, test2.shape[0], test2.shape[1], 1])
test2_patches = tf.image.extract_patches(
    images=test2,
    sizes=[1, 60, 60, 1],
    strides=[1, 5, 5, 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)

test2_patches = tf.reshape(test2_patches, [test2_patches.shape[1], test2_patches.shape[2], test2_patches.shape[3]])

test2_patches = tf.reshape(test2_patches, [test2_patches.shape[0], test2_patches.shape[1], 60, 60])

test2_patches = np.array(test2_patches)

test2_patch = np.zeros((test2_patches.shape[0], test2_patches.shape[1], 200, 200))

prediction = np.zeros((test2_patches.shape[0], test2_patches.shape[1]))

#file = open("prediction.txt", "r+")
#file.truncate(0)
#f = open('prediction.txt', 'w', encoding='utf-8')

def model_prediction(test_patches, test_patch, prediction): 
    for i in range(test2_patches.shape[0]):
        for j in range(test2_patches.shape[1]):
            test2_patch[i][j] = cv2.resize(test2_patches[i][j], (200, 200))
            pred = model.predict([test2_patch[i][j].reshape(-1, 200, 200, 1)])
            prediction[i][j] = int(pred[0][0])

testing_time = timeit.Timer(stmt='model_prediction(test2_patches, test2_patch, prediction)', \
                            setup='from __main__ import model_prediction, test2_patches, test2_patch, prediction')
print(' ')
print("Testing time: ", testing_time.timeit(number=1), "seconds")
print(' ')
    #        f.write(str(int(pred[0][0])))
    #        # plt.imshow(test2_patch[i][j], cmap="gray")
    #        # plt.show()
    #    f.write('\r\n')

    #print(prediction)
    #f.close()

# manually use https://onlineimagetools.com/pixelate-image tool to get the indexes of faces
faces_position = [[10,5],[10,21],[6,36],[7,48],[7,61],[8,75],[7,89],[7,102],[20,1],[18,16],[15,32],[19,44],[19,55], [10,67],[18,80],[15,92],[13,109],[31,8],[30,27],[31,48],[28,66],[27,88],[28,110]]

# assume the nearby 5 scans predict the same face, create label matrix
file = open("label.txt", "r+")
file.truncate(0)
f = open('label.txt', 'w', encoding='utf-8')
label = np.zeros((prediction.shape[0], prediction.shape[1]))
flag_detected = 0

for p in faces_position:
    for i in range(p[0]-5, p[0]+6):
        for j in range(p[1]-5,p[1]+6):
            if prediction[i][j] == 1.0:
                prediction[i][j]=0.0
                flag_detected = 1
    if flag_detected==1:
        prediction[p[0]][p[1]]=1
    label[p[0]][p[1]]=1

for k in range(label.shape[0]):
    for j in range(label.shape[1]):
        f.write(str(int(label[k][j])))
    f.write('\r\n')
print(label)
f.close()

file = open("prediction_adjusted.txt", "r+")
file.truncate(0)
f = open('prediction_adjusted.txt', 'w', encoding='utf-8')
for k in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
        f.write(str(int(prediction[k][j])))
    f.write('\r\n')           
f.close()

def plot_confusion_matrix(model, y_m_test, y_m_pred):
    plt.figure(figsize=(10, 5))
    cf_matrix = confusion_matrix(y_m_test, y_m_pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('%s Image_1\n' % model)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    ## Display the visualization of the Confusion Matrix.
    plt.ion()
    plt.show()

plot_confusion_matrix("Face_Detection_CNN", np.ravel(label), np.ravel(prediction))

