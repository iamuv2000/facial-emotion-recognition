from google.colab import drive 
drive.mount('/content/drive')

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D 
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

num_classes = 7  #angry, disgust, fear, happy, sad, surprise, neutral batch_size = 128
epochs = 50
with open("./drive/My Drive/fer2013.csv") as f: 
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ",num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1,num_of_instances): 
    try:
        emotion, img, usage = lines[i].split(",") 
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes) 
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels) 
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels) 
    except:
        print("", end="")

x_train = np.array(x_train, 'float32') 
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32') 
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1] 
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1) 
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1) 
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples') 
print(x_test.shape[0], 'test samples')

#construct CNN structure
model = Sequential()

#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1))) 
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) 

#------------------------------
#batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
fit = True

if fit == True:
    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) 
else:
    model.load_weights('/data/facial_expression_model_weights.h5') #load weights

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.5) plt.xticks(y_pos, objects) plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#Saving the model
model.save('model.h5')

#Evaluation
train_score = model.evaluate(x_train, y_train, verbose=0) 
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
test_score = model.evaluate(x_test, y_test, verbose=0) 
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])

!ls model.h5

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default() 
drive = GoogleDrive(gauth)

model.save('model.h5')
model_file = drive.CreateFile({'title' : 'model.h5'}) 
model_file.SetContentFile('model.h5') 
model_file.Upload()

# download to google drive
drive.CreateFile({'id': model_file.get('id')})