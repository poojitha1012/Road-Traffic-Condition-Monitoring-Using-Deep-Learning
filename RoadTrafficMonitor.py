from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
import cv2
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import keras
import pickle
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("Road Traffic Condition Monitoring using Deep Learning") #designing main screen
main.geometry("1000x650")

global filename
global classifier

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")

    if filename:  # Check if a directory was selected
        text.delete('1.0', END)
        text.insert(END, f"{filename} Loaded\n")
        text.insert(END, "Dataset Loaded")
    else:
        text.delete('1.0', END)
        text.insert(END, "No directory selected.")


def processImages():
    text.delete('1.0', END)
    X_train = np.load('TrafficMonitoring/model/X.txt.npy')
    Y_train = np.load('TrafficMonitoring/model/Y.txt.npy')
    text.insert(END,'Total images found in dataset for training = '+str(X_train.shape[0])+"\n\n")
    test = X_train[30]
    test = cv2.resize(test,(600,400))
    cv2.imshow('Preprocess sample image showing as output', test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   


def generateModel():
    global classifier

    # Assuming 'text' is your output display widget
    text.delete('1.0', 'end')

    if os.path.exists('TrafficMonitoring/model/model.json'):
        # Load pre-existing model
        with open('TrafficMonitoring/model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("TrafficMonitoring/model/model.weights.h5")
        print(classifier.summary())

        # Load training history
        with open('TrafficMonitoring/model/history.pckl', 'rb') as f:
            data = pickle.load(f)
        acc = data.get('accuracy', None)  # Get accuracy value
        if acc is not None:
            accuracy = acc[-1] * 100
            text.insert('end', f"CNN Training Model Accuracy = {accuracy:.2f}%\n")
        else:
            text.insert('end', "Error: Accuracy information not available in the loaded history.\n")
    else:
        # Create and train a new model
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        classifier.add(Conv2D(128, (3, 3), activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units=256, activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units=4, activation='softmax'))

        print(classifier.summary())

        classifier.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Load training data
        X_train = np.load('TrafficMonitoring/model/X.txt.npy')
        Y_train = np.load('TrafficMonitoring/model/Y.txt.npy')

        # Split the data into training and validation sets
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Use flow method to generate augmented data batches
        augmented_data_generator = datagen.flow(X_train, Y_train, batch_size=16)

        # ModelCheckpoint to save the best weights
        checkpoint = ModelCheckpoint('TrafficMonitoring/model/model.weights.h5', monitor='val_loss', save_best_only=True)

        # Train the model
        hist = classifier.fit(augmented_data_generator, epochs=50, validation_data=(X_val, Y_val), callbacks=[checkpoint])

        # Save model architecture
        model_json = classifier.to_json()
        with open("TrafficMonitoring/model/model.json", "w") as json_file:
            json_file.write(model_json)

        # Save training history
        with open('TrafficMonitoring/model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

        # Display final accuracy
        accuracy = hist.history.get('accuracy', None)  # Get accuracy value
        if accuracy is not None:
            final_accuracy = accuracy[-1] * 100
            text.insert('end', f"CNN Training Model Accuracy = {final_accuracy:.2f}%\n")
        else:
            text.insert('end', "Error: 'accuracy' key not found in the training history.\n")

  
  
def predictTraffic():
    name = filedialog.askopenfilename(initialdir="testImages")    
    img = cv2.imread(name)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    XX = np.asarray(im2arr)
    XX = XX.astype('float32')
    XX = XX/255
    preds = classifier.predict(XX)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(name)
    img = cv2.resize(img,(450,450))
    msg = ''
    if predict == 0:
        cv2.putText(img, 'Accident Occured', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
        msg = 'Accident Occured'
    if predict == 1:
        cv2.putText(img, 'Heavy Traffic Detected', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
        msg = 'Heavy Traffic Detected'
    if predict == 2:
        cv2.putText(img, 'Fire Accident Occured', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
        msg = 'Fire Accident Occured'
    if predict == 3:
        cv2.putText(img, 'Low Traffic', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
        msg = 'Low Traffic'
    cv2.imshow(msg,img)
    cv2.waitKey(0)



def graph():
    f = open('TrafficMonitoring/model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN Accuracy & Loss Graph')
    plt.show()
   
font = ('times', 16, 'bold')
title = Label(main, text='Road Traffic Condition Monitoring using Deep Learning', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Image Preprocessing", command=processImages)
processButton.place(x=280,y=100)
processButton.config(font=font1) 

cnnButton = Button(main, text="Generate CNN Traffic Model", command=generateModel)
cnnButton.place(x=10,y=150)
cnnButton.config(font=font1) 

predictButton = Button(main, text="Upload Test Image & Predict Traffic", command=predictTraffic)
predictButton.place(x=280,y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=10,y=200)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()