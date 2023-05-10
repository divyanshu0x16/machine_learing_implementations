# baseline model for the dogs vs cats dataset
import io
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import image as tf_image
from tensorflow import expand_dims as tf_expand_dims
from tensorflow import summary
from keras import backend as K
from keras import callbacks
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

logs_folder = "logs"
if os.path.exists(logs_folder):
    os.system(f"rm -rf {logs_folder}")

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf_image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf_expand_dims(image, 0)
  return image
  
def image_grid(images,label_arr,pred_arr):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(50,50))
  for i in range(40):
    # Start next subplot.
    plt.subplot(8, 5, i + 1, title=f'Actual :+{label_arr[i]}+| Pred : {pred_arr[i]}')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)

  return figure

def vgg1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def vgg3():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#ChatGPT
def mlp_model():
    # Define the number of units in each layer of the MLP
    units = [ 144, 64, 32, 16]
    # Create a Sequential model
    model = Sequential()
    # Add dense layers to the model with the specified number of units
    # Flatten the input into a 1D array
    model.add(Flatten(input_shape=(200, 200, 3)))

    for i in range(len(units)):
        model.add(Dense(units[i], activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    # Print the summary of the model to see the total number of parameters
    model.build()
    model.summary()
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def vgg16():
     # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    print(flat1.shape,"##################")
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness(model, data_augmentation = False,pretrained=False):
 # define model
    model = model
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(logdir)
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

    if(pretrained==False):
        if(data_augmentation == False):
            datagen = ImageDataGenerator(rescale=1.0/255.0)

            train_it = datagen.flow_from_directory('dataset/train/', class_mode='binary', batch_size=64, target_size=(200, 200))
            test_it = datagen.flow_from_directory('dataset/test/', class_mode='binary', batch_size=64, target_size=(200, 200))
        else:
            train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1.0/255.0)

            train_it = train_datagen.flow_from_directory('dataset/train/',class_mode='binary', batch_size=64, target_size=(200, 200))
            test_it = test_datagen.flow_from_directory('dataset/test/', class_mode='binary', batch_size=64, target_size=(200, 200))
    else:
        datagen = ImageDataGenerator(featurewise_center=True)
        datagen.mean = [123.68, 116.779, 103.939]
        train_it = datagen.flow_from_directory('dataset/train/', class_mode='binary', batch_size=64, target_size=(224, 224))
        test_it = datagen.flow_from_directory('dataset/test/', class_mode='binary', batch_size=64, target_size=(224, 224))
        

    # fit model
    start_time = time.time()
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=1, callbacks=[tensorboard_callback],)
    # evaluate model
    end_time = time.time()

    print('Time: ', end_time-start_time)
    print('No. of Parameters: ', model.count_params())
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)   
    
    label_dict = {}
    for key, value in test_it.class_indices.items():
        label_dict[value] = key

    x, y = test_it.next()
    images = x
    labels = y
    predictions=model.predict(images)
    # print(x.shape,y.shape)
    label_vals=[]
    pred_vals=[]

    for j in range(len(labels)):
        image = images[j]
        label = labels[j]
        pred=round(predictions[j][0])
        # print(k, label_dict[int(label)], label_dict[pred])
        label_vals.append(label_dict[int(label)])
        pred_vals.append(label_dict[pred])
        # k += 1

    figure = image_grid(images, label_vals, pred_vals)
    file_writer = summary.create_file_writer(logdir)
    with file_writer.as_default():
        summary.image("Training data", plot_to_image(figure), step=0)

# entry point, run the test harness
# print(vgg16().count_params()," VGG")
# print(mlp_model().count_params(), " MLP")
run_test_harness(model= mlp_model(), data_augmentation=False,pretrained=False)