# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

import os

loaded_ok = False
if os.path.isfile("traied_model.h5"):
    from tensorflow.keras.models import load_model
    try:
        tmp_classifier = load_model("traied_model.h5")
        classifier = tmp_classifier
        print("Loaded full model")
        loaded_ok = True
    except:
        pass

if not loaded_ok:
    print("Creating compiling and training a new model")

    # Part 1 - Building the CNN

    # Importing the Keras libraries and packages
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense

    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


    # Part 2 - Fitting the CNN to the images
    # reference: https://keras.io/api/preprocessing/image/

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 16, ##32,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 16, ##32,
                                                class_mode = 'binary')

    #classifier.fit_generator(training_set,
    classifier.fit(training_set,
                             steps_per_epoch = 448, ##8000,
                             epochs = 25,
                             validation_data = test_set,
                             validation_steps = 16) ##2000)

    '''
    Found 8000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.
    WARNING:tensorflow:From D:/.../cnn_medium/cnn.py:66: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use Model.fit, which supports generators.
    Epoch 1/25
     250/8000 [..............................] - ETA: 30:05 - loss: 0.6762 - accuracy: 0.5677
    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 200000 batches). You may need to use the repeat() function when building your dataset.
    '''

    classifier.save("traied_model.h5")

    '''
    Found 8000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.
    Epoch 1/25
    448/448 [==============================] - 26s 58ms/step - loss: 0.6794 - accuracy: 0.5700 - val_loss: 0.6716 - val_accuracy: 0.5586
    Epoch 2/25
    448/448 [==============================] - 25s 57ms/step - loss: 0.6502 - accuracy: 0.6264 - val_loss: 0.6216 - val_accuracy: 0.6484
    Epoch 3/25
    448/448 [==============================] - 25s 57ms/step - loss: 0.6181 - accuracy: 0.6547 - val_loss: 0.6049 - val_accuracy: 0.6523
    Epoch 4/25
    448/448 [==============================] - 25s 57ms/step - loss: 0.5824 - accuracy: 0.6917 - val_loss: 0.6060 - val_accuracy: 0.6523
    
    Epoch 25/25
    448/448 [==============================] - 25s 56ms/step - loss: 0.2239 - accuracy: 0.9061 - val_loss: 0.5894 - val_accuracy: 0.7812
    '''

# Part 3 - Making new predictions

import numpy as np
from tensorflow.keras.preprocessing import image

for idx in range(1,3):
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_%d.jpg' % idx, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    ##training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print("Single prediction: ", idx, " is ", prediction)

