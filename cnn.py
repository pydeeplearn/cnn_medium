# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

import os

loaded_ok = False
if os.path.isfile("trained_model.h5"):
    from tensorflow.keras.models import load_model
    try:
        tmp_classifier = load_model("trained_model.h5")
        classifier = tmp_classifier
        print("Loaded full model")
        loaded_ok = True
    except:
        pass

if not loaded_ok:
    print("Could not load \"trained_model.h5\"")
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
    print("  output shape step 1:  ", repr(classifier.output_shape)) # (None, 62, 62, 32)

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    print("  output shape step 2: ", repr(classifier.output_shape)) # (None, 31, 31, 32)

    # Step 3 and 4 - Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    print("  output shape step 3: ", repr(classifier.output_shape)) # (None, 29, 29, 32)
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    print("  output shape step 4: ", repr(classifier.output_shape)) # (None, 14, 14, 32)

    # Step 5 - Flattening
    classifier.add(Flatten())
    print("  output shape step 5: ", repr(classifier.output_shape)) # (None, 6272)

    # Step 6 and 7 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    print("  output shape step 6: ", repr(classifier.output_shape)) # (None, 128)
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    print("  output shape step 7: ", repr(classifier.output_shape)) # (None, 1)

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

if loaded_ok:
    epochs_to_run = 4
else:
    epochs_to_run = 25

cfg_incremental_train = False # set to True to train smaller batches
if cfg_incremental_train or not loaded_ok:

    def find_new_nidx():
        file_prefix = "trained_model_"
        file_suffix = ".h5"
        fn_used, dn_used = 0, 0 # file number used, dir number used
        files_list = [x for x in os.listdir("./") if os.path.isfile(x) and x.startswith(file_prefix)]
        dirs_list = [x for x in os.listdir("./saved_model/") if os.path.isfile(x) and x.isdigit()]
        files_list = sorted(files_list)
        dirs_list = sorted(dirs_list)
        if len(files_list) > 0:
            last_fn_str = files_list[-1][len(file_prefix):]
            if len(last_fn_str) > 0 and last_fn_str[0].isdigit():
                sfx_idx = last_fn_str.find(file_suffix)
                if sfx_idx > 0:
                    last_fn_str = last_fn_str[:sfx_idx]
                    last_fn = int(last_fn_str)
                    if last_fn > 0:
                        fn_used = last_fn
        if len(dirs_list) > 0:
            last_dn_str = dirs_list[-1]
            if last_dn_str.isdigit():
                last_dn = int(last_dn_str)
                if last_dn > 0:
                    dn_used = last_dn
        return max(fn_used, dn_used) + 1
    save_nidx_new = find_new_nidx()
    save_file = "trained_model_%d.h5" % save_nidx_new
    save_dir = "./saved_model/%d/" % save_nidx_new
    print("Will save result to file \"%s\" and directory \"%s\"" % (save_file, save_dir))

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
                             epochs = epochs_to_run, ##25,
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

    classifier.save(save_file)
    classifier.save(save_dir) # https://towardsdatascience.com/how-to-use-a-saved-model-in-tensorflow-2-x-1fd76d491e69
    print("Saved model to file \"%s\" directory \"%s\"" % (save_file, save_dir))
    print("Please copy the file \"%s\" to \"%s\" to make use of it" % (save_file, "trained_model.h5"))

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
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 2 of 2). These functions will not be directly callable after loading.
    --note 2022-6-27: the warning is from tf 2.9.1, not tf 2.5.0
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
print("The two singles are: the first a dog, and the second a cat.")

