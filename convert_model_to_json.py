#!/usr/bin/env python
# convert_model_to_json.py
#  ref1: search save keras model
#        https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#  ref2: search tensorflow save weights as json
#        https://stackoverflow.com/questions/43971649/dump-weights-of-cnn-in-json-using-keras
#  ref3: search keras model get layer weights
#        https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras


import os


loaded_ok = False
if os.path.isfile("trained_model.h5"):
    from tensorflow.keras.models import load_model
    try:
        tmp_classifier = load_model("trained_model.h5")
        classifier = tmp_classifier
        print("Loaded full model from trained_model.h5")
        loaded_ok = True
    except:
        pass

if not loaded_ok:
    print("Could not load \"trained_model.h5\"")
    if os.path.isdir("trained_model") and os.path.isdir("trained_model/variables"):
        # it needs "saved_model.pb" and "variables/" under "trained_model"
        from tensorflow.keras.models import load_model
        try:
            tmp_classifier = load_model("trained_model")
            classifier = tmp_classifier
            print("Loaded full model from trained_model/")
            loaded_ok = True
        except Exception as e:
            pass

if not loaded_ok:
    print("Could not load \"trained_model.h5\" or \"trained_model/\"")
else:
    print("Converting the model to json and h5")

    # serialize model to JSON
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("model.h5")
    print("Saved model to disk")

    # dump all weights. ref2
    print("")
    print("Dumping all weights")
    weights_list = classifier.get_weights()
    for i, weights in enumerate(weights_list):
        #print(" weights index ", i, "  ", repr(weights) )
        print(" weights index ", i, "  shape ", repr(weights.shape))
    '''
         weights index  0   shape  (3, 3, 3, 32)
         weights index  1   shape  (32,)
         weights index  2   shape  (3, 3, 32, 32)
         weights index  3   shape  (32,)
         weights index  4   shape  (6272, 128)
         weights index  5   shape  (128,)
         weights index  6   shape  (128, 1)
         weights index  7   shape  (1,)
    '''

    # dump weights by layer. ref3
    print("")
    print("Dumping weights by layers")
    w_i = 0
    for i, layer in enumerate(classifier.layers):
        weights_list = layer.get_weights()
        for w, weights in enumerate(weights_list):
            print(" layer ", i, " name ", repr(layer.name), " weights ", w,
                  " index ", w_i,
                  " in shape ", repr(weights.shape) )
            #print("      ", repr(weights))
            w_i += 1
    '''
         layer  0  name  'conv2d'  weights  0  index  0  in shape  (3, 3, 3, 32)
         layer  0  name  'conv2d'  weights  1  index  1  in shape  (32,)
         layer  2  name  'conv2d_1'  weights  0  index  2  in shape  (3, 3, 32, 32)
         layer  2  name  'conv2d_1'  weights  1  index  3  in shape  (32,)
         layer  5  name  'dense'  weights  0  index  4  in shape  (6272, 128)
         layer  5  name  'dense'  weights  1  index  5  in shape  (128,)
         layer  6  name  'dense_1'  weights  0  index  6  in shape  (128, 1)
         layer  6  name  'dense_1'  weights  1  index  7  in shape  (1,)
    '''

    # later...

    # load json and create model
    from tensorflow.keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

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

    # Part 4 - More experiments
    # ref https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
    for idx in range(1,3):
        test_image = image.load_img('dataset/single_prediction/cat_or_dog_%d.jpg' % idx, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        #result = classifier.predict(test_image)
        from tensorflow import keras

        use_method_1 = False
        if use_method_1 == True:
            # method 1:
            extractor = keras.Model(inputs=classifier.inputs,
                                    outputs=[layer.output for layer in classifier.layers])
        else:
            # method 2:
            layer_name = 'dense' # 'dense' with 128 unites, or 'dense_1' with 1 unit
            extractor = keras.Model(inputs=classifier.input,
                                    outputs=classifier.get_layer(layer_name).output)

        results = extractor(test_image) # results is a list

        if use_method_1 == True:
            print("  method_1: idx %d results " % ( idx ))
            for i,x in enumerate(results):
                print("    results %d shape " % i, x.shape)

            result = results[-1][0][0]
        else:
            print("  method_2: idx %d results  shape %s" % ( idx, results.shape )) # shape (1,128)
            result = results[0][0]

        if result == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        print("Single prediction: ", idx, " is ", prediction)
