#!/usr/bin/env python
# convert_model_to_json.py
#  ref1: search save keras model
#        https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#  ref2: search tensorflow save weights as json
#        https://stackoverflow.com/questions/43971649/dump-weights-of-cnn-in-json-using-keras
#  ref3: search keras model get layer weights
#        https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras


import os


classifier = None
inferencer = None

# attempt .h5
if os.path.isfile("trained_model.h5"):
    from tensorflow.keras.models import load_model
    try:
        tmp_classifier = load_model("trained_model.h5")
        classifier = tmp_classifier
        print("Loaded full model from trained_model.h5")
    except Exception as e:
        print("Exception loading \"trained_model.h5\": ", repr(e))
        pass

# attempt saved_model
if classifier is None:
    if os.path.isdir("trained_model") and os.path.isdir("trained_model/variables"):
        # it needs "saved_model.pb" and "variables/" under "trained_model"
        from tensorflow.keras.models import load_model
        try:
            tmp_classifier = load_model("trained_model")
            classifier = tmp_classifier
            print("Loaded full model from trained_model/")
        except Exception as e:
            print("Exception loading \"trained_model/\": ", repr(e))
            pass

# attempt frozen_models/frozen_graph.pb
if classifier is None:
    # ref6:
    # https://github.com/leimao/Frozen-Graph-TensorFlow/blob/master/TensorFlow_v2/example_1.py

    import tensorflow as tf

    def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        if print_graph == True:
            print("-" * 50)
            print("Frozen model layers: ")
            layers = [op.name for op in import_graph.get_operations()]
            for layer in layers:
                print(layer)
            print("-" * 50)

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

    # Load frozen graph using TensorFlow 1.x functions
    if os.path.isdir("frozen_models") and os.path.isfile("frozen_models/frozen_graph.pb"):
        with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)
    inferencer = frozen_func


if classifier is not None:
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


if classifier is not None or inferencer is not None:
    # Part 3 - Making new predictions
    import numpy as np
    from tensorflow.keras.preprocessing import image

    for idx in range(1,3):
        test_image = image.load_img('dataset/single_prediction/cat_or_dog_%d.jpg' % idx, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        if classifier is not None:
            result = classifier.predict(test_image)
            ##training_set.class_indices
            if result[0][0] == 1:
                prediction = 'dog'
            else:
                prediction = 'cat'
            print("Single prediction: ", idx, " is ", prediction)
        else:
            result = inferencer(x=tf.constant(test_image))
            if result[0][0] == 1:
                prediction = 'dog'
            else:
                prediction = 'cat'
            print("Single prediction: ", idx, " is ", repr(result), "  a ", prediction)
    print("The two singles are: the first a dog, and the second a cat.")


if classifier is not None and inferencer is not None:
    # impossible. keep the code ahd revisit later.

    # Part 4 - More experiments
    # ref https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
    for idx in range(1,3):
        test_image = image.load_img('dataset/single_prediction/cat_or_dog_%d.jpg' % idx, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        #result = classifier.predict(test_image)
        from tensorflow import keras

        use_method_1 = True
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


if classifier is not None:
    # save to single pb file: ref4
    # https://stackoverflow.com/questions/58119155/freezing-graph-to-pb-in-tensorflow2
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    import numpy as np

    model = classifier

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="frozen_graph.pb",
                      as_text=False)

