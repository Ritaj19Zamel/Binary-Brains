class Task_one:
    import tensorflow as tf 
    import matplotlib.pyplot as plt
    import numpy as np
    mnist = tf.keras.datasets.mnist
    (x_train , y_train) ,(x_test , y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train,axis=1)
    x_test = tf.keras.utils.normalize(x_test , axis = 1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(optimizer = 'adam' ,
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
    epoch = 9
    model.fit(x_train , y_train , epochs= epoch)

    value_loss , value_accuracy = model.evaluate(x_test , y_test)
    print(f'the loss is {value_loss} ==> the accuracy is {value_accuracy}')

    plt.imshow(x_train[0] , cmap=plt.cm.binary) # cm --> colormap 
    plt.show()
    print(x_train[0])

    model.save('binary_brains.model')

    new_model = tf.keras.models.load_model('binary_brains.model')

    predictions = new_model.predict(x_test)

    print(predictions)

    print(np.argmax(predictions[0]))

    plt.imshow(x_test[0])