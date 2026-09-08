import tensorflow as tf

def treinar_modelo_cnn(x_train, y_train):
    modelo = tf.keras.Sequential([
        tf.keras.Input(
                shape=(28,28,1)),
            tf.keras.layers.Conv2D(
                    filters=5,
                    kernel_size=3, 
                    activation= 'relu'),
            tf.keras.layers.MaxPool2D(
                    pool_size=2, strides=2, padding= 'valid'
                ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation= 'relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ])
    modelo.compile(
        optimizer= 'adam', 
        loss = 'sparse_categorical_crossentropy',
        metrics= ['accuracy']
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights= True)

    modelo.fit(x_train, y_train, epochs= 20, batch_size=32, validation_split=0.1, callbacks=[callback])

    return modelo