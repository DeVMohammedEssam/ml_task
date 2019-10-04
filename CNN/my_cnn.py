import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer = 'adam' ,metrics=['accuracy'],loss='binary_crossentropy')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,  
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=2000)

import numpy as np
from keras.preprocessing import image
tested_image = image.load_img("dataset/single_prediction/cat_or_dot_1.jpg",target_size=(64,64))
tested_image = image.img_to_array(tested_image)
tested_image = np.expand_dims(tested_image ,axis=0 )
result = classifier.predict(tested_image)
train_datagen.class_indeces