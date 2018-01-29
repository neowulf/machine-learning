import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import bcolz


class KerasVgg16:
    def __init__(self, weights_dir):
        self.ttl_outputs = 2
        self.learning_rate = 0.01
        self.weights_dir = weights_dir
        self.model = self.create_model(self.learning_rate, self.ttl_outputs)

    def create_model(self, learning_rate, ttl_outputs):
        base_model = VGG16(weights='imagenet', include_top=True)

        inputs = base_model.input
        outputs = Dense(ttl_outputs, activation='softmax')(base_model.output)

        model = Model(inputs=inputs, outputs=outputs)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def process_img(self, img, using_generator=True):
        """
        :param img: could be the file path or the numpy representation
        :param using_generator: if True, `img` is assumed to be the numpy repr
        :return:
        """
        if using_generator is False:
            img = image.load_img(img, target_size=(224, 244))
            f = img.img_to_array(img)
        f = np.expand_dims(img, axis=0)
        f = preprocess_input(f)
        return f

    def save_weights(self, filename):
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.weights_dir, filename + '.h5'))

    ################################################################################################
    # Processes the images while training - takes a long time!
    ################################################################################################

    def generator(self, directory, batch_size, class_mode='categorical', shuffle=True):
        datagen = ImageDataGenerator(
            data_format='channels_last',
            preprocessing_function=self.process_img)

        datagenerator = datagen.flow_from_directory(
            directory,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )

        return datagenerator

    def save_generator(self, generator, filename):
        batches = generator
        batch_array = np.concatenate([batches.next() for i in range(batches.samples)])
        return self.save_array(filename, batch_array)

    def load_bcolz_generator(self, filename, labels, batch_size):
        """
        Feed the result in fit_generator as a generator
        """
        batch_array = self.load_array(filename)
        gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
                                       width_zoom_range=0.05, zoom_range=0.05, channel_shift_range=10,
                                       height_shift_range=0.05, shear_range=0.05, horizontal_flip=True)

        gen = image.ImageDataGenerator()
        return gen.flow(batch_array, labels, batch_size)

    def onehot(self, x):
        return to_categorical(x)

    def save_array(self, filename, arr):
        f = os.path.join(self.weights_dir, filename + '.colz')
        c=bcolz.carray(arr, rootdir=f, mode='w')
        c.flush()
        return c

    def load_array(self, filename):
        f = os.path.join(self.weights_dir, filename + '.colz')
        return bcolz.open(f)[:]

    def classifications(self, train_generator):
        classes = list(iter(train_generator.class_indices))
        for c in train_generator.class_indices:
            classes[train_generator.class_indices[c]] = c

        return classes

    def finetune(self,
                 train_generator,
                 validation_generator,
                 epochs=3,
                 weights_filename_template=None,
                 use_multiprocessing=False):

        train_steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = validation_generator.samples // validation_generator.batch_size

        for epoch in range(epochs):
            self.model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=use_multiprocessing,
            )

            if weights_filename_template is not None:
                self.save_weights('{}_{}'.format(weights_filename_template, epoch))

    def save_weights(self, filename):
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.weights_dir, filename + '.h5'))

    def load_weights(self, filename):
        self.model.load_weights(os.path.join(self.weights_dir, filename + '.h5'))

    ################################################################################################
    # Preprocess the images using bcolz
    ################################################################################################



    ################################################################################################
    # Predictions
    ################################################################################################

    def predict_generator(self, test_dir):
        def get_data_as_np(path):
            batches = self.generator(path, 10, class_mode=None, shuffle=False)
            return np.concatenate([batches.next() for i in range(len(batches))])

        preds = self.model.predict(get_data_as_np(test_dir), verbose=1)

        return preds

    ################################################################################################
    # Kaggle Submission
    ################################################################################################

    def classifications(self, train_generator):
        classes = list(iter(train_generator.class_indices))
        for c in train_generator.class_indices:
            classes[train_generator.class_indices[c]] = c

        return classes
