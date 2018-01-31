import os
from pathlib import Path

import bcolz
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from bcolz_array_iterator import BcolzArrayIterator

SEED=10
current_dir = Path(os.getcwd())
DATASET_DIR = current_dir / 'dataset'
CROSSVALID_DIR = DATASET_DIR / 'cross_valid'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
SAMPLE_DIR = DATASET_DIR / 'sample'

SAMPLE_TRAIN_DIR = SAMPLE_DIR / 'train'
SAMPLE_CROSSVALID_DIR = SAMPLE_DIR / 'cross_valid'

WEIGHTS_DIR = current_dir / 'weights'

np.random.seed(SEED)

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

    ################################################################################################
    # Processes the images while training - takes a long time!
    ################################################################################################

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

    def steps(self, generator):
        result = generator.samples // generator.batch_size
        if result < 1:
            result = 1
        return result

    def finetune(self,
                 train_generator,
                 validation_generator,
                 epochs=3,
                 weights_filename_template=None,
                 use_multiprocessing=False):

        train_steps_per_epoch = self.steps(train_generator)
        validation_steps = self.steps(validation_generator)

        for epoch in range(epochs):
            self.model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps
            )
            #     max_queue_size=10,
            #     workers=1,
            #     use_multiprocessing=use_multiprocessing
            # )

            if weights_filename_template is not None:
                self.save_weights('{}_{}'.format(weights_filename_template, epoch))

    ################################################################################################
    # Save the model weights
    ################################################################################################

    def save_weights(self, filename):
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.weights_dir, filename + '.h5'))

    def load_weights(self, filename):
        self.model.load_weights(os.path.join(self.weights_dir, filename + '.h5'))

    ################################################################################################
    # Save datagenerator to bcolz array. To load:
    #   data = bcolz.open(data_dir)
    #
    # Source: http://notes.johnvial.info/using-bcolz-with-keras-generators.html
    ################################################################################################

    def save_bcolz_generator(self, gen, dir, type):
        """
        The generator should use batch_size of 1. See the following comment in https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson2.ipynb
            # Use batch size of 1 since we're just doing preprocessing on the CPU

        Save the output from a generator without loading all images into memory.

        Does not return anything, instead writes data to disk.

        :gen: A Keras ImageDataGenerator object
        :data_dir: The folder name to store the bcolz array representing the features in.
        :labels_dir: The folder name to store the bcolz array representing the labels in.
        :mode: the write mode. Set to 'a' for append, set to 'w' to overwrite existing data and 'r' to read only.
        """
        data_dir = Path(dir) / '..' / (type + '_data.bcolz')
        labels_dir = Path(dir) / '..' / (type + '_label.bcolz')
        for directory in [data_dir, labels_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                print('Found "%s". May have been processed already!' % (directory))
                return

        num_samples = gen.samples

        d, l = gen.__next__()

        data = bcolz.carray(d, rootdir=data_dir, mode='w')
        labels = bcolz.carray(l, rootdir=labels_dir, mode='w')

        for i in tqdm(range(num_samples - 1)):
            d, l = gen.__next__()
            data.append(d)
            labels.append(l)
        data.flush()
        labels.flush()

    def load_bcolz_generator(self, dir, type, batch_size=64, shuffle=True):
        X = bcolz.open(Path(dir) / '..' / (type + '_data.bcolz'), mode='r')
        y = bcolz.open(Path(dir) / '..' / (type + '_label.bcolz'), mode='r')
        return BcolzArrayIterator(X, y, batch_size=batch_size, shuffle=shuffle, seed=SEED)

    ################################################################################################
    # Preprocess the images using bcolz
    ################################################################################################

    # def save_generator(self, generator, filename):
    #     # bcolz approach
    #     # train_generator = self.generator(TRAIN_DIR, 50, shuffle=False)
    #     # self.save_generator(train_generator, 'sample_train')
    #
    #     batches = generator
    #     batch_array = np.concatenate([batches.next() for i in tqdm(range(batches.samples))])
    #     return self.save_array(filename, batch_array)
    #
    # def load_bcolz_generator(self, filename, labels, batch_size):
    #     """
    #     Feed the result in fit_generator as a generator
    #     """
    #     batch_array = self.load_array(filename)
    #
    #     # TODO image augmentation parameters
    #     # gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
    #     #                                width_zoom_range=0.05, zoom_range=0.05, channel_shift_range=10,
    #     #                                height_shift_range=0.05, shear_range=0.05, horizontal_flip=True)
    #
    #     gen = image.ImageDataGenerator()
    #     return gen.flow(batch_array, labels, batch_size)
    #
    # def onehot(self, x):
    #     return to_categorical(x)
    #
    def save_array(self, filename, arr):
        bcolz_dir = os.path.join(current_dir, filename + '.colz')

        if not os.path.exists(bcolz_dir):
            os.makedirs(bcolz_dir)
        else:
            print('dir already exists %s' % bcolz_dir)
            return 

        print('Saving to %s'.format(bcolz_dir))
        c = bcolz.carray(arr, rootdir=bcolz_dir, mode='w')
        c.flush()
        return c

    def load_array(self, filename):
        bcolz_dir = os.path.join(current_dir, filename + '.colz')
        return bcolz.open(bcolz_dir)[:]

    ################################################################################################
    # Predictions
    ################################################################################################

    def predict_generator_dir(self, test_dir):
        test_generator = self.generator(test_dir, 10, class_mode=None, shuffle=False)
        preds = self.model.predict_generator(test_generator, self.steps(test_generator))
        return test_generator, preds

    ################################################################################################
    # Kaggle Submission
    ################################################################################################

    def classifications(self, train_generator):
        classes = list(iter(train_generator.class_indices))
        for c in train_generator.class_indices:
            classes[train_generator.class_indices[c]] = c

        return classes
