import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


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

    def classifications(self, train_generator):
        classes = list(iter(train_generator.class_indices))
        for c in train_generator.class_indices:
            classes[train_generator.class_indices[c]] = c

        return classes

    def finetune(self, weights_filename_template,
                 train_generator,
                 validation_generator,
                 epochs=3,
                 save_weights=False):

        train_steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = validation_generator.samples // validation_generator.batch_size

        for epoch in range(epochs):
            self.model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps)

            if save_weights is True:
                self.save_weights('{}_{}'.format(weights_filename_template, epoch))

    def save_weights(self, filename):
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.weights_dir, filename + '.h5'))

    def predict_generator(self, test_dir):
        def get_data_as_np(path):
            batches = self.generator(path, 10, class_mode=None, shuffle=False)
            return np.concatenate([batches.next() for i in range(len(batches))])

        preds = self.model.predict(get_data_as_np(test_dir), verbose=1)

        return preds
