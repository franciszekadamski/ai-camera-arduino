import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2

class MobileNetClassifier:
    def __init__(self, img_height=224, img_width=224, color_mode='rgb', batch_size=32, epochs=10, models_dir='models'):
        self.img_height = img_height
        self.img_width = img_width
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.models_dir = models_dir
        self.model = None

    def _convert_grayscale_to_rgb(self, image):
        return np.stack((image,) * 3, axis=-1)

    def _get_data_generators(self, train_dir, val_dir):
        if self.color_mode == 'rgb':
            color_mode = 'rgb'
            preprocessing_function = preprocess_input
        else:
            color_mode = 'grayscale'
            preprocessing_function = lambda x: preprocess_input(self._convert_grayscale_to_rgb(x))
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),  # Resize to 224x224
            color_mode=color_mode,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),  # Resize to 224x224
            color_mode=color_mode,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        input_shape = (224, 224, 3)
        
        return train_generator, validation_generator, input_shape

    def build_model(self, input_shape):
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Input shape 224x224x3
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, data_dir):
        train_dir = data_dir
        val_dir = data_dir

        train_generator, validation_generator, input_shape = self._get_data_generators(train_dir, val_dir)
        self.build_model(input_shape)
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        checkpoint = ModelCheckpoint(os.path.join(self.models_dir, 'mobilenet_best.keras'), monitor='val_accuracy', save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_steps=validation_generator.samples // self.batch_size,
            validation_data=validation_generator,
            epochs=self.epochs,
            callbacks=[checkpoint, early_stopping]
        )

        self.model.save(os.path.join(self.models_dir, 'mobilenet_final.keras'))

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def predict_path(self, img_path=None):
        img = load_img(img_path, target_size=(224, 224), color_mode=self.color_mode)  # Resize to 224x224
        img_array = img_to_array(img)

        if self.color_mode == 'grayscale':
            img_array = self._convert_grayscale_to_rgb(img_array)
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.model.predict(img_array)
        return 'a' if prediction[0][0] < 0.5 else 'b'

    def predict(self, img):
        img_array = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        if self.color_mode == 'grayscale':
            img_array = self._convert_grayscale_to_rgb(img_array)
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.model.predict(img_array)
        return 'a' if prediction[0][0] < 0.5 else 'b'
