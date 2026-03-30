import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os


train_dir = 'data/train'
test_dir = 'data/test'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)


target_shape = (96, 96) 

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_shape,
    batch_size=32,
    color_mode="rgb", 
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_shape,
    batch_size=32,
    color_mode="rgb",
    class_mode='categorical',
    shuffle=False
)


classes = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weight_dict = dict(enumerate(class_weights))


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))


base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, restore_best_weights=True)


print("--- Bat dau qua trinh huan luyen Transfer Learning ---")
steps_per_epoch = max(1, (train_generator.n // train_generator.batch_size) - 1)
validation_steps = max(1, (validation_generator.n // validation_generator.batch_size) - 1)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)


model.save('D:/face-classification/my_emotion_model_transfer.h5')
print("--- Da luu model Transfer Learning thanh cong! ---")