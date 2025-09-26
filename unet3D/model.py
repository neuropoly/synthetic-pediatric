import os
from unet3D.data import training_data, load_data_from_csv
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.callbacks as KC

def unet_3d(input_shape):

    inputs = layers.Input(shape=input_shape)

    ##  Encoder

    # Level 1
    conv1 = layers.Conv3D(24, (3, 3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = layers.GroupNormalization(groups=6)(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv3D(24, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = layers.GroupNormalization(groups=6)(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.1)(conv1)
    pool1 = layers.MaxPooling3D((2, 2, 2), padding='same')(conv1)

    # Level 2
    conv2 = layers.Conv3D(36, (3, 3, 3), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = layers.GroupNormalization(groups=6)(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv3D(36, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = layers.GroupNormalization(groups=6)(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.1)(conv2)
    pool2 = layers.MaxPooling3D((2, 2, 2), padding='same')(conv2)

    # Level 3
    conv3 = layers.Conv3D(48, (3, 3, 3), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = layers.GroupNormalization(groups=6)(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv3D(48, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = layers.GroupNormalization(groups=6)(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Dropout(0.1)(conv3)
    pool3 = layers.MaxPooling3D((2, 2, 2), padding='same')(conv3)

    # Level 4
    conv4 = layers.Conv3D(60, (3, 3, 3), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = layers.GroupNormalization(groups=6)(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv3D(60, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = layers.GroupNormalization(groups=6)(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Dropout(0.1)(conv4)
    pool4 = layers.MaxPooling3D((2, 2, 2), padding='same')(conv4)

    # Level 5 (Bottleneck)
    conv5 = layers.Conv3D(72, (3, 3, 3), kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = layers.GroupNormalization(groups=6)(conv5)
    conv5 = layers.Activation('relu')(conv5)

    ##  Decoder

    # Level 4
    up6 = layers.Conv3DTranspose(60, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    concat6 = layers.concatenate([up6, conv4], axis=-1)
    conv6 = layers.Conv3D(60, (3, 3, 3), kernel_initializer='he_normal', padding='same')(concat6)
    conv6 = layers.GroupNormalization(groups=6)(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv3D(60, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = layers.GroupNormalization(groups=6)(conv6)
    conv6 = layers.Activation('relu')(conv6)

    # Level 3
    up7 = layers.Conv3DTranspose(48, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    concat7 = layers.concatenate([up7, conv3], axis=-1)
    conv7 = layers.Conv3D(48, (3, 3, 3), kernel_initializer='he_normal', padding='same')(concat7)
    conv7 = layers.GroupNormalization(groups=6)(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv3D(48, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = layers.GroupNormalization(groups=6)(conv7)
    conv7 = layers.Activation('relu')(conv7)

    # Level 2
    up8 = layers.Conv3DTranspose(36, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    concat8 = layers.concatenate([up8, conv2], axis=-1)
    conv8 = layers.Conv3D(36, (3, 3, 3), kernel_initializer='he_normal', padding='same')(concat8)
    conv8 = layers.GroupNormalization(groups=6)(conv8)
    conv8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv3D(36, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = layers.GroupNormalization(groups=6)(conv8)
    conv8 = layers.Activation('relu')(conv8)

    # Level 1
    up9 = layers.Conv3DTranspose(24, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
    concat9 = layers.concatenate([up9, conv1], axis=-1)
    conv9 = layers.Conv3D(24, (3, 3, 3), kernel_initializer='he_normal', padding='same')(concat9)
    conv9 = layers.GroupNormalization(groups=6)(conv9)
    conv9 = layers.Activation('relu')(conv9)
    conv9 = layers.Conv3D(24, (3, 3, 3), kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = layers.GroupNormalization(groups=6)(conv9)
    conv9 = layers.Activation('relu')(conv9)

    # Output layer
    outputs = layers.Conv3D(1, (1, 1, 1), activation='linear')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


for fold_index in range(5):
    
    model_dir = f'model/fold_{fold_index+1}'
    os.makedirs(model_dir, exist_ok=True)

    input_shape = (192, 192, 160, 1)
    model = unet_3d(input_shape)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    model.summary()

    save_file_name = os.path.join(model_dir, '{epoch:03d}.h5')

    path_x_train, path_y_train, path_x_val, path_y_val = load_data_from_csv(fold_index, n_folds=5)

    x_sample, y_sample, generator, val_generator = training_data(path_x_train, path_y_train, path_x_val, path_y_val, batch_size=2, dim=(192, 192, 160))  

    for batch_x, batch_y in generator:
        print("X shape:", batch_x.shape, "Y shape:", batch_y.shape)
        break

    model.fit(generator, validation_data=val_generator, batch_size=2, epochs=150, steps_per_epoch=len(generator),
            validation_steps=len(val_generator), callbacks=KC.ModelCheckpoint(save_file_name, save_best_only=True, verbose=1))