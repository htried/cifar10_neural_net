""""""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from keras.preprocessing import image
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import Callback

(x_train, y_train), (x_val, y_val) = cifar10.load_data()

number, height, width, channels = x_train.shape

print x_train.shape
print x_val.shape

epochs = 12
output_dim = 10
lr = 1e-3
batch_size = 32
steps_per_epoch = number / batch_size
optim = optimizers.adam(lr=lr)

ey_train = np.squeeze(np.eye(output_dim)[y_train])
ey_val = np.squeeze(np.eye(output_dim)[y_val])

datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None)
datagen.fit(x_train)

model = Sequential()

num_filts,filt_height,filt_width = 96,3,3

# model.add(Dropout(0.1,input_shape=(height, width, channels)))

model.add(Conv2D(num_filts,(filt_height,filt_width),input_shape=(height, width, channels)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(num_filts,(filt_height,filt_width)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(num_filts,(filt_height,filt_width),strides=(2,2)))
model.add(LeakyReLU())
model.add(BatchNormalization())
num_filts *= 2
# model.add(Dropout(0.4))
model.add(Conv2D(num_filts,(filt_height,filt_width)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(num_filts,(filt_height,filt_width)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(num_filts,(filt_height,filt_width),strides=(2,2)))
model.add(LeakyReLU())
model.add(BatchNormalization())
# model.add(Dropout(0.4))
model.add(Conv2D(num_filts,(filt_height,filt_width)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(num_filts,(1,1)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(output_dim,(1,1)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(Dense(output_dim, activation='softmax'))

for layer in model.layers:
    print(layer.input_shape,layer.output_shape)

model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])

class History(Callback):
    """Save losses at each batch with a Keras model callback function."""
    def on_train_begin(self, logs={}):
        """Initialize an empty list for storing model losses."""
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        """Append a loss after each batch ends."""
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

history = History()

output = model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_val,
        ey_val,
        batch_size=batch_size),
    validation_steps=ey_val.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[history])

losses = history.losses
accuracies = history.accuracies

# Plot your training loss and accuracy across epochs
f, ax1 = plt.subplots()
ax1.plot(losses, 'b-')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(accuracies, 'r-')
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params('y', colors='r')
f.tight_layout()
plt.savefig('datathon_figure.png')
plt.show()
plt.close(f)

    
