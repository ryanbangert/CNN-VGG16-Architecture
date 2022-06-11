import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.utils import to_categorical

num_classes=10
(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
print(train_X.shape)
print(test_X.shape)
img_height, img_width, channel = train_X.shape[1],train_X.shape[2],3

train_X = train_X.reshape(-1, 32, 32, 3) # 3 for channel number
test_X = test_X.reshape(-1,32,32,3)

# normalize data
train_X = train_X.astype("float32")
test_X = test_X.astype("float32")
train_X = train_X/255
test_X = test_X/255

test_Y_one_hot = to_categorical(test_Y, num_classes=num_classes)
train_Y_one_hot = to_categorical(train_Y, num_classes=num_classes)

model = Sequential()
model.add(Conv2D(input_shape=(img_height,img_width,channel),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=num_classes, activation="softmax"))

model.summary()

import tensorflow.keras 

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=100)
test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
print("Test Loss", test_loss)
print("Test Accuracy",test_acc)
