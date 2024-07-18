!pip install tensorflow
import numpy  as np
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense,Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Define image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

#creating training data
train_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)

#creating data with abbove parameters
train_generator=train_datagen.flow_from_directory(
    r'/content/drive/MyDrive/Brain_Tumor_Detection-20240301T063046Z-001/Brain_Tumor_Detection/Train',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator=train_datagen.flow_from_directory(
    r'/content/drive/MyDrive/Brain_Tumor_Detection-20240301T063046Z-001/Brain_Tumor_Detection/Train',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/Brain_Tumor_Detection-20240301T063046Z-001/Brain_Tumor_Detection/Test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

#define the model
model=keras.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

# complile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the model
model.fit(train_generator,validation_data=val_generator,epochs=5)

model.save("Model.h5","label.txt")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
