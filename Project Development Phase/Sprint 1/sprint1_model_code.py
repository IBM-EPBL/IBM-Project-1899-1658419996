"""
# **Data Collection**

Dataset available given link -
https://drive.google.com/file/d/1ITbDvhLwyTTkuUYfNjOKhcIZh7hDgi64/view?usp=sharing
"""

from google.colab import drive

drive.mount('./content/')

# Unzip the dataset
!unzip '/content/content/MyDrive/Datasets/conversation engine for deaf and dumb.zip'

"""# **Image Preprocessing**"""

"""**Image Augumentation**"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    '/content/Dataset/training_set',
    batch_size = 100,
    class_mode = 'categorical',
    target_size = (64,64)
)

test_set = test_datagen.flow_from_directory(
    '/content/Dataset/test_set',
    batch_size = 50,
    class_mode = 'categorical',
    target_size = (64,64)
)

"""# **Model Building**

**Importing the Libraries**
"""

from keras.models import Sequential
from tensorflow import keras
from keras import optimizers

"""**Adding the layers**"""

classifier = keras.Sequential([
    keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(64,64,3)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(32,(3,3), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64,(3,3), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(9, activation='softmax')
])

"""**Compiling the Model**"""

classifier.compile(
    optimizer = optimizers.SGD(learning_rate=0.01),
    loss = 'categorical_crossentropy',
    metrics =['accuracy']
)
"""**Fitting and Saving the Model**"""

# Model is fit
model = classifier.fit(
    train_set,
    steps_per_epoch=len(train_set),
    epochs=25,
    validation_data = test_set,
    validation_steps = len(test_set)
)

model_save = classifier.save('Trained_Model.h5') # Model is saved

"""# **Testing the Model**"""

# Importing the libraries
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

imag = cv2.imread('/content/Test_image(G).jpeg', cv2.IMREAD_UNCHANGED)

# Preprocessing the Image
image_greyscale = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
thresh, image_black = cv2.threshold(image_greyscale, 230, 255, cv2.THRESH_BINARY_INV)
resize_img = cv2.resize(image_black, (64,64)) # Resizing the image to (64,64)
img = cv2.cvtColor(resize_img, cv2.COLOR_GRAY2BGR) # Converting Gray to Color image from shape(64,64,1) to (64,64,3)

# Testing the image
def test_image(img):
  classes = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I'} # Classes of the image
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x/255.0
  pred = np.argmax(classifier.predict(x)) # Predicting the image
  print(classes[pred])

test_image(img) # Final Output

