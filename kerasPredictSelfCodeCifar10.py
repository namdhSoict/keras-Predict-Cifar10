import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import SGD



# Load train and test dataset
def load_dataset():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	return x_train, y_train, x_test, y_test


def prep_pixels(train, test):
	#convert from interger to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	#normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0

	return train_norm, test_norm


#define cnn model:
def define_model():
	model = Sequential()
	# kernel_initializer='he_uniform' se khoi tao gia tri cho cac trong cac so tuan theo quy luat phan phoi xac suat
	model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	
	#complie model
	'''
	SGD la phuong phap hoi tu nghiem thuong duoc dung trong cac mang than kinh phuc tap
	2 tham so quan trong la learning rate va momentum 
	'''
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model



# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	x_train, y_train, x_test, y_test = load_dataset()
	# prepare pixel data
	x_train, x_test = prep_pixels(x_train, x_test)
	# define model
	model = define_model()
	# fit model
	model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
	# save model
	model.save('final_model.h5')


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample_image.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict_classes(img)
	print(result[0])


# entry point, run the test harness
run_test_harness()

# entry point, run the example
run_example()
