import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path

print("TensorFlow version: ", tf.__version__)

def data():
	train_path = os.path.join(os.getcwd(), 'train')
	test_path = os.path.join(os.getcwd(), 'validation')

	train_cat_dir = os.path.join(train_path, 'Cats')	
	train_dog_dir = os.path.join(train_path, 'dogs')

	validation_cat_dir = os.path.join(test_path, 'cats')
	validation_dog_dir = os.path.join(test_path, 'dogs')

	print('total training cats images:', len(os.listdir(train_cat_dir)))
	print('total training dogs images:', len(os.listdir(train_dog_dir)))
	print('total validation cats images:', len(os.listdir(validation_cat_dir)))
	print('total validation dogs images:', len(os.listdir(validation_dog_dir)))

	return train_path, test_path

def data_generate(train_path, test_path):
	from tensorflow.keras.preprocessing.image import ImageDataGenerator

	IMAGE_SIZE = 500
	BATCH_SIZE = 20

	print("Image size: ", IMAGE_SIZE, "\nBatch size: ", BATCH_SIZE)

	# All images will be rescaled by 1./255
	train_datagen = ImageDataGenerator(rescale=1/255,
			rotation_range=40,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='nearest')
	validation_datagen = ImageDataGenerator(rescale=1/255)

	# Flow training images in batches of 128 using train_datagen generator
	train_generator = train_datagen.flow_from_directory(
			train_path,  # This is the source directory for training images
			target_size=(IMAGE_SIZE, IMAGE_SIZE),  # All images will be resized to 150x150
			batch_size=BATCH_SIZE,
			# Since we use binary_crossentropy loss, we need binary labels
			class_mode='binary')

	# Flow training images in batches of 128 using train_datagen generator
	validation_generator = validation_datagen.flow_from_directory(
			test_path,  # This is the source directory for training images
			target_size=(IMAGE_SIZE, IMAGE_SIZE),  # All images will be resized to 150x150
			batch_size=BATCH_SIZE,
			# Since we use binary_crossentropy loss, we need binary labels
			class_mode='binary')

	return train_generator, validation_generator

def create_base_model(trainable_param, fine_tune_layer):
	IMAGE_SIZE = 500
	IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
													include_top=False,
													weights='imagenet')
	base_model.trainable = trainable_param

	if(trainable_param==True):
		# Fine tune from this layer onwards
		fine_tune_at = fine_tune_layer
		# Freeze all the layers before the `fine_tune_at` layer
		for layer in base_model.layers[:fine_tune_at]:
			layer.trainable =  False

	return base_model

def create_model_head(base_model):
	model = tf.keras.Sequential([
			base_model,
			tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(1, activation='sigmoid')])

	print(model.summary())
	from tensorflow.keras.optimizers import RMSprop

	model.compile(loss='binary_crossentropy',
					optimizer=RMSprop(lr=0.001),
					metrics=['acc'])

	print("Number of trainable variables = ", len(model.trainable_variables))

	return model

def train(model):

	EPOCHS = 50

	train_path, test_path = data()
	train_generator, validation_generator = data_generate(train_path, test_path)

	checkpoint_path = 'modnet_training_1/cp.ckpt'
	checkpoint_dir = os.path.dirname(checkpoint_path)

	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
			save_weights_only=True,
			verbose=1)

	history = model.fit_generator(
    		train_generator,
    		steps_per_epoch=875,
    		epochs=EPOCHS,
    		verbose=1,
    		callbacks=[cp_callback],
    		validation_data = validation_generator,
    		validation_steps=375)
	"""
	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	plt.figure(figsize=(8, 8))
	plt.subplot(2, 1, 1)
	plt.plot(acc, label='Training Accuracy')
	plt.plot(val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.ylabel('Accuracy')
	plt.ylim([min(plt.ylim()),1])
	plt.title('Training and Validation Accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.ylabel('Cross Entropy')
	plt.ylim([0,1.0])
	plt.title('Training and Validation Loss')
	plt.xlabel('epoch')
	plt.show()
	"""
	save_model(model)

def save_model(model):
	import os.path

	export_dir = os.path.join(os.getcwd(), 'DLServer')
	export_dir = os.path.join(export_dir, 'DLModels')
	file_path = os.path.join(export_dir, 'ImageNetCatsVDogs.h5')
	if os.path.isfile(export_dir) is False:
		model.save(file_path)


def main():
	#creating model
	model = create_base_model(trainable_param=True, fine_tune_layer=100)
	model = create_model_head(model)

	#training
	train(model)

main()
