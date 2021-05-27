import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, MaxPooling2D, AveragePooling2D, BatchNormalization, Layer
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras import Model
import tensorflow as tf

import datetime
import shutil
from utils import *
from configs import *

def process_img(img: np.ndarray):
	'''Image processing'''
	height, width = img.shape[1], img.shape[2]
	top_pos	 	= round(height*CROP_TOP)
	bottom_pos	= round(height - height*CROP_BOTTOM)
	left_pos	= round(width*CROP_LEFT)
	right_pos	= round(width - width*CROP_RIGHT)

	img = img[:, top_pos:bottom_pos, left_pos:right_pos]
	img = tf.dtypes.cast(img, tf.float32)
	img = img/(255/(VALUE_RESCALE_MAX - VALUE_RESCALE_MIN)) + VALUE_RESCALE_MIN
	img = tf.image.resize(img, RESIZE)

	sv_img = tf.image.rgb_to_hsv(img)[..., 1:3]
	return sv_img

def model(input_data):
	output_data = Lambda(process_img,
		input_shape=(160, 320, 3), output_shape=(83, 179, 2), name='process'
	)(input_data)

	output_data = Conv2D(24, 5, (2, 2), activation=relu,
		input_shape=(83, 179, 2), data_format='channels_last', name='conv1a'
	)(output_data)
	output_data = Conv2D(24, 3, (1, 1), activation=relu,
		input_shape=(40, 88, 24), data_format='channels_last', name='conv1b'
	)(output_data)
	output_data = MaxPooling2D((3, 3), (2, 2),
		input_shape=(38, 86, 24), data_format='channels_last', name='pool1c'
	)(output_data)

	output_data = Conv2D(32, 3, (2, 2), activation=relu,
		input_shape=(19, 43, 24), data_format='channels_last', name='conv2a'
	)(output_data)
	output_data = Conv2D(32, 3, (1, 1), activation=relu,
		input_shape=(8, 20, 32), data_format='channels_last', name='conv2b'
	)(output_data)
	output_data = Conv2D(42, 3, (2, 2), activation=relu,
		input_shape=(6, 18, 32), data_format='channels_last', name='conv2c'
	)(output_data)

	output_data = Flatten(input_shape=(2, 8, 48), name='flatten')(output_data)
	output_data = Dense(128, activation=relu, input_shape=(768,), name='dense1')(output_data)
	output_data = Dense(16,  activation=relu, input_shape=(128,), name='dense2')(output_data)
	output_data = Dense(1,   activation=tanh, input_shape=(16,),  name='dense3')(output_data)
	return output_data

def create_model():
	input_layer = tf.keras.Input((160, 320, 3))
	output_tensors = model(input_layer)
	model_output = tf.keras.Model(input_layer, output_tensors)
	return model_output


timer = TrainTimer(1)
finished = False
def train_model(model, train_ds, test_ds, save_to, quiet=True) -> Model:
	global finished
	finished = False

	if USE_TENSORBOARD:
		try: shutil.rmtree('logs')
		except FileNotFoundError: pass
		train_log_dir = "logs/fit/" + str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + "_model_train"
		test_log_dir  = "logs/fit/" + str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + "_model_test"

		train_summary_writer = tf.summary.create_file_writer(logdir=train_log_dir)
		test_summary_writer  = tf.summary.create_file_writer(logdir=test_log_dir)

	if (EPOCHS <= WARMUP_EPOCHS) or (LEARNING_RATE_MIN > LEARNING_RATE_MAX): raise Exception
	global_steps = 1
	batches = len(train_ds)
	main_epochs = EPOCHS - WARMUP_EPOCHS
	end_epochs = main_epochs/2
	main_epochs -= end_epochs
	warmup_batches	= WARMUP_EPOCHS	* batches
	main_batches	= main_epochs	* batches
	end_batches		= end_epochs	* batches

	total_pred  = (len(train_ds) + len(test_ds)) * EPOCHS
	total_train = len(train_ds) * EPOCHS

	loss_object = tf.keras.losses.MeanSquaredError()

	metrics_loss = tf.keras.metrics.Mean(name='train_loss')
	metrics_steering_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_accuracy')

	def train_step(images, labels):
		with tf.GradientTape() as tape:
			predictions = model(images, training=True)
			loss = loss_object(labels, predictions)
		timer.increment(0)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		timer.increment(1)

		metrics_loss(loss)
		metrics_steering_accuracy(labels[:, 0], predictions[:, 0])
		if not quiet:
			print(labels[:, 0])
			print(predictions[:, 0])

	def test_step(images, labels):
		predictions = model(images, training=False)
		t_loss = loss_object(labels, predictions)
		timer.increment(0)

		metrics_loss(t_loss)

		metrics_steering_accuracy(labels[:, 0], predictions[:, 0])
		return predictions

	def write_summary():
		tf.summary.scalar('loss', metrics_loss.result(), step=global_steps)
		tf.summary.scalar('steering error', metrics_steering_accuracy.result()*25, step=global_steps)

	global timer
	timer = TrainTimer(total_pred, total_train)
	timer.start_timer()
	for epoch in range(EPOCHS):
		metrics_loss.reset_states()
		metrics_steering_accuracy.reset_states()
		batch_num = 0

		for images, labels in train_ds:
			if global_steps > warmup_batches + main_batches:
				lr = LEARNING_RATE_MIN + (LEARNING_RATE_MAX-LEARNING_RATE_MIN)*((end_batches-global_steps+main_batches+warmup_batches)/end_batches)
			elif global_steps > warmup_batches:
				lr = LEARNING_RATE_MAX
			else:
				lr = LEARNING_RATE_MIN + (LEARNING_RATE_MAX-LEARNING_RATE_MIN)*(global_steps/warmup_batches)
			optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

			train_step(images, labels)
			if USE_TENSORBOARD:
				with train_summary_writer.as_default(): write_summary()

			if not quiet:
				train_result(epoch, EPOCHS, batch_num, batches,
							 metrics_loss.result(), metrics_steering_accuracy.result(), 0.0,
							 lr, timer.time_left(), global_steps)

			global_steps	+= 1
			batch_num 		+= 1
			batch_num 		%= batches

			metrics_loss.reset_states()
			metrics_steering_accuracy.reset_states()

		full_val, full_pred = np.array([]), np.array([])
		for test_images, test_labels in test_ds:
			predictions = test_step(test_images, test_labels)

			full_val  = np.concatenate((full_val,  test_labels[:, 0].numpy()))
			full_pred = np.concatenate((full_pred, predictions[:, 0].numpy()))

		if USE_TENSORBOARD:
			with test_summary_writer.as_default(): write_summary()

		if not quiet:
			test_result(epoch, EPOCHS,
						metrics_loss.result(), metrics_steering_accuracy.result(), 0.0, timer.time_left())
	model.compile(run_eagerly=True)
	model.save(save_to)
	finished = True

if __name__ == '__main__':
	model = create_model()
	model.summary()