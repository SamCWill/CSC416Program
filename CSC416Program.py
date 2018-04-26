# MNIST classification with CNN.
# Based on the tensorflow tutorial at:
#	https:://www.tensorflow.org/tutorials/layers
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Set logging behavior
tf.logging.set_verbosity(tf.logging.INFO)

# Some constants
# These are all values within the network that are arbitrarily
# chosen, and can be changed for speed and quality.
CONV1_FILTERS = 16
CONV2_FILTERS = 32
DENSE_NODES = 512
DROPOUT_RATE = 0.4
LEARN_RATE = 0.001
BATCH_SIZE = 100
TRAIN_STEPS = 5000
LOG_ITER = 50
FILE_NAME = "mnist_06"

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def cnn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Second arg is [batch_size, image_width, image_height, channels]
	# -1 batch_size means that batch size is computed
	inputLayer = tf.reshape(features["x"], [-1, 28, 28, 1])

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Each feature is its own channel.
	# Complexity greatly increased here.
	conv1 = tf.layers.conv2d(
		inputs=inputLayer,
		filters=CONV1_FILTERS,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# For each 2x2 group of pixels, grouped such that the same
	# pixels are in no two groups, take the max value.
	# Cuts complexity by a factor of 4.
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1, 
		pool_size=[2, 2], 
		strides=2)

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	conv2 = tf.layers.conv2d(
	      inputs=pool1,
	      filters=CONV2_FILTERS,
	      kernel_size=[5, 5],
	      padding="same",
	      activation=tf.nn.relu)

	# Pooling Layer #2
	# Quarters complexity again
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2, 
		pool_size=[2, 2], 
		strides=2)

	# Flatten tensor into a batch of vectors
	flat = tf.reshape(pool2, [-1, 7 * 7 * CONV2_FILTERS])

	# Dense Layer
	# Densely connected layer with some number of neurons
	# Number of neurons picked arbitrarily
	dense = tf.layers.dense(
		inputs=flat, 
		units=DENSE_NODES, 
		activation=tf.nn.relu)

	# Dropout Layer
	# 40% of features are randomly dropped during training.
	# This encourages the network not to rely on just one
	# feature. Should be able to determine the digit even
	# if part of it is covered.
	dropout = tf.layers.dropout(
	     	inputs=dense, 
		rate=DROPOUT_RATE, 
		training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Converts dense outputs into 10 values: one for each digit
	logits = tf.layers.dense(inputs=dropout, units=10)

	# Generate predictions (for PREDICT and EVAL mode)
	predictions = {
		# Pick single most likely class [0-9]
		"classes": tf.argmax(input=logits, axis=1),
		# Generate probabilities for each class
		"probabilities": tf.nn.softmax(logits, name="softmax")
	}

	# Return prediction (PREDICT mode)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (TRAIN and EVAL modes)
	# How far are we from our desired output?
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (TRAIN mode)
	# Perform training via gradient descent.
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
		trainer = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=trainer)

	# Add evaluation metrics (EVAL mode)
	metrics = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=metrics)


def main(unusedArg):
	# Load training and eval data
	# MNIST dataset
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	trainData = mnist.train.images  # Returns np.array
	trainLabels = np.asarray(mnist.train.labels, dtype=np.int32)
	evalData = mnist.test.images  # Returns np.array
	evalLabels = np.asarray(mnist.test.labels, dtype=np.int32)

	# Create the Estimator
	classifier = tf.estimator.Estimator(
		model_fn=cnn, model_dir="/tmp/" + FILE_NAME)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	# logValues = {"probabilities": "softmax"}
	# logHook = tf.train.LoggingTensorHook(
	# 	tensors=logValues, 
	#	every_n_iter=LOG_ITER)

	# Train the model
	# Randomly select some number of training samples at a time
	# Repeat for some number of steps
	trainFunction = tf.estimator.inputs.numpy_input_fn(
		x={"x": trainData},
		y=trainLabels,
		batch_size=BATCH_SIZE,
		num_epochs=None,
		shuffle=True)
	classifier.train(
		input_fn=trainFunction,
		# hooks=[logHook],
		steps=TRAIN_STEPS)

	# Evaluate the model and print results
	evalFunction = tf.estimator.inputs.numpy_input_fn(
		x={"x": evalData},
		y=evalLabels,
		num_epochs=1,
		shuffle=False)
	results = classifier.evaluate(input_fn=evalFunction)
	print(results)


if __name__ == "__main__":
	tf.app.run()

