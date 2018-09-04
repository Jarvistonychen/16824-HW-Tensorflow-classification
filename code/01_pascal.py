from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
#import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 56 * 56 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, weights=features["w"]), name='loss')

    #tf.summary.scalar('loss',loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #eval_metric_ops = {
    #    "accuracy": tf.metrics.accuracy(
    #        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss)


def load_pascal(data_dir, split='train'):
	IMG_H = 224
	IMG_W = 224
	IMG_C = 3
	NUM_CLASS = 20
	# count number of images
	NUM_IMG = 0
	IMGPATH = data_dir+'/JPEGImages'
	IDXPATH = data_dir+'/ImageSets/Main'

	with open(IDXPATH+'/'+CLASS_NAMES[0]+'_'+split+'.txt', 'r') as f:
		for line in f.readlines():
		    NUM_IMG += 1
	images = np.zeros((NUM_IMG, IMG_H, IMG_W, IMG_C), dtype=np.float32)
	labels = np.zeros((NUM_IMG,NUM_CLASS), dtype=np.float32)
	weights = np.zeros((NUM_IMG,NUM_CLASS), dtype=np.float32)

	img_dict = dict()
	img_count= 0
	for class_idx, class_name in enumerate(CLASS_NAMES):
		with open(IDXPATH+'/'+class_name+'_'+split+'.txt', 'r') as f:
		    print('file: ', class_name+'_'+split+'.txt')
		    for line in f.readlines():
			img_label = line.strip().split()
			#print(img_label)
			img_idx = img_label[0]
			label = int(img_label[1])
			#print('image: ', img_idx+'.jpg')
			if img_idx not in img_dict.keys():
			    img_dict[img_idx] = img_count

			    img = Image.open(IMGPATH+'/'+img_idx+'.jpg')
			    #print('PIL image size:', img.size)
			    img = img.resize((IMG_W, IMG_H))
			    img_np = np.asarray(img, dtype=np.float32)
			    img_np = img_np[None,...]
			    #print('numpy image size: ', img_np.shape)
			    #print(img_np)

			    images[img_dict[img_idx],...] = img_np

			    img_count+=1

			if label==1 or label==0:
			    labels[img_dict[img_idx],class_idx] = 1
			if label:
			    weights[img_dict[img_idx],class_idx] = 1

	return images, labels, weights


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    BATCH_SIZE = 100
    NUM_ITERS = 100
    args = parse_args()
    print(args.data_dir)
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="/tmp/pascal_model_scratch3")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": eval_data, "w": eval_weights},
	y=eval_labels,
	num_epochs=1,
	shuffle=False)

    for ind in range(10):
	    pascal_classifier.train(
		input_fn=train_input_fn,
		steps=NUM_ITERS,
		hooks=[logging_hook])
	    # Evaluate the model and print results
	    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
	    pred = np.stack([p['probabilities'] for p in pred])
	    rand_AP = compute_map(
		eval_labels, np.random.random(eval_labels.shape),
		eval_weights, average=None)
	    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
	    gt_AP = compute_map(
		eval_labels, eval_labels, eval_weights, average=None)
	    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
	    AP = compute_map(eval_labels, pred, eval_weights, average=None)
	    print('Obtained {} mAP'.format(np.mean(AP)))
	    print('per class:')
	    for cid, cname in enumerate(CLASS_NAMES):
		print('{}: {}'.format(cname, _get_el(AP, cid)))


if __name__ == "__main__":
    main()
