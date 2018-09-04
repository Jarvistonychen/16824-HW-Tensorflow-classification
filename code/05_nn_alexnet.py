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
from sklearn.metrics.pairwise import euclidean_distances as edist

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

RAND_IMG_IND = [
	320-1, 
	44-1,
	425-1,
	583-1,
	402-1,
	559-1,
	345-1,
	567-1,
	401-1,
	752-1
]

def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    input_origin = tf.reshape(features["x"], [-1, 256, 256, 3])
    print(input_origin.shape)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #input_layer_crop = tf.random_crop(input_origin,size=[-1,224,224,3])
        #input_layer = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer_crop)
        input_layer = tf.image.resize_image_with_crop_or_pad(input_origin,224,224) 
    else:
	#input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
        input_layer = tf.image.resize_image_with_crop_or_pad(input_origin,224,224) 

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides=4,
        padding="valid",
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.zeros_initializer(),
        padding="same")

    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
    # Dense Layer
    #pool3_flat = tf.reshape(pool3, [-1, 13*13*256])
    pool3_flat = tf.layers.flatten(pool3)
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                            bias_initializer = tf.zeros_initializer(),
                            activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                            bias_initializer = tf.zeros_initializer(),
                            activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20,
                            kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                            bias_initializer = tf.zeros_initializer()
			    )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor"),
	"pool5_feature": pool3_flat,
	"fc7_feature": dense2
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, weights=features["w"]), name='loss')

    #tf.summary.scalar('loss',loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        global_step = tf.train.get_global_step()
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                           10000, 0.5, staircase=True)
# Passing global_step to minimize() will increment it at each step.
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #eval_metric_ops = {
    #    "accuracy": tf.metrics.accuracy(
    #        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss)


def load_pascal(data_dir, split='train'):
	IMG_H = 256
	IMG_W = 256
	IMG_C = 3
	NUM_CLASS = 20
	# count number of images
	NUM_IMG = 0
	IMGPATH = data_dir+'/JPEGImages'
	IDXPATH = data_dir+'/ImageSets/Main'

	with open(IDXPATH+'/'+CLASS_NAMES[0]+'_'+split+'.txt', 'r') as f:
		for line in f.readlines():
		    NUM_IMG += 1
	print('Number of images: ', NUM_IMG)
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
    BATCH_SIZE = 10
    NUM_ITERS = 400
    args = parse_args()
    print(args.data_dir)
    # Load training and eval data
    #train_data, train_labels, train_weights = load_pascal(
    #    args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=eval_labels.shape[1]),
        model_dir="/home/ubuntu/assignments/hw1/pascal_model_scratch_alexnet_q23weights")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=NUM_ITERS)
    # Train the model
    #train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": train_data, "w": train_weights},
    #    y=train_labels,
    #    batch_size=BATCH_SIZE,
    #    num_epochs=None,
    #    shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": eval_data, "w": eval_weights},
	y=eval_labels,
	num_epochs=1,
	shuffle=False)

    # Evaluate the model and print results
    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pool5_feature = np.stack([p['pool5_feature'] for p in pred])
    pool5_selected = pool5_feature[RAND_IMG_IND,:]
    pool5_dist = edist(pool5_selected, pool5_feature)
    pool5_nn = np.argsort(pool5_dist,axis=1)[:,1]+1
    print('feature5_nn:', pool5_nn)

    fc7_feature = np.stack([p['fc7_feature'] for p in pred])
    fc7_selected = fc7_feature[RAND_IMG_IND,:]
    fc7_dist = edist(fc7_selected, fc7_feature)
    fc7_nn = np.argsort(fc7_dist,axis=1)[:,1]+1
    print('fc7_nn:', fc7_nn)

    pred = np.stack([p['probabilities'] for p in pred])
    rand_AP = compute_map(
	eval_labels, np.random.random(eval_labels.shape),
	eval_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = compute_map(
	eval_labels, eval_labels, eval_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    AP = compute_map(eval_labels, pred, eval_weights, average=None)
    tf.summary.scalar('mAP',tf.metrics.mean(AP) )
    print('Obtained {} mAP'.format(np.mean(AP)))
    print('per class:')
    for cid, cname in enumerate(CLASS_NAMES):
	print('{}: {}'.format(cname, _get_el(AP, cid)))


if __name__ == "__main__":
    main()
