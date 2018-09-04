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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    with tf.variable_scope('vgg_16'):
        input_origin = tf.reshape(features["x"], [-1, 256, 256, 3])
        print(input_origin.shape)
        if mode == tf.estimator.ModeKeys.TRAIN:
            input_layer_crop = tf.random_crop(input_origin,size=[10,224,224,3])
            input_layer = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer_crop)
            print(input_layer.shape)
        else:
            input_layer = tf.image.resize_image_with_crop_or_pad(input_origin,224,224)

        tf.summary.image('input_image', input_layer, max_outputs=10)

        with tf.variable_scope('conv1'):
            conv1_1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv1_1')

            conv1_2 = tf.layers.conv2d(
                inputs=conv1_1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv1_2')
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

        with tf.variable_scope('conv2'):
            conv2_1 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv2_1')

            conv2_2 = tf.layers.conv2d(
                inputs=conv2_1,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv2_2')

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

        with tf.variable_scope('conv3'):
            conv3_1 = tf.layers.conv2d(
                inputs=pool2,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv3_1')

            conv3_2 = tf.layers.conv2d(
                inputs=conv3_1,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv3_2')

            conv3_3 = tf.layers.conv2d(
                inputs=conv3_2,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv3_3')

        pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

        with tf.variable_scope('conv4'):
            conv4_1 = tf.layers.conv2d(
                inputs=pool3,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv4_1')

            conv4_2 = tf.layers.conv2d(
                inputs= conv4_1,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv4_2')

            conv4_3 = tf.layers.conv2d(
                inputs= conv4_2,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                activation=tf.nn.relu,
                name='conv4_3')

        pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

        with tf.variable_scope('conv5'):
            conv5_1 = tf.layers.conv2d(
                inputs=pool4,
                filters=512,
                kernel_size=[3, 3],
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                padding="same",
                activation=tf.nn.relu,
                name='conv5_1')

            conv5_2 = tf.layers.conv2d(
                inputs=conv5_1,
                filters=512,
                kernel_size=[3, 3],
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                padding="same",
                activation=tf.nn.relu,
                name='conv5_2')

            conv5_3 = tf.layers.conv2d(
                inputs=conv5_2,
                filters=512,
                kernel_size=[3, 3],
                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                bias_initializer = tf.zeros_initializer(),
                padding="same",
                activation=tf.nn.relu,
                name='conv5_3')

        pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)
        # Dense Layer
        #pool3_flat = tf.reshape(pool3, [-1, 13*13*256])
        pool5_flat = tf.layers.flatten(pool5)
        conv6_1 = tf.layers.conv2d(
            inputs=pool5,
            filters=4096,
            kernel_size=[7, 7],
	    strides=7,
            kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
            bias_initializer = tf.zeros_initializer(),
            padding="same",
            activation=tf.nn.relu,
            name='fc6')
        dropout1 = tf.layers.dropout(
            inputs=conv6_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, name='drop6')

        conv7_1 = tf.layers.conv2d(
            inputs=conv6_1,
            filters=4096,
            kernel_size=[1, 1],
	    strides=1,
            kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
            bias_initializer = tf.zeros_initializer(),
            padding="same",
            activation=tf.nn.relu,
            name='fc7')
        #dense2 = tf.layers.dense(inputs=dropout1, units=4096,
        #                        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        #                        bias_initializer = tf.zeros_initializer(),
        #                        activation=tf.nn.relu,
        #                        name='fc7')
        dropout2 = tf.layers.dropout(
            inputs=conv7_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, name='drop7')
        # Logits Layer
        conv7_flat = tf.layers.flatten(dropout2)
        logits = tf.layers.dense(inputs=conv7_flat, units=20,
                                kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                                bias_initializer = tf.zeros_initializer(),
                                name='fc8')

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.sigmoid(logits, name="sigmoid_tensor"),
	    "pool5_feature": pool5_flat,
	    "fc7_feature": conv7_flat
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.identity(tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels, logits=logits, weights=features["w"]), name='loss')

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_global_step()
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                               10000, 0.5, staircase=True)
# Passing global_step to minimize() will increment it at each step.
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

            train_summary=[]
            grads_and_vars = optimizer.compute_gradients(loss)
            for g,v in grads_and_vars:
                grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name),g)
                train_summary.append(grad_hist_summary)

            train_op = optimizer.minimize(
                loss=loss,
                global_step=global_step)

	    tf.summary.scalar('learning_rate',learning_rate)

            tf.summary.merge(train_summary)
	    tf.summary.merge_all()
            #summary_hook = tf.train.SummarySaverHook(400, output_dir='/tmp/pascal_model_scratch_vgg16' , summary_op=tf.summary.merge_all())

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)#, training_hooks=[summary_hook])

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
    NUM_ITERS = 2000
    args = parse_args()
    print(args.data_dir)
    # Load training and eval data
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=eval_labels.shape[1]),
        model_dir="/home/ubuntu/assignments/hw1/pascal_model_finetune_vgg16_LRe-3_noSubMean")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=NUM_ITERS)
    # Train the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": eval_data, "w": eval_weights},
	y=eval_labels,
	num_epochs=1,
	shuffle=False)

    # Evaluate the model and print results
    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    #pool5_feature = np.stack([p['pool5_feature'] for p in pred])
    #pool5_selected = pool5_feature[RAND_IMG_IND,:]
    #pool5_dist = edist(pool5_selected, pool5_feature)
    #pool5_nn = np.argsort(pool5_dist,axis=1)[:,1]+1
    #print('pool5_nn:', pool5_nn)

    fc7_feature = np.stack([p['fc7_feature'] for p in pred])
    select_ind = np.random.permutation(fc7_feature.shape[0])[:1000]
    gt_labels_selected = eval_labels[select_ind[:1000],:]
    fc7_selected = fc7_feature[select_ind[:1000],:]
    fc7_embedded = TSNE().fit_transform(fc7_selected)
    plt.scatter(fc7_embedded[:, 0], fc7_embedded[:, 1], c=gt_labels_selected.argmax(axis=1), cmap=plt.cm.Spectral)
    plt.savefig('tsne.jpg')

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
