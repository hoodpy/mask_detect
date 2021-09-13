import tensorflow as tf
import numpy as np
import os
import xml.etree.ElementTree as ET
import tools
import cv2
import time
import glob
import random
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow


class Timer():
	def __init__(self):
		self.total_time = 0
		self.calls = 0
		self.start_time = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff


class Vgg16():
	def __init__(self, batch_size=1):
		self._feat_stride = [16, ]
		self._feat_compress = [1./16, ]
		self._batch_size = batch_size
		self._predictions = {}
		self._losses = {}
		self._anchor_targets = {}
		self._proposal_targets = {}
		self._layers = {}
		self._act_summaries = []
		self._score_summaries = {}
		self._train_summaries = []
		self._event_summaries = {}
		self._variables_to_fix = {}

	def build_head(self, is_training):
		net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope="conv1")
		net = slim.max_pool2d(net, [2, 2], padding="SAME", scope="pool1")
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope="conv2")
		net = slim.max_pool2d(net, [2, 2], padding="SAME", scope="pool2")
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope="conv3")
		net = slim.max_pool2d(net, [2, 2], padding="SAME", scope="pool3")
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope="conv4")
		net = slim.max_pool2d(net, [2, 2], padding="SAME", scope="pool4")
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope="conv5")
		self._act_summaries.append(net)
		self._layers["head"] = net
		return net

	def _anchor_component(self):
		with tf.variable_scope("ANCHOR_default"):
			height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
			width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
			anchors, anchor_length = tf.py_func(tools.generate_anchors_pre,
				[height, width, self._feat_stride, self._anchor_scales, self._anchor_ratios],
				[tf.float32, tf.int32], name="generate_anchors")
			anchors.set_shape([None, 4])
			anchor_length.set_shape([])
			self._anchors = anchors
			self._anchor_length = anchor_length

	def _reshape_layer(self, bottom, num_dim, name):
		input_shape = tf.shape(bottom)
		with tf.variable_scope(name):
			to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
			reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
			to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
			return to_tf

	def _softmax_layer(self, bottom, name):
		if name == "rpn_cls_prob_reshape":
			input_shape = tf.shape(bottom)
			bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
			reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
			return tf.reshape(reshaped_score, input_shape)
		return tf.nn.softmax(bottom, name=name)

	def build_rpn(self, net, is_training, initializer):
		self._anchor_component()
		rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
		self._act_summaries.append(rpn)
		rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer,
			padding="VALID", activation_fn=None, scope="rpn_cls_score")
		rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, "rpn_cls_score_reshape")
		rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
		rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
		rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer,
			padding="VALID", activation_fn=None, scope="rpn_bbox_pred")
		return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

	def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
		with tf.variable_scope(name):
			rois, rpn_scores = tf.py_func(tools.proposal_layer, [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
				self._feat_stride, self._anchors, self._num_anchors], [tf.float32, tf.float32])
			rois.set_shape([None, 5])
			rpn_scores.set_shape([None, 1])
		return rois, rpn_scores

	def _anchor_target_layer(self, rpn_cls_score, name):
		with tf.variable_scope(name):
			rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(tools.anchor_target_layer,
				[rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
				[tf.float32, tf.float32, tf.float32, tf.float32])
			rpn_labels.set_shape([1, 1, None, None])
			rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
			rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
			rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])
			rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
			self._anchor_targets["rpn_labels"] = rpn_labels
			self._anchor_targets["rpn_bbox_targets"] = rpn_bbox_targets
			self._anchor_targets["rpn_bbox_inside_weights"] = rpn_bbox_inside_weights
			self._anchor_targets["rpn_bbox_outside_weights"] = rpn_bbox_outside_weights
			self._score_summaries.update(self._anchor_targets)
		return rpn_labels

	def _proposal_target_layer(self, rois, roi_scores, name):
		with tf.variable_scope(name):
			rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(tools.proposal_target_layer,
				[rois, roi_scores, self._gt_boxes, self._num_classes], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
			rois.set_shape([256, 5])
			roi_scores.set_shape([256])
			labels.set_shape([256, 1])
			bbox_targets.set_shape([256, self._num_classes * 4])
			bbox_inside_weights.set_shape([256, self._num_classes * 4])
			bbox_outside_weights.set_shape([256, self._num_classes * 4])
			self._proposal_targets["rois"] = rois
			self._proposal_targets["labels"] = tf.to_int32(labels, name="to_int32")
			self._proposal_targets["bbox_targets"] = bbox_targets
			self._proposal_targets["bbox_inside_weights"] = bbox_inside_weights
			self._proposal_targets["bbox_outside_weights"] = bbox_outside_weights
			self._score_summaries.update(self._proposal_targets)
			return rois, roi_scores

	def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
		with tf.variable_scope(name):
			rois, rpn_scores = tf.py_func(tools.proposal_top_layer, [rpn_cls_prob, rpn_bbox_pred,
				self._im_info, self._feat_stride, self._anchors, self._num_anchors], [tf.float32, tf.float32])
			rois.set_shape([300, 5])
			rpn_scores.set_shape([300, 1])
		return rois, rpn_scores

	def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
		if is_training:
			rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
			rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
			with tf.control_dependencies([rpn_labels]):
				rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
		else:
			rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
		return rois

	def _crop_pool_layer(self, bottom, rois, name):
		with tf.variable_scope(name):
			batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
			bottom_shape = tf.shape(bottom)
			height = (tf.to_float(bottom_shape[1]) - 1.0) * np.float32(self._feat_stride[0])
			width = (tf.to_float(bottom_shape[2]) - 1.0) * np.float32(self._feat_stride[0])
			x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
			y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
			x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
			y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
			pre_pool_size = 7 * 2
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
		return slim.max_pool2d(crops, [2, 2], padding="SAME")

	def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
		pool5 = self._crop_pool_layer(net, rois, "pool5")
		pool5_flat = slim.flatten(pool5, scope="flatten")
		with tf.device("/cpu:0"):
			fc6 = slim.fully_connected(pool5_flat, 4096, scope="fc6")
			if is_training:
				fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope="dropout6")
			fc7 = slim.fully_connected(fc6, 4096, scope="fc7")
			if is_training:
				fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope="dropout7")
		cls_score = slim.fully_connected(fc7, self._num_classes,
			weights_initializer=initializer, trainable=is_training, activation_fn=None, scope="cls_score")
		cls_prob = self._softmax_layer(cls_score, "cls_prob")
		bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4,
			weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope="bbox_pred")
		return cls_score, cls_prob, bbox_prediction

	def build_network(self, sess, is_training=True):
		with tf.variable_scope("vgg_16", "vgg_16"):
			initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
			initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
			net = self.build_head(is_training)
			rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)
			rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)
			cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)
			self._predictions["rpn_cls_score"] = rpn_cls_score
			self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
			self._predictions["rpn_cls_prob"] = rpn_cls_prob
			self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
			self._predictions["cls_score"] = cls_score
			self._predictions["cls_prob"] = cls_prob
			self._predictions["bbox_pred"] = bbox_pred
			self._predictions["rois"] = rois
			self._score_summaries.update(self._predictions)
		return rois, cls_prob, bbox_pred

	def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
		sigma_2 = sigma ** 2
		box_diff = bbox_pred - bbox_targets
		in_box_diff = bbox_inside_weights * box_diff
		abs_in_box_diff = tf.abs(in_box_diff)
		smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1.0 / sigma_2)))
		in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
		out_loss_box = bbox_outside_weights * in_loss_box
		loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
		return loss_box

	def _add_losses(self, sigma_rpn=3.0):
		with tf.variable_scope("loss_" + self._tag):
			rpn_cls_score = tf.reshape(self._predictions["rpn_cls_score_reshape"], [-1, 2])
			rpn_label = tf.reshape(self._anchor_targets["rpn_labels"], [-1])
			rpn_select = tf.where(tf.not_equal(rpn_label, -1))
			rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
			rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
			rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
			rpn_bbox_pred = self._predictions["rpn_bbox_pred"]
			rpn_bbox_targets = self._anchor_targets["rpn_bbox_targets"]
			rpn_bbox_inside_weights = self._anchor_targets["rpn_bbox_inside_weights"]
			rpn_bbox_outside_weights = self._anchor_targets["rpn_bbox_outside_weights"]
			rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
				rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
			cls_score =self._predictions["cls_score"]
			label = tf.reshape(self._proposal_targets["labels"], [-1])
			cross_entropy = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))
			bbox_pred = self._predictions["bbox_pred"]
			bbox_targets = self._proposal_targets["bbox_targets"]
			bbox_inside_weights = self._proposal_targets["bbox_inside_weights"]
			bbox_outside_weights = self._proposal_targets["bbox_outside_weights"]
			loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
			self._losses["cross_entropy"] = cross_entropy
			self._losses["loss_box"] = loss_box
			self._losses["rpn_cross_entropy"] = rpn_cross_entropy
			self._losses["rpn_loss_box"] = rpn_loss_box
			loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
			self._losses["total_loss"] = loss
			self._event_summaries.update(self._losses)
		return loss

	def train_step(self, sess, blobs, train_op):
		feed_dict = {self._image: blobs["data"], self._im_info: blobs["im_info"], self._gt_boxes: blobs["gt_boxes"]}
		rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"], self._losses["rpn_loss_box"],
			self._losses["cross_entropy"], self._losses["loss_box"], self._losses["total_loss"], train_op], feed_dict=feed_dict)
		return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

	def get_variables_to_restore(self, variables, var_keep_dic):
		variables_to_restore = []
		for v in variables:
			if v.name == "vgg_16/fc6/weights:0" or v.name == "vgg_16/fc7/weights:0":
				self._variables_to_fix[v.name] = v
				continue
			if v.name == "vgg_16/conv1/conv1_1/weights:0":
				self._variables_to_fix[v.name] = v
				continue
			if v.name.split(":")[0] in var_keep_dic:
				print("Variables restored: %s" % v.name)
				variables_to_restore.append(v)
		return variables_to_restore

	def fix_variables(self, sess, pretrained_model):
		print("Fixed Vgg16 Layers..")
		with tf.variable_scope("Fix_VGG16"):
			with tf.device("/cpu:0"):
				fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
				fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
				conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
				restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv, "vgg_16/fc7/weights": fc7_conv,
					"vgg_16/conv1/conv1_1/weights": conv1_rgb})
				restorer_fc.restore(sess, pretrained_model)
				sess.run(tf.assign(self._variables_to_fix["vgg_16/fc6/weights:0"], tf.reshape(fc6_conv,
					self._variables_to_fix["vgg_16/fc6/weights:0"].get_shape())))
				sess.run(tf.assign(self._variables_to_fix["vgg_16/fc7/weights:0"], tf.reshape(fc7_conv,
					self._variables_to_fix["vgg_16/fc7/weights:0"].get_shape())))
				sess.run(tf.assign(self._variables_to_fix["vgg_16/conv1/conv1_1/weights:0"], tf.reverse(conv1_rgb, [2])))

	def create_architecture(self, sess, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
		with tf.device("/cpu:0"):
			self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
			self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
			self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
		self._tag = tag
		self._num_classes = num_classes
		self._mode = mode
		self._anchor_scales = anchor_scales
		self._num_scales = len(anchor_scales)
		self._anchor_ratios = anchor_ratios
		self._num_ratios = len(anchor_ratios)
		self._num_anchors = self._num_scales * self._num_ratios
		training = mode == "TRAIN"
		testing = mode == "TEST"
		assert tag != None
		weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
		biases_regularizer = tf.no_regularizer
		with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
			weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer, biases_initializer=tf.constant_initializer(0.0)):
			rois, cls_prob, bbox_pred = self.build_network(sess, training)
		layers_to_output = {"rois": rois}
		layers_to_output.update(self._predictions)
		for var in tf.trainable_variables():
			self._train_summaries.append(var)
		if testing:
			stds = np.tile(np.array((0.1, 0.1, 0.1, 0.1)), (self._num_classes))
			means = np.tile(np.array((0.0, 0.0, 0.0, 0.0)), (self._num_classes))
			self._predictions["bbox_pred"] *= stds
			self._predictions["bbox_pred"] += means
		else:
			self._add_losses()
			layers_to_output.update(self._losses)
		val_summaries = []
		return layers_to_output

	def test_image(self, sess, image, im_info):
		feed_dict = {self._image: image, self._im_info: im_info}
		cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"], self._predictions["cls_prob"],
			self._predictions["bbox_pred"], self._predictions["rois"]], feed_dict=feed_dict)
		return cls_score, cls_prob, bbox_pred, rois


class Train():
	def __init__(self):
		self._xml_path_mask = "D:/program/mask_detect/mask_dataset/label_mask/*"
		self._xml_path_nomask = "D:/program/mask_detect/mask_dataset/label_nomask/*"
		self._file_path = "D:/program/mask_detect/mask_dataset/"
		self._xml_list = glob.glob(self._xml_path_mask)
		self._xml_list.extend(glob.glob(self._xml_path_nomask))
		random.shuffle(self._xml_list)
		self._num_samples = len(self._xml_list)
		self._classes = ['__background__', 'have_mask', 'no_mask']
		self._ckpt_file = "D:/program/mask_detect/vgg16.ckpt"
		self._output_dir = "D:/program/mask_detect/model/"
		self.net = Vgg16()

	def prep_im_for_blob(self, im, pixel_means=np.array([[[102.9801, 115.9465, 122.7717]]]), target_size=600, max_size=1000):
		im = im.astype(np.float32, copy=False)
		im -= pixel_means
		im_shape = im.shape
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		im_scale = float(target_size) / float(im_size_min)
		if np.round(im_scale * im_size_max) > max_size:
			im_scale = float(max_size) / float(im_size_max)
		im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
		return im, im_scale

	def parse_xml(self, xml_file):
		tree = ET.parse(xml_file)
		root = tree.getroot()
		name = root.find("path").text.split("/")
		name_length = len(name)
		image_name = self._file_path + name[name_length-2] + "/" + name[name_length-1]
		image = cv2.imread(image_name)
		image, image_scale = self.prep_im_for_blob(image)
		image = image[np.newaxis, :, :, :]
		image_attitude = [[image.shape[1], image.shape[2], image_scale]]
		category, x_min, y_min, x_max, y_max = [], [], [], [], []
		for object in root.findall("object"):
			category.append(int(self._classes.index(object.find("name").text)))
			x_min.append((float(object.find("bndbox").find("xmin").text) - 1) * image_scale)
			y_min.append((float(object.find("bndbox").find("ymin").text) - 1) * image_scale)
			x_max.append((float(object.find("bndbox").find("xmax").text) - 1) * image_scale)
			y_max.append((float(object.find("bndbox").find("ymax").text) - 1) * image_scale)
		grow_truth = np.vstack((x_min, y_min, x_max, y_max, category)).transpose()
		blobs = {"data": image, "im_info": image_attitude, "gt_boxes": grow_truth}
		return blobs

	def get_variables_in_checkpoint_file(self, file_name):
		reader = pywrap_tensorflow.NewCheckpointReader(file_name)
		var_to_shape_map = reader.get_variable_to_shape_map()
		return var_to_shape_map

	def train(self):
		sess = tf.Session()
		with sess.graph.as_default():
			tf.set_random_seed(3)
			layers = self.net.create_architecture(sess, "TRAIN", 3, tag="default")
			loss = layers["total_loss"]
			lr = tf.Variable(0.001, trainable=False)
			momentum = 0.9
			optimizer = tf.train.MomentumOptimizer(lr, momentum)
			gvs = optimizer.compute_gradients(loss)
			final_gvs = []
			with tf.variable_scope("Gradient_Mult"):
				for grad, var in gvs:
					scale = 1.0
					if "/biases:" in var.name:
						scale *= 2.0
					if not np.allclose(scale, 1.0):
						grad = tf.multiply(grad, scale)
					final_gvs.append((grad, var))
			train_op = optimizer.apply_gradients(final_gvs)
			self.saver = tf.train.Saver(max_to_keep=100000)
		variables = tf.global_variables()
		sess.run(tf.global_variables_initializer())
		var_keep_dic = self.get_variables_in_checkpoint_file(self._ckpt_file)
		variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self._ckpt_file)
		print("Load.")
		self.net.fix_variables(sess, self._ckpt_file)
		print("Fixed.")
		sess.run(tf.assign(lr, 0.001))
		last_compute_pic = 0
		iter = last_compute_pic + 1
		timer = Timer()
		while iter < 40000 + 1:
			if iter == 30000 + 1:
				sess.run(tf.assign(lr, 0.001 * 0.1))
			timer.tic()
			xml_name = self._xml_list[last_compute_pic]
			blobs = self.parse_xml(xml_name)
			try:
				rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op)
			except:
				print("image invalid, skipping")
				continue
			timer.toc()
			if iter % 10 == 0:
				print("Iter %d / %d, total_loss: %.6f\n >>> rpn_loss_cls: %.6f\n"
					">>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n" % \
					(iter, 40000, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
				print("speed: {:.3f}s / iter".format(timer.average_time))
			last_compute_pic += 1
			if last_compute_pic == self._num_samples:
				last_compute_pic = 0
			if iter % 600 == 0:
				self.snapshot(sess, iter)
			iter += 1

	def snapshot(self, sess, iter):
		net = self.net
		file_name = os.path.join(self._output_dir, "model" + str(iter) + ".ckpt")
		self.saver.save(sess, file_name)
		print("Wrote snapshot to: " + file_name)
		return file_name


if __name__ == "__main__":
	trainer = Train()
	trainer.train()