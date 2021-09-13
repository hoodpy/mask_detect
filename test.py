import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import tools
import glob
from network import Vgg16
from network import Timer


classes = ['__background__', 'have_mask', 'no_mask']
model_path = "D:/program/mask_detect/model/model1200.ckpt"
file_path = "D:/program/mask_detect/test_image"
image_list = os.listdir(file_path)
n_classes = len(classes)

def _get_image_blob(im, pixel_means=np.array([[[102.9801, 115.9465, 122.7717]]]), target_size=600, max_size=1000):
	im_orig = im.astype(np.float32, copy=True)
	im_orig -= pixel_means
	im_shape = np.shape(im_orig)
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])
	im_scale = float(target_size) / float(im_size_min)
	if np.round(im_size_max * im_scale) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
	im = im[np.newaxis, :, :, :]
	return im, im_scale

def _get_blobs(im):
	blobs = {}
	blobs["data"], im_scales = _get_image_blob(im)
	return blobs, im_scales

def im_detect(sess, net, im):
	blobs, im_scales = _get_blobs(im)
	im_blob = blobs["data"]
	blobs["im_info"] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales]], dtype=np.float32)
	_, scores, bbox_pred, rois = net.test_image(sess, blobs["data"], blobs["im_info"])
	boxes = rois[:, 1:5] / im_scales
	scores = np.reshape(scores, [scores.shape[0], -1])
	bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
	pred_boxes = tools.bbox_transform_inv(boxes, bbox_pred)
	pred_boxes = tools.clip_boxes(pred_boxes, im.shape)
	return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.5):
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		return
	im = im[:, :, (2, 1, 0)]
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.imshow(im, aspect="equal")
	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]
		ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor="red", linewidth=3))
		ax.text(bbox[0], bbox[1] - 2, "{:s}{:.3f}".format(class_name, score), bbox=dict(facecolor="blue", alpha=0.5), fontsize=14, color="white")
	ax.set_title("{} detections with p({} | box) >= {:.1f}".format(class_name, class_name, thresh))
	plt.axis("off")
	plt.tight_layout()
	plt.draw()

def demo(sess, net, image_name):
	im = cv2.imread(image_name)
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(sess, net, im)
	timer.toc()
	print("Detection took {:.3f}s for {:d} object proposals".format(timer.total_time, boxes.shape[0]))
	CONF_THRESH = 0.1
	NMS_THRESH = 0.1
	for cls_ind, cls in enumerate(classes[1:]):
		cls_ind += 1
		cls_boxes = boxes[:, 4 * cls_ind : 4 * (cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = tools.nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		vis_detections(im, cls, dets, thresh=CONF_THRESH)
		

if __name__ == "__main__":
	sess = tf.Session()
	net = Vgg16()
	net.create_architecture(sess, "TEST", n_classes, tag="default")
	saver = tf.train.Saver()
	saver.restore(sess, model_path)
	print("Load Network " + model_path)
	for image_name in image_list:
		the_file_path = os.path.join(file_path, image_name)
		demo(sess, net, the_file_path)
	plt.show()