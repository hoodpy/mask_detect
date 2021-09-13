import tensorflow as tf
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from cython_bbox import bbox_overlaps


def _whctrs(anchor):
	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	x_ctr = anchor[0] + 0.5 * (w - 1)
	y_ctr = anchor[1] + 0.5 * (h - 1)
	return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
	ws = ws[:, np.newaxis]
	hs = hs[:, np.newaxis]
	anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
						y_ctr - 0.5 * (hs - 1),
						x_ctr + 0.5 * (ws - 1),
						y_ctr + 0.5 * (hs - 1)))
	return anchors

def _ratio_enum(anchor, ratios):
	w, h, x_ctr, y_ctr = _whctrs(anchor)
	size = w * h
	size_ratios = size / ratios
	ws = np.round(np.sqrt(size_ratios))
	hs = np.round(ws * ratios)
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

def _scale_enum(anchors, scales):
	w, h, x_ctr, y_ctr = _whctrs(anchors)
	ws = w * scales
	hs = h * scales
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
	base_anchor = np.array([1, 1, base_size, base_size]) - 1
	ratio_anchors = _ratio_enum(base_anchor, ratios)
	anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
	return anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
	anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
	A = anchors.shape[0]
	shift_x = np.arange(0, width) * feat_stride
	shift_y = np.arange(0, height) * feat_stride
	shift_x, shift_y = np.meshgrid(shift_x, shift_y)
	shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
	K = shifts.shape[0]
	anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
	anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
	length = np.int32(anchors.shape[0])
	return anchors, length

def bbox_transform_inv(boxes, deltas):
	if boxes.shape[0] == 0:
		return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
	boxes = boxes.astype(deltas.dtype, copy=False)
	widths = boxes[:, 2] - boxes[:, 0] + 1.0
	heights = boxes[:, 3] - boxes[:, 1] + 1.0
	ctr_x = boxes[:, 0] + 0.5 * widths
	ctr_y = boxes[:, 1] + 0.5 * heights
	dx, dy, dw, dh = deltas[:, 0::4], deltas[:, 1::4], deltas[:, 2::4], deltas[:, 3::4]
	pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
	pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
	pred_w = np.exp(dw) * widths[:, np.newaxis]
	pred_h = np.exp(dh) * heights[:, np.newaxis]
	pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
	pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
	pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
	pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
	pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
	return pred_boxes

def bbox_transform(ex_rois, gt_rois):
	ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
	ex_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
	ex_ctr_x = ex_rois[:, 0] + ex_widths * 0.5
	ex_ctr_y = ex_rois[:, 1] + ex_heights * 0.5
	gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
	gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
	gt_ctr_x = gt_rois[:, 0] + gt_widths * 0.5
	gt_ctr_y = gt_rois[:, 1] + gt_heights * 0.5
	targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
	targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
	targets_dw = np.log(gt_widths / ex_widths)
	targets_dh = np.log(gt_heights / ex_heights)
	targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
	return targets

def clip_boxes(boxes, im_shape):
	boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
	boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
	boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
	boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
	return boxes

def py_cpu_nms(dets, thresh):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 +1)
		h = np.maximum(0.0, yy2 - yy1 +1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep

def nms(dets, thresh, force_cpu=False):
	if dets.shape[0] == 0:
		return []
	return py_cpu_nms(dets=dets, thresh=thresh)

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
	if type(cfg_key) == bytes:
		cfg_key = cfg_key.decode("utf-8")
	if cfg_key == "TRAIN":
		pre_nms_topN, post_nms_topN, nms_thresh = 12000, 2000, 0.7
	else:
		pre_nms_topN, post_nms_topN, nms_thresh = 6000, 300, 0.7
	im_info = im_info[0]
	scores = rpn_cls_prob[:, :, :, num_anchors:]
	rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
	scores = scores.reshape((-1, 1))
	proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
	proposals = clip_boxes(proposals, im_info[:2])
	order = scores.ravel().argsort()[::-1]
	if pre_nms_topN > 0:
		order = order[:pre_nms_topN]
	proposals = proposals[order, :]
	scores = scores[order]
	keep = nms(np.hstack((proposals, scores)), nms_thresh)
	if post_nms_topN > 0:
		keep = keep[:post_nms_topN]
	proposals = proposals[keep, :]
	scores = scores[keep]
	batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
	blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
	return blob, scores

def _compute_targets(ex_rois, gt_rois):
	assert ex_rois.shape[0] == gt_rois.shape[0]
	assert ex_rois.shape[1] == 4
	assert gt_rois.shape[1] == 5
	return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _unmap(data, count, inds, fill=0):
	if len(data.shape) == 1:
		ret = np.empty((count,), dtype=np.float32)
		ret.fill(fill)
		ret[inds] = data
	else:
		ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
		ret.fill(fill)
		ret[inds, :] = data
	return ret

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
	A = num_anchors
	total_anchors = all_anchors.shape[0]
	K = total_anchors / num_anchors
	im_info = im_info[0]
	_allowed_border = 0
	height, width = rpn_cls_score.shape[1:3]
	inds_inside = np.where((all_anchors[:, 0] >= -_allowed_border) & (all_anchors[:, 1] >= - _allowed_border) &
		(all_anchors[:, 2] < im_info[1] + _allowed_border) & (all_anchors[:, 3] < im_info[0] + _allowed_border))[0]
	anchors = all_anchors[inds_inside, :]
	labels = np.empty((len(inds_inside),), dtype=np.float32)
	labels.fill(-1)
	overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
	argmax_overlaps = overlaps.argmax(axis=1)
	max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
	gt_argmax_overlaps = overlaps.argmax(axis=0)
	gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
	gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
	labels[max_overlaps < 0.3] = 0
	labels[gt_argmax_overlaps] = 1
	labels[max_overlaps > 0.7] = 1
	num_fg = int(256 * 0.5)
	fg_inds = np.where(labels == 1)[0]
	if len(fg_inds) > num_fg:
		disable_inds = npr.choice(fg_inds, size=len(fg_inds) - num_fg, replace=False)
		labels[disable_inds] = -1
	num_bg = 256 - np.sum(labels ==1)
	bg_inds = np.where(labels == 0)[0]
	if len(bg_inds) > num_bg:
		disable_inds = npr.choice(bg_inds, size=len(bg_inds) - num_bg, replace=False)
		labels[disable_inds] = -1
	bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
	bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
	bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))
	bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
	num_examples = np.sum(labels >= 0)
	positive_weights = np.ones((1, 4)) * 1.0 / num_examples
	negative_weights = np.ones((1, 4)) * 1.0 / num_examples
	bbox_outside_weights[labels == 1, :] = positive_weights
	bbox_outside_weights[labels == 0, :] = negative_weights
	labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
	bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
	bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
	bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
	labels = labels.reshape((1, height, width, A)).transpose((0, 3, 1, 2))
	labels = labels.reshape((1, 1, height * A, width))
	rpn_labels = labels
	bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
	rpn_bbox_targets = bbox_targets
	bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
	rpn_bbox_inside_weights = bbox_inside_weights
	bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
	rpn_bbox_outside_weights = bbox_outside_weights
	return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def _compute_targets_label(ex_rois, gt_rois, labels):
	assert ex_rois.shape[0] == gt_rois.shape[0]
	assert ex_rois.shape[1] == 4
	assert gt_rois.shape[1] == 4
	targets = bbox_transform(ex_rois, gt_rois)
	targets = (targets - np.array((0.0, 0.0, 0.0, 0.0))) / np.array((0.1, 0.1, 0.1, 0.1))
	return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _get_bbox_regression_labels(bbox_target_data, num_classes):
	clss = bbox_target_data[:, 0]
	bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
	bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
	inds = np.where(clss > 0)[0]
	for ind in inds:
		cls = clss[ind]
		start = int(4 * cls)
		end = start + 4
		bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
		bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
	return bbox_targets, bbox_inside_weights

def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
	overlaps = bbox_overlaps(
		np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float), np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
	gt_assignment = overlaps.argmax(axis=1)
	max_overlaps = overlaps.max(axis=1)
	labels = gt_boxes[gt_assignment, 4]
	fg_inds = np.where(max_overlaps >= 0.5)[0]
	bg_inds = np.where((max_overlaps < 0.5) & (max_overlaps >= 0.1))[0]
	if fg_inds.size > 0 and bg_inds.size > 0:
		fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
		fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
		bg_rois_per_image = rois_per_image - fg_rois_per_image
		to_replace = bg_inds.size < bg_rois_per_image
		bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
	elif fg_inds.size > 0:
		to_replace = fg_inds.size < rois_per_image
		fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
		fg_rois_per_image = rois_per_image
	elif bg_inds.size > 0:
		to_replace = bg_inds.size < rois_per_image
		bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
		fg_rois_per_image = 0
	else:
		raise Exception()
	keep_inds = np.append(fg_inds, bg_inds)
	labels = labels[keep_inds]
	labels[int(fg_rois_per_image):] = 0
	rois = all_rois[keep_inds]
	roi_scores = all_scores[keep_inds]
	bbox_target_data = _compute_targets_label(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
	bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)
	return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
	all_rois = rpn_rois
	all_scores = rpn_scores
	num_images = 1
	rois_per_image = 256 / num_images
	fg_rois_per_image = np.round(0.25 * rois_per_image)
	labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
		all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes)
	rois = rois.reshape(-1, 5)
	roi_scores = roi_scores.reshape(-1)
	labels = labels.reshape(-1, 1)
	bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
	bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
	bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
	return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
	rpn_top_n = 300
	im_info = im_info[0]
	scores = rpn_cls_prob[:, :, :, num_anchors:]
	rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
	scores = scores.reshape((-1, 1))
	length = scores.shape[0]
	if length < rpn_top_n:
		top_inds = npr.choice(length, size=rpn_top_n, replace=True)
	else:
		top_inds = scores.argsort(0)[::-1]
		top_inds = top_inds[:rpn_top_n]
		top_inds = top_inds.reshape(rpn_top_n,)
	anchors = anchors[top_inds, :]
	rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
	scores = scores[top_inds]
	proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
	proposals = clip_boxes(proposals, im_info[:2])
	batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
	blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
	return blob, scores