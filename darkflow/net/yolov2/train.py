import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math

def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']
    if self.FLAGS['classLoss'] == 'default':
        size_adjusted_prob = [-1, H*W, B, C]
    elif self.FLAGS['classLoss'] == 'cross':
        size_adjusted_prob = [-1, C]
    elif self.FLAGS['classLoss'] == 'focal':
        size_adjusted_prob = [-1, C]
    elif self.FLAGS['classLoss'] == 'lr':
        size_adjusted_prob = [-1, C]
    else:
        print('No such classLoss! Plesse choose one from default, cross, focal and lr')
        exit()

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))
    print('\tclassLoss = {}'.format(self.FLAGS['classLoss']))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1) #true class
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H*W, B, 4])
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = net_out_reshape[:, :, :, :, 5:]
    adjusted_prob = tf.reshape(adjusted_prob, size_adjusted_prob)

    # adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)
    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)

    wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:,:,:,0] * wh[:,:,:,1]
    centers = coords[:,:,:,0:2]
    floor = centers - (wh * .5)
    ceil  = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    # true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
    true = tf.concat([_coord, tf.expand_dims(confs, 3)], 3)
    # wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3)], 3)

    probs_class = tf.reshape(_probs, [-1, C])
    if self.FLAGS['classLoss'] == 'default':
        adjusted_prob = tf.nn.softmax(adjusted_prob)
        loss_class = tf.pow(adjusted_prob - _probs, 2)
        loss_class = tf.multiply(loss_class, proid)
        loss_class = tf.reshape(loss_class, [-1, H*W*B*C])
    elif self.FLAGS['classLoss'] == 'cross':
        proid = tf.reshape(proid, [-1, C])
        loss_class = tf.nn.softmax_cross_entropy_with_logits(logits=adjusted_prob, labels=probs_class)
        proid = tf.reduce_mean(proid, axis=-1)
        loss_class = tf.multiply(loss_class, proid)
        loss_class = tf.reshape(loss_class, [-1, H*W*B])
    elif self.FLAGS['classLoss'] == 'focal':
        proid = tf.reshape(proid, [-1, C])
        loss_class = focal_loss(logits=adjusted_prob, labels=probs_class, gamma=2, alpha=0.25, proid=proid)
        loss_class = tf.reshape(loss_class, [-1, H*W*B])
    elif self.FLAGS['classLoss'] == 'lr':
        proid = tf.reshape(proid, [-1, C])
        loss_class = tf.nn.sigmoid_cross_entropy_with_logits(logits=adjusted_prob, labels=probs_class)
        loss_class = tf.multiply(loss_class, proid)
        loss_class = tf.reduce_sum(loss_class, axis=-1)
        loss_class = tf.reshape(loss_class, [-1, H*W*B])

    loss_class = tf.reduce_sum(loss_class, 1)


    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = multiply
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1)])
    loss = tf.reduce_sum(loss, 1)
    loss = loss + loss_class
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)


def focal_loss(logits, labels, gamma, alpha, proid):
    per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    per_entry_cross_ent = tf.multiply(per_entry_cross_ent, proid)
    prediction_probabilities = tf.nn.sigmoid(logits)
    p_t = ((labels * prediction_probabilities) +
           (1 - labels) * (1 - prediction_probabilities))
    alpha_weight_factor = (labels * alpha + (1 - labels) * (1 - alpha))
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    focal_cross_entropy_loss = tf.reduce_sum(modulating_factor * alpha_weight_factor * per_entry_cross_ent, axis=-1)
    return focal_cross_entropy_loss
