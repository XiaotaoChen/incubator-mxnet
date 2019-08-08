import sys

import mxnet as mx
import numpy as np
eps = 1e-5


def residual_unit_int8(data, channel, num_filter, stride, dim_match, name, bottle_neck=True,
                  bn_mom=0.9, workspace=512, memonger=False, is_train=True, quant_mod='minmax'):
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        relu1_q = mx.sym.Quantization_int8(data=act1, is_weight=False,
                                           ema_decay=0.99, delay_quant=0, is_train=is_train,
                                           name=name + "_relu1_quant", quant_mod=quant_mod)
        # weight to be quantized
        weight_conv1 = mx.sym.Variable(name=name + "_conv1_weight", shape=(num_filter // 4, channel, 1, 1))
        weight_conv1_q = mx.sym.Quantization_int8(weight_conv1, name=name + "_conv1_weight_quant",
                                                  quant_mod=quant_mod)

        conv1 = mx.sym.Convolution(data=relu1_q, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0), no_bias=True, workspace=workspace,
                                   name=name + '_conv1', weight=weight_conv1_q)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        relu2_q = mx.sym.Quantization_int8(data=act2, is_weight=False,
                                           ema_decay=0.99, delay_quant=0, is_train=is_train,
                                           name=name + "_relu2_quant", quant_mod=quant_mod)
        weight_conv2 = mx.sym.Variable(name=name + "_conv2_weight", shape=(num_filter // 4, num_filter // 4, 3, 3))
        weight_conv2_q = mx.sym.Quantization_int8(weight_conv2, name=name + "_conv2_weight_quant",
                                                  quant_mod=quant_mod)

        conv2 = mx.sym.Convolution(data=relu2_q, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1), no_bias=True, workspace=workspace,
                                   name=name + '_conv2', weight=weight_conv2_q)
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')

        relu3_q = mx.sym.Quantization_int8(data=act3, is_weight=False,
                                           ema_decay=0.99, delay_quant=0, is_train=is_train,
                                           name=name + "_relu3_quant", quant_mod=quant_mod)
        weight_conv3 = mx.sym.Variable(name=name + "_conv3_weight", shape=(num_filter, num_filter // 4, 1, 1))
        weight_conv3_q = mx.sym.Quantization_int8(weight_conv3, name=name + "_conv3_weight_quant",
                                                  quant_mod=quant_mod)

        conv3 = mx.sym.Convolution(data=relu3_q, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace,
                                   name=name + '_conv3', weight=weight_conv3_q)
        if dim_match:
            shortcut = data
        else:
            weight_sc = mx.sym.Variable(name=name + "_sc_weight", shape=(num_filter, channel, 1, 1))
            weight_sc_q = mx.sym.Quantization_int8(weight_sc, name=name + "_sc_weight_quant", quant_mod=quant_mod)

            shortcut = mx.sym.Convolution(data=relu1_q, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc', weight=weight_sc_q)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=eps, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        relu1_q = mx.sym.Quantization_int8(data=act1, is_weight=False,
                                           ema_decay=0.99, delay_quant=0, is_train=is_train, quant_mod=quant_mod)
        # weight to be quantized
        weight_conv1 = mx.sym.Variable(name=name + "_conv1_weight", shape=(num_filter, channel, 3, 3))
        weight_conv1_q = mx.sym.Quantization_int8(weight_conv1, quant_mod=quant_mod)

        conv1 = mx.sym.Convolution(data=relu1_q, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1',
                                   weight=weight_conv1_q)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=eps, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        relu2_q = mx.sym.Quantization_int8(data=act2, is_weight=False,
                                           ema_decay=0.99, delay_quant=0, is_train=is_train, quant_mod=quant_mod)
        weight_conv2 = mx.sym.Variable(name=name + "_conv2_weight", shape=(num_filter, num_filter, 3, 3))
        weight_conv2_q = mx.sym.Quantization_int8(weight_conv2, quant_mod=quant_mod)

        conv2 = mx.sym.Convolution(data=relu2_q, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2',
                                   weight=weight_conv2_q)
        if dim_match:
            shortcut = data
        else:
            weight_sc = mx.sym.Variable(name=name + "_sc_weight", shape=(num_filter, channel, 1, 1))
            weight_sc_q = mx.sym.Quantization_int8(weight_sc, quant_mod=quant_mod)

            shortcut = mx.sym.Convolution(data=relu1_q, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc',
                                          weight=weight_sc_q)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return conv2 + shortcut


def resnet_int8(units, num_stage, filter_list, num_classes, data_type, bottle_neck=True,
           bn_mom=0.9, workspace=512, memonger=False, grad_scale=1.0, is_train=True, quant_mod='minmax'):
    num_unit = len(units)
    assert (num_unit == num_stage)

    data = mx.sym.Variable(name='data')
    if data_type == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif data_type == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    weight = mx.sym.Variable(name="conv0_weight", shape=(filter_list[0], 3, 7, 7))
    weight_q = mx.sym.Quantization_int8(weight, name="conv0_weight_quant", quant_mod=quant_mod)
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace, weight=weight_q)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.sym.Quantization_int8(data=body, is_weight=False,
                                    ema_decay=0.99, delay_quant=0, is_train=is_train,
                                           name="relu0_quant", quant_mod=quant_mod)

    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    for i in range(num_stage):
        body = residual_unit_int8(body, filter_list[i], filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit_int8(body, filter_list[i + 1], filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    relu1 = mx.sym.Quantization_int8(data=relu1, is_weight=False,
                                    ema_decay=0.99, delay_quant=0, is_train=is_train,
                                    name="relu1_quant", quant_mod=quant_mod)


    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if data_type == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax', grad_scale=grad_scale)
    else:
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls

def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256, dtype='float32', **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet_int8(units = units,
                  num_stage   = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  data_type   = dtype,
                  workspace   = conv_workspace)