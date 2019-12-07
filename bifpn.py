import mxnet as mx

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name))
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}_{}".format(name, act_type))
    return relu


def bifpn(layers, dest_channels):
    compose_layers = [i for i in range(0, len(layers))]

    trans_layers = []
    for i, layer in enumerate(layers):
        layer_ = conv_act_layer(layer, name='transition_%i' % (i), num_filter=dest_channels, kernel=(3, 3), pad=(1, 1),
                                stride=(1, 1), act_type='relu')
        trans_layers.append(layer_)

    for i in range(len(trans_layers) - 1, -1, -1):  # enumerate(trans_layers):
        layer = trans_layers[i]
        if (i == 0):
            compose_layers[i] = None

        elif (i == (len(layers) - 1)):  # last layer ,just add
            compose_layers[i] = None

        else:

            # layer_ = conv_act_layer(layer, 'com_1x1_%i' % (i), layers_channel[i])

            if (compose_layers[i + 1] is None):
                prev_layer = trans_layers[i + 1]
                # deconv = mx.symbol.UpSampling(next_layer,scale=2,sample_type='nearest',name='up_layer_%i'%(i))
                # compose_nf = layers_channel[i+1]//2
                # print(compose_nf)
                deconv = mx.symbol.Deconvolution(data=prev_layer, num_filter=dest_channels, kernel=(4, 4),
                                                 stride=(2, 2),
                                                 pad=(1, 1), name='com_%d_deconv' % (i))
                # deconv_bn = mx.symbol.BatchNorm(data=deconv, name="deconv_bn{}".format(i))
                deconv_relu = mx.symbol.Activation(data=deconv, act_type='relu', name="com_{}_deconv_relu".format(i))
                # com_layer = mx.symbol.concat(*[deconv_relu, layer_], name='compose_up_%i' % (i))
                com_layer = deconv_relu + layer
                com_layer_conv = conv_act_layer(com_layer, name='com_layer_%i' % (i), num_filter=dest_channels,
                                                kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
                compose_layers[i] = com_layer_conv
            else:
                prev_layer = compose_layers[i + 1]
                # deconv = mx.symbol.UpSampling(next_layer,scale=2,sample_type='nearest',name='up_layer_%i'%(i))
                # compose_nf = layers_channel[i+1]//2
                # print(compose_nf)
                deconv = mx.symbol.Deconvolution(data=prev_layer, num_filter=dest_channels, kernel=(4, 4),
                                                 stride=(2, 2), pad=(1, 1), name='com_%d_deconv' % (i))
                # deconv_bn = mx.symbol.BatchNorm(data=deconv, name="deconv_bn{}".format(i))
                deconv_relu = mx.symbol.Activation(data=deconv, act_type='relu', name="com_{}_deconv_relu".format(i))
                com_layer = deconv_relu + layer
                com_layer_conv = conv_act_layer(com_layer, name='com_layer_%i' % (i), num_filter=dest_channels,
                                                kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
                compose_layers[i] = com_layer_conv

            # com_layer = layer + deconv_relu

            # deconv_crop = mx.symbol.Crop(*[deconv, layer], name='FPN_crop%d' % (i))
            # com_layer = mx.symbol.Concat(*[layer, deconv_crop], name='FPN_Concat%d' % (i))

            # com_layer = up_layer + layer
            # new_filter = layers_channel[i] + 32
            # compose_num_filters.append(new_filter)

    compose_layers_2 = [i for i in range(0, 5)]
    compose_num_filters_2 = []

    for i in range(0, len(compose_layers)):

        if (i == 0):
            layer = trans_layers[i]

            next_layer = compose_layers[i + 1]

            conv_half = mx.symbol.Deconvolution(data=next_layer, num_filter=dest_channels, kernel=(4, 4),
                                                stride=(2, 2),
                                                pad=(1, 1), name='com_%d_deconv_2' % (i))
            # deconv_bn = mx.symbol.BatchNorm(data=deconv, name="deconv_bn{}".format(i))
            conv_half_relu = mx.symbol.Activation(data=conv_half, act_type='relu',
                                                  name="com_{}_deconv_relu_2".format(i))
            com_layer = conv_half_relu + layer
            com_layer_conv = conv_act_layer(com_layer, name='com_layer_%i_2' % (i), num_filter=dest_channels,
                                            kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
            compose_layers_2[i] = com_layer_conv
        elif (i == (len(layers) - 1)):
            layer = trans_layers[i]

            next_layer = compose_layers_2[i - 1]
            deconv = mx.symbol.Convolution(data=next_layer, num_filter=dest_channels, kernel=(3, 3),
                                           stride=(2, 2),
                                           pad=(1, 1), name='com_%d_deconv_2' % (i))
            # deconv_bn = mx.symbol.BatchNorm(data=deconv, name="deconv_bn{}".format(i))
            deconv_relu = mx.symbol.Activation(data=deconv, act_type='relu', name="com_{}_deconv_relu_2".format(i))

            com_layer = deconv_relu + layer
            com_layer_conv = conv_act_layer(com_layer, name='com_layer_%i_2' % i, num_filter=dest_channels,
                                            kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
            compose_layers_2[i] = com_layer_conv

        else:
            # No.1
            layer_ = trans_layers[i]
            # No.2
            com_layer = compose_layers[i]

            # No.3
            com_layer_2 = compose_layers_2[i - 1]

            conv_half = mx.symbol.Convolution(data=com_layer_2, num_filter=dest_channels, kernel=(3, 3),
                                              stride=(2, 2),
                                              pad=(1, 1), name='com_%d_deconv_2' % (i))
            # deconv_bn = mx.symbol.BatchNorm(data=deconv, name="deconv_bn{}".format(i))
            conv_half_relu = mx.symbol.Activation(data=conv_half, act_type='relu',
                                                  name="com_{}_deconv_relu_2".format(i))
            com_layer = layer_ + com_layer + conv_half_relu
            com_layer_conv = conv_act_layer(com_layer, name='com_layer_%i_2' % i, num_filter=dest_channels,
                                            kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')

            compose_layers_2[i] = com_layer_conv
    print('-------------------------bifpn--------------------')
    for i in range(len(compose_layers_2)):
        arg_shape, output_shape, aux_shape = compose_layers_2[i].infer_shape(data=(1, 3, 512, 512))
        print('layers' + str(i) + ',output_shape, ', output_shape)

    return compose_layers_2
