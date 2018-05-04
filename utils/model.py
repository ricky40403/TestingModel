from keras.applications.imagenet_utils import _obtain_input_shape
import keras.layers as KL
import keras.models as KM
from utils.BilinearUpSampling import *
from keras.regularizers import l2
############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

# Atrous-Convolution version of residual blocks
def atrous_identity_block(kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay), use_bias=True)(input_tensor)
        x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay), use_bias=True)(x)
        x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay), use_bias=True)(x)
        x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        return x
    return f

def atrous_conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay), use_bias=True)(input_tensor)
        x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay), use_bias=True)(x)
        x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay), use_bias=True)(x)
        x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay), use_bias=True)(input_tensor)
        shortcut = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)
        return x
    return f

def ASPP(input_map,rate):
    
    
    P_1 = KL.Conv2D(256, (1, 1), activation= "relu", padding='same')(input_map)

    
    P_3_1 = KL.Conv2D(256, (3, 3), activation= "relu", padding='same', dilation_rate=(rate * 3, rate * 3))(input_map)
    P_3_2 = KL.Conv2D(256, (3, 3), activation= "relu", padding='same', dilation_rate=(rate * 6, rate * 6))(input_map)
    P_3_3 = KL.Conv2D(256, (3, 3), activation= "relu", padding='same', dilation_rate=(rate * 9, rate * 9))(input_map)

        
    P_5 = KL.AveragePooling2D(padding='same')(input_map)
    P_5 = KL.Conv2D(256, (1, 1), activation= "relu", padding="SAME")(P_5)
    print(P_5)
    inputs_size = tf.shape(input_map)[1:3]
    P_5 = tf.image.resize_bilinear(P_5, inputs_size, name='upsample')
    
#     P_5 = BilinearUpSampling2D(target_size=(input_map.shape[1], input_map.shape[2]))(P_5)
    
    P = KL.concatenate([P_1,P_3_1,P_3_2,P_3_3])
    P = KL.Conv2D(256, (1, 1), padding='same')(P)
    P = KL.BatchNormalization(axis=-1)(P)
    P = KL.Activation("relu")(P)
    
    
    return P


def resnet_graph(input_image, architecture, input_shape = None,batch_momentum=0.9, stage5=False,include_top=True,weights='imagenet'):
    assert architecture in ["resnet50", "resnet101"]
    pool_size = (2,2)
    if architecture == "resnet50":
        use_bias=True
    else:
        use_bias=False

    input_shape = _obtain_input_shape(input_shape,
                                     default_size=224,
                                     min_size=197,
                                     data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    print("input_image : {} ".format(input_image))
    
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name='bn_conv1')(x)
    C = x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    
    print("C1 : {} ".format(C1))
    
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_bias=use_bias)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', use_bias=use_bias)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', use_bias=use_bias)
    
    #C2 , mask_2 = MaxPoolingWithArgmax2D((2,2))(x)
    print("C2 : {} ".format(C2))
    
    # Stage 3
    C3 = []
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', use_bias=use_bias)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', use_bias=use_bias)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', use_bias=use_bias)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', use_bias=use_bias)
    
    print("C3 : {} ".format(C3))
    
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', use_bias=use_bias)
#     x = atrous_conv_block(3, [256, 256, 1024], stage=4, block='a', atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        C4 = x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), use_bias=use_bias)
#         C4 = x = atrous_identity_block(3, [256, 256, 1024], stage=4, block=chr(98 + i), atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    print("C4 : {} ".format(C4))
    
    # Stage 5   
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', use_bias=use_bias)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', use_bias=use_bias)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', use_bias=use_bias)
#         x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
#         x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
#         C5 = x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
            
    else:
        C5 = None
        
#     C5 , mask_5 = MaxPoolingWithArgmax2D((2,2))(x)
    print("C5 : {} ".format(C5))
    
#     C6 = ASPP(C5,2)
#     print("C6 : {} ".format(C6))
    
    
#     xfc = KL.AveragePooling2D((7,7), name='avg_pool')(x)
#     xfc = KL.Flatten()(xfc)
#     print("xfc Flatten : {} ".format(xfc))
#     xfc_out = xfc
    
#     xfc = KL.Dense(2048, activation='softmax', name='fc1000')(xfc)
#     xfc = KL.Dense(1000, activation='softmax', name='fc1000')(xfc)
#     print("xfc Dense : {} ".format(xfc))    

    model = KM.Model(input_image,C5)
    if architecture =="resnet50":
        weights_path = 'keras_resnet50_weight.hdf5'
    else:
        weights_path = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(weights_path,by_name=True)
    
    return model,C,C1,C2,C3,C4,C5





