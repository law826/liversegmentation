#%% Generalized Block Model
import numpy as np
from keras.layers import Input, Cropping2D, Conv2D
from keras.layers import concatenate, BatchNormalization
from keras.layers import Conv2DTranspose, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import ELU
from keras.models import Model
init = 'glorot_normal'
def BlockModel(input_shape,filt_num=16,numBlocks=3):
    lay_input = Input(shape=(input_shape[1:]),name='input_layer')
        
     #calculate appropriate cropping
    mod = np.mod(input_shape[1:3],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[1:3]-padamt)
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    if minsize<4:
        raise ValueError('Too small of input for this many blocks. Use fewer blocks or larger input')
    
    crop = Cropping2D(cropping=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_input)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',kernel_initializer=init,name='Conv1_{}'.format(rr))(crop)
    lay_conv1 = BatchNormalization()(lay_conv1)
    lay_conv1 = ELU()(lay_conv1)
    lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv3_{}'.format(rr))(crop)
    lay_conv3 = BatchNormalization()(lay_conv3)
    lay_conv3 = ELU()(lay_conv3)
    lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv51_{}'.format(rr))(crop)
    lay_conv51 = ELU()(lay_conv51)
    lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv52_{}'.format(rr))(lay_conv51)
    lay_conv52 = BatchNormalization()(lay_conv52)
    lay_conv52 = ELU()(lay_conv52)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',kernel_initializer=init,name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=init,name='ConvStride_{}'.format(rr))(lay_act)
    bn = BatchNormalization()(lay_stride)
    lay_act = ELU(name='elu{}_2'.format(rr))(bn)
    act_list = [lay_act]
    
    # contracting blocks 2-n 
    for rr in range(2,numBlocks+1):
        lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',kernel_initializer=init,name='Conv1_{}'.format(rr))(lay_act)
        lay_conv1 = BatchNormalization()(lay_conv1)
        lay_conv1 = ELU()(lay_conv1)
        lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv3_{}'.format(rr))(lay_act)
        lay_conv3 = BatchNormalization()(lay_conv3)
        lay_conv3 = ELU()(lay_conv3)
        lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv51_{}'.format(rr))(lay_act)
        lay_conv51 = ELU()(lay_conv51)
        lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv52_{}'.format(rr))(lay_conv51)
        lay_conv52 = BatchNormalization()(lay_conv52)
        lay_conv52 = ELU()(lay_conv52)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',kernel_initializer=init,name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu{}_1'.format(rr))(bn)
        lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=init,name='ConvStride_{}'.format(rr))(lay_act)
        bn = BatchNormalization()(lay_stride)
        lay_act = ELU(name='elu{}_2'.format(rr))(bn)
        act_list.append(lay_act)
        
    # expanding block n
    dd=numBlocks
    lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',kernel_initializer=init,name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv1 = BatchNormalization()(lay_deconv1)
    lay_deconv1 = ELU()(lay_deconv1)
    lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',kernel_initializer=init,name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv3 = BatchNormalization()(lay_deconv3)
    lay_deconv3 = ELU()(lay_deconv3)
    lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',kernel_initializer=init,name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv51 = ELU()(lay_deconv51)
    lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',kernel_initializer=init,name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_deconv52 = BatchNormalization()(lay_deconv52)
    lay_deconv52 = ELU()(lay_deconv52)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',kernel_initializer=init,name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    lay_stride = Conv2DTranspose(filt_num*dd,(4,4),strides=(2,2),kernel_initializer=init,name='DeConvStride_{}'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_stride)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(bn)
        
    # expanding blocks n-1
    expnums = list(range(1,numBlocks))
    expnums.reverse()
    for dd in expnums:
        lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',kernel_initializer=init,name='DeConv1_{}'.format(dd))(lay_act)
        lay_deconv1 = BatchNormalization()(lay_deconv1)
        lay_deconv1 = ELU()(lay_deconv1)
        lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',kernel_initializer=init,name='DeConv3_{}'.format(dd))(lay_act)
        lay_deconv3 = BatchNormalization()(lay_deconv3)
        lay_deconv3 = ELU()(lay_deconv3)
        lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',kernel_initializer=init,name='DeConv51_{}'.format(dd))(lay_act)
        lay_deconv51 = ELU()(lay_deconv51)
        lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',kernel_initializer=init,name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_deconv52 = BatchNormalization()(lay_deconv52)
        lay_deconv52 = ELU()(lay_deconv52)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',kernel_initializer=init,name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_stride = Conv2DTranspose(filt_num*dd,(4,4),strides=(2,2),kernel_initializer=init,name='DeConvStride_{}'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_stride)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(bn)
                
    lay_pad = ZeroPadding2D(padding=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_act)
    lay_cleanup = Conv2D(filt_num,(3,3),padding='same',kernel_initializer=init,name='CleanUp_1')(lay_pad)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU()(bn)
    lay_cleanup = Conv2D(filt_num,(3,3),padding='same',kernel_initializer=init,name='CleanUp_2')(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU()(bn)
    # output
    lay_out = Conv2D(1,(1,1), activation='sigmoid',kernel_initializer=init,name='output_layer')(lay_cleanup)
    
    return Model(lay_input,lay_out)

#%% Parameterized 2D Block Model
def BlockModel2D(input_shape,filt_num=16,numBlocks=3,output_chan=1):
    """Creates a Block CED model for segmentation or image regression problems
    Args:
        input shape: a list or tuple of [rows,cols,channels] of input images
        filt_num: the number of filters in the first and last layers
        This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
    Returns:
        An unintialized Keras model
        
    Example useage: SegModel = BlockModel2D([256,256,1],filt_num=8)
        
    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
    the rows/cols must be divisible by 2^numBlocks for skip connections
    to match up properly
    """
    
    # check for input shape compatibility
    rows,cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"
    
    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    assert minsize>4, "Too small of input for this many blocks. Use fewer blocks or larger input"
    
    # input layer
    lay_input = Input(shape=input_shape,name='input_layer')
    
    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1,numBlocks+1):
        x1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1,x3,x52],name='merge_{}'.format(rr))
        x = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1,1),name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='DownSample_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (3, 3),padding='same',name='ConvClean_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_clean_{}'.format(rr))(x)
        skip_list.append(x)
        
        
    # expanding blocks
    expnums = list(range(1,numBlocks+1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd-1],x],name='skip_connect_{}'.format(dd))
        x1 = Conv2D(filt_num*dd, (1, 1),padding='same',name='DeConv1_{}'.format(dd))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_Dx1_{}'.format(dd))(x1)
        x3 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv3_{}'.format(dd))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_Dx3_{}'.format(dd))(x3)
        x51 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_Dx51_{}'.format(dd))(x51)
        x52 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_Dx52_{}'.format(dd))(x52)
        x = concatenate([x1,x3,x52],name='Dmerge_{}'.format(dd))
        x = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_Dall_{}'.format(dd))(x)
        x = UpSampling2D(size=(2,2),name='UpSample_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConvClean1_{}'.format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean1_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConvClean2_{}'.format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean2_{}'.format(dd))(x)
        
    # classifier
    if output_chan==1:
        lay_out = Conv2D(output_chan,(1,1), activation='sigmoid',name='output_layer')(x)
    else:
        lay_out = Conv2D(output_chan,(1,1), activation='softmax',name='output_layer')(x)
    
    return Model(lay_input,lay_out)

import keras.backend as K
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# function for calculating over multiple subjects
def CalcVolumes(input_file,target_file,vox_vol,model):
    # load selected input
    input = np.load(input_file)[...,np.newaxis]
    # get mask prediction
    output = model.predict(input,batch_size=16)
    # threshold
    mask = (output>.5).astype(np.int)
    # count voxels
    tot_voxels = np.sum(mask)
    # get volume
    volume = tot_voxels * vox_vol
    # load selected target
    target = np.load(target_file)
    truth_mask = (target>.5).astype(np.int)
    # count voxels
    tot_truth_voxels = np.sum(truth_mask)
    # get volume
    truth_volume = tot_truth_voxels * vox_vol
    return volume,truth_volume