#!/usr/bin/env python

# some imports
import os
from functools import reduce
import operator
from skimage.draw import polygon
from scipy import interpolate
import numpy as np
np.random.seed(seed=1)
from glob import glob
from natsort import natsorted
import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

# import custom functions and viewing tools
from VisTools import multi_slice_viewer0, mask_viewer0
from KerasModel import BlockModel, dice_coef_loss

#~# some parameters to set for training #~#
# path to save best model weights
model_version = 5
model_weights_path = os.path.join(os.getcwd(),
                                  'BestModelWeights_dataset2_v{:02d}.h5'.format(model_version))
# set number of unique subjects to be used for testing
test_num = 15
# set number of unique subjects to to be used for validation
val_num = 10
# whether to use data augmentation or not
augment = True
# how many iterations of data to train on
numEp = 100
# augmentation factor
augFact = 2

# set data directories
dataset_dir = os.path.join('/home','bashirmllab','dataset2')
subdirs = ['opposed','SSFSE','t1nfs']
data_dirs = [os.path.join(dataset_dir,d+'_output') for d in subdirs]

# find unique subjects
all_groupings = []
all_grouped_inputs = []
all_grouped_targets = []
all_numScans = 0
all_numSubjs = 0
for cur_data_dir,cur_subdir in zip(data_dirs,subdirs):
    all_inputs = natsorted(glob(os.path.join(cur_data_dir,"input*.npy")))
    all_targets = natsorted(glob(os.path.join(cur_data_dir,"target*.npy")))
    stem_length = len(cur_subdir)+7
    stems = [f[:-stem_length] for f in all_inputs]
    unq_stems = np.unique(stems)
    # get number of unique subjects
    numSubjs = len(unq_stems)
    print('For {} directory:'.format(cur_subdir))
    print('{} total scans found'.format(len(all_inputs)))
    print('{} unique subjects'.format(numSubjs))
    # group repeated scans together
    groupings = [[i for i, e in enumerate(stems) if e == u] for u in unq_stems]
    grouped_inputs = [[all_inputs[g] for g in group] for group in groupings]
    grouped_targets = [[all_targets[g] for g in group] for group in groupings]
    # add to cumulative lists
    all_groupings += groupings
    all_grouped_inputs += grouped_inputs
    all_grouped_targets += grouped_targets
    all_numScans += len(all_inputs)
    all_numSubjs += numSubjs
print('----------------------------------')
print('{} total scans found'.format(all_numScans))
print('{} total unique subjects found'.format(all_numSubjs))
print('----------------------------------')

# Randaomly select test subject indices
# numpy is seeded so this is repeatable
tv_inds = np.random.choice(all_numSubjs,test_num+val_num,replace=False)
test_inds = tv_inds[:test_num]
val_inds = tv_inds[test_num:]

# split into test,train,validation
# grab testing files and reduce to a single list
input_files_test = reduce(operator.add,[all_grouped_inputs[i] for i in test_inds])
target_files_test = reduce(operator.add,[all_grouped_targets[i] for i in test_inds])
# grab validation files and reduce to a single list
input_files_val = reduce(operator.add,[all_grouped_inputs[i] for i in val_inds])
target_files_val = reduce(operator.add,[all_grouped_targets[i] for i in val_inds])
# take the rest of the groups that aren't validation or test and bring together into
# single lists of inputs and targets
train_input_groups = [f for i,f in enumerate(all_grouped_inputs) if i not in tv_inds]
train_target_groups = [f for i,f in enumerate(all_grouped_targets) if i not in tv_inds]
input_files_train = reduce(operator.add,train_input_groups)
target_files_train = reduce(operator.add,train_target_groups)

# load input data
print('Loading input data...')
inputs_test = np.concatenate([np.load(f) for f in input_files_test])
inputs_val = np.concatenate([np.load(f) for f in input_files_val])
inputs_train = np.concatenate([np.load(f) for f in input_files_train])
# add singleton dimension for grayscale channel
testX = inputs_test[...,np.newaxis]
valX = inputs_val[...,np.newaxis]
trainX = inputs_train[...,np.newaxis]
print('Input data loaded')
print('{} training slices'.format(trainX.shape[0]))
print('{} validation slices'.format(valX.shape[0]))
print('{} testing slices'.format(testX.shape[0]))
# load target data
print('Loading target data...')
targets_test = np.concatenate([np.load(f) for f in target_files_test])
targets_val = np.concatenate([np.load(f) for f in target_files_val])
targets_train = np.concatenate([np.load(f) for f in target_files_train])
# add singleton dimension for grayscale channel
testY = targets_test[...,np.newaxis]
valY = targets_val[...,np.newaxis]
trainY = targets_train[...,np.newaxis]
print('Target data loaded')

# make model
model = BlockModel(trainX.shape,filt_num=32,numBlocks=4)
model.compile(optimizer=RMSprop(lr=2e-3), loss=dice_coef_loss)

# setup image data generator
if augment:
    datagen1 = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    datagen2 = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
else:
    datagen1 = ImageDataGenerator()
    datagen2 = ImageDataGenerator()
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
datagen1.fit(trainX, seed=seed)
datagen2.fit(trainY, seed=seed)
batchsize = 16
datagen = zip( datagen1.flow( trainX, None, batchsize, seed=seed), datagen2.flow( trainY, None, batchsize, seed=seed) )

# calculate number of batches
if augment:
    steps = np.int(trainX.shape[0]/batchsize*augFact)
else:
    steps = np.int(trainX.shape[0]/batchsize)
    
# make callback for checkpointing
cb_check = ModelCheckpoint(model_weights_path,monitor='val_loss',
                                   verbose=0,save_best_only=True,
                                   save_weights_only=True,mode='auto',period=1)

# make callback for learning rate schedule
def Scheduler(epoch,lr):
    jump = int(epoch/10)
    return 2e-3 * (1/2)**jump
cb_schedule = LearningRateScheduler(Scheduler,verbose=1)

# train model
history = model.fit_generator(datagen,
                    steps_per_epoch=steps,
                    epochs=numEp,
                    callbacks=[cb_check,cb_schedule],
                    verbose=1,
                    validation_data=(valX,valY))
# load best weights
model.load_weights(model_weights_path)
# evaluate on test data
score = model.evaluate(testX,testY,verbose=0)
print("Test Dice score is {:.03f}".format(1-score))
