import os
from skimage.draw import polygon
from skimage.transform import resize
from scipy import interpolate
import nibabel as nib
import pydicom
import numpy as np
from glob import glob
import xmltodict
from ast import literal_eval
from matplotlib import pyplot as plt
from operator import itemgetter
from natsort import natsorted
from VisTools import multi_slice_viewer0, mask_viewer0
import zipfile
import shutil
from pillow_handler import get_pixeldata
import sys
sys.path.append('/usr/local/Cellar/gdcm/2.8.9/lib/python3.7/site-packages')
import gdcm
import decimal


# def determine_patient_orientation(dcm_file):
#     '''
#     Determines the orientation of a patient for processing.
#     '''


#     # import pdb; pdb.set_trace()
#     read_dcm = pydicom.dcmread(dcm_file)

#     patient_position = read_dcm.PatientPosition

#     if str(read_dcm).find('SIEMENS') != -1:
#         manufacturer = 'SIEMENS'
#     else:
#         manufacturer = 'GE'

#     if manufacturer == 'GE' and patient_position == 'FFS':
#         z_reversed = False
#     elif manufacturer == 'SIEMENS' and patient_position == 'FFS':
#         z_reversed = True
#     elif manufacturer == 'SIEMENS' and patient_position == 'HFS':
#         z_reversed = False
#     else:
#         print('Another manufacturer and position found!')
#         import pdb; pdb.set_trace()

#     print('Manufacturer header is %s' %(manufacturer))
#     return z_reversed

# function for loading coordinate data from ROI
def GetROIcoords(roi):
    # get index of ROI
    # ind = int(roi['integer'][1])
    # get coordinate data of ROI
    # print(roi['array']['dict']['string'][0])
    zcoord = float(roi['array']['dict']['string'][0].split(' ')[2].replace(')', ''))

    x = roi['array']['dict']['array'][1]['string']
    # convert string coordinates to tuples
    coords = [literal_eval(coord) for coord in x]
    # parse out x and y and make closed loop
    x = [i[0] for i in coords] + [coords[0][0]]
    y = [i[1] for i in coords] + [coords[0][1]]
    # apply parametric spline interpolation
    tck, _ = interpolate.splprep([x,y], s=0, per=True)
    x, y = interpolate.splev(np.linspace(0,1,500), tck)
    return zcoord,x,y

# function for loading in images and mask given file paths
def GetImageMaskData(file_xml,dcm_files,new_dims):

    # z_reversed = determine_patient_orientation(dcm_files[0])

    # open up xml file and grab the list of ROIs
    with open(file_xml) as f:
        doc = xmltodict.parse(f.read())
    roi_list = doc['plist']['dict']['array']['dict']

    # Sort ROIs by slice location.
    roi_locs = [(r, float(((r['array']['dict']['string'][0].split(' ')[2].replace(')', ''))))) for r in roi_list]

    roi_locs.sort(key=itemgetter(1))

    # roi_list = [r[0] for r in roi_locs]

    # parse out the image shape
    # imshape = (int(roi_list[0]['integer'][0]),int(roi_list[0]['integer'][3]))
    # get slice locations of all dicoms
    locs = [(d,float(pydicom.dcmread(d).SliceLocation)) for d in dcm_files]
    # sort
    locs.sort(key=itemgetter(1))
    dcm_files = [l[0] for l in locs]

    image_zcoords = [round(l[1], 6) for l in locs]



    # print([l[1] for l in locs])
    # print([r[1] for r in roi_locs])

    # load dicoms into volume
    image_volume = np.stack([pydicom.dcmread(d).pixel_array for d in dcm_files]).astype(np.float)
    # normalize by image
    for i in range(image_volume.shape[0]):
        im = image_volume[i]
        im /= np.max(im)
        image_volume[i] = im
    # resample image volume to desired dimensions
    output_shape = (image_volume.shape[0],new_dims[0],new_dims[1])
    im_vol = resize(image_volume,output_shape)
    # convert contours into masks
    # make empty mask
    mask = np.zeros(output_shape)
    # import pdb; pdb.set_trace()
    # calculate rescaling factor in each dimension
    x_scale = float(new_dims[1])/float(image_volume.shape[2])
    y_scale = float(new_dims[0])/float(image_volume.shape[1])
    # loop over ROIs
    for cur_roi in roi_list:
        zcoord,x,y = GetROIcoords(cur_roi)
        # Get index of the closest number to zcoord. 
        takeClosest = lambda zcoord,image_zcoords:min(image_zcoords,key=lambda x:abs(x-zcoord))
        ROI_ind = image_zcoords.index(takeClosest(zcoord, image_zcoords))
        xs = [d*x_scale for d in x]
        ys = [d*y_scale for d in y]
        rr, cc = polygon(ys, xs)
        # one of these two lines determines
        #  z-axis orientation of the mask
        # z_reversed = True
        # if z_reversed:
        #     zind = ind
        # else:
        #    zind = mask.shape[0]-ind-1
        mask[ROI_ind,rr, cc] = 1
    return im_vol,mask