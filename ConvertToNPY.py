# some imports
import os
from skimage.draw import polygon
from skimage.transform import resize
from scipy import interpolate
import nibabel as nib
import pydicom as dcm
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


# set desired in-plane dimensions
new_dims = (512,512)

# turn plotting on or off
display_plots = False

# set z-axis orientation to normal or reversed
z_reversed = True

# function for loading coordinate data from ROI
def GetROIcoords(roi):
    # get index of ROI
    ind = int(roi['integer'][1])
    # get coordinate data of ROI
    x = roi['array']['dict']['array'][1]['string']
    # convert string coordinates to tuples
    coords = [literal_eval(coord) for coord in x]
    # parse out x and y and make closed loop
    x = [i[0] for i in coords] + [coords[0][0]]
    y = [i[1] for i in coords] + [coords[0][1]]
    # apply parametric spline interpolation
    tck, _ = interpolate.splprep([x,y], s=0, per=True)
    x, y = interpolate.splev(np.linspace(0,1,500), tck)
    return ind,x,y

# function for loading in images and mask given file paths
def GetImageMaskData(file_xml,dcm_files,new_dims):
    # open up xml file and grab the list of ROIs
    with open(file_xml) as f:
        doc = xmltodict.parse(f.read())
    roi_list = doc['plist']['dict']['array']['dict']
    # parse out the image shape
    # imshape = (int(roi_list[0]['integer'][0]),int(roi_list[0]['integer'][3]))
    # get slice locations of all dicoms
    locs = [(d,float(dcm.dcmread(d).SliceLocation)) for d in dcm_files]
    # sort
    locs.sort(key=itemgetter(1))
    dcm_files = [l[0] for l in locs]
    # load dicoms into volume
    image_volume = np.stack([dcm.dcmread(d).pixel_array for d in dcm_files]).astype(np.float)
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
        ind,x,y = GetROIcoords(cur_roi)
        xs = [d*x_scale for d in x]
        ys = [d*y_scale for d in y]
        rr, cc = polygon(ys, xs)
        # one of these two lines determines
        #  z-axis orientation of the mask
        if z_reversed:
            zind = ind
        else:
            zind = mask.shape[0]-ind-1
        
        mask[zind,rr, cc] = 1
        # import pdb; pdb.set_trace()
    return im_vol,mask


# set current working directory
#os.chdir(os.path.join('C:\\Users','Johns','Box','UpworkClient'))
# os.chdir(os.path.join('/Users/ln30/Dropbox/Radiology/Research/UpworkClient'))
# root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/SSFSE_JKB01_165'

# root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/temp_run'
root_dir = os.path.join('C:\\Users','Johns','Box','UpworkClient','SampleData2')

os.chdir(os.path.join(root_dir))
output_dir = os.path.join(root_dir, 'output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sub_dirs = glob(os.path.join(os.getcwd(),'*'))

# take off the first elements which is a spreadsheet
# sub_dirs.pop(0)


for data_dir in sub_dirs:
    path_to_zip_file = glob(os.path.join(data_dir, '*.zip'))
    path_to_zip_file = path_to_zip_file[0]
    directory_to_extract_to = path_to_zip_file.replace('.zip', '')

    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # delete resource fork: weird os x thing
    resource_fork_path = os.path.join(directory_to_extract_to, '__MACOSX')

    if os.path.exists(resource_fork_path):
        shutil.rmtree(resource_fork_path)

    # set data directory
    # data_dir = os.path.join(os.getcwd(),'SampleData')
    # set output directory
    # output_dir = os.path.join(os.getcwd(),'NPYdata')
    # os.makedirs(output_dir, exist_ok=True)


    # subject index
    cur_subj = 0

    # find xml files
    xml_files = natsorted(glob(os.path.join(data_dir, "*.xml")))
    # grab current subject file
    cur_xml_file = xml_files[cur_subj]

    # find dicom files
    temp_dir = glob(os.path.join(data_dir,'*',''))
    temp_dir = glob(os.path.join(temp_dir[0],'*',''))
    temp_dir = glob(os.path.join(temp_dir[0],'*',''))
    
    dicom_dirs = natsorted(glob(os.path.join(temp_dir[0],'*','')))
    # get current subject set of dicom files
    cur_dcm_files = glob(os.path.join(dicom_dirs[cur_subj], "*.dcm"))


    # # load in images and mask
    imvol, maskvol = GetImageMaskData(cur_xml_file,cur_dcm_files,new_dims)

    # # display
    if display_plots:
        mask_viewer0(imvol,.5*maskvol)
        plt.show()

    # save arrays as .npy
    # image_name = os.path.join(output_dir,'input{:04d}.npy'.format(cur_subj))
    # target_name = os.path.join(output_dir,'target{:04d}.npy'.format(cur_subj))
    subject_basename = os.path.basename(os.path.normpath(data_dir))
    image_name = os.path.join(output_dir,'input_' + subject_basename)
    target_name = os.path.join(output_dir,'target_' + subject_basename)
    np.save(image_name,imvol)
    np.save(target_name,maskvol)

    print(subject_basename)


    
