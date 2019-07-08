## Usage: python3 /Users/ln30/Git/liversegmentation/ConvertToNPY_CT.py


# root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/CT/troubleshoot'
unzipfile = True
root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/CT/temp_exclude'
# root_dir = 
# root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/CT/done_by_steve_split'
# root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/SSFSE_JKB01_165'
# root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/temp_run'

# some imports
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

new_dims = (512,512) # set desired in-plane dimensions
display_plots = True # turn plotting on or off
save_files = True # Save actual files
# z_reversed = True # set z-axis orientation to normal or reversed


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

    # GE goes forwards and SIEMENS goes backwards.
    # manufacturer = pydicom.dcmread(dcm_files[0]).Manufacturer

    # if manufacturer == 'GE MEDICAL SYSTEMS' or 'MPTronic software':
    #     z_reversed = False
    # elif manufacturer == 'SIEMENS':
    #     z_reversed = True
    # else:
    #     print('Another manufacturer found!')

    patient_position = pydicom.dcmread(dcm_files[0]).PatientPosition

    if str(pydicom.dcmread(dcm_files[0])).find('SIEMENS') != -1:
        manufacturer = 'SIEMENS'
    else:
        manufacturer = 'GE'

    if manufacturer == 'GE' and patient_position == 'FFS':
        z_reversed = False
    elif manufacturer == 'SIEMENS' and patient_position == 'FFS':
        z_reversed = True
    elif manufacturer == 'SIEMENS' and patient_position == 'HFS':
        z_reversed = False
    else:
        print('Another manufacturer and position found!')
        import pdb; pdb.set_trace()

    print(manufacturer)

    # open up xml file and grab the list of ROIs
    with open(file_xml) as f:
        doc = xmltodict.parse(f.read())
    roi_list = doc['plist']['dict']['array']['dict']
    # parse out the image shape
    # imshape = (int(roi_list[0]['integer'][0]),int(roi_list[0]['integer'][3]))
    # get slice locations of all dicoms
    locs = [(d,float(pydicom.dcmread(d).SliceLocation)) for d in dcm_files]
    # sort
    locs.sort(key=itemgetter(1))
    dcm_files = [l[0] for l in locs]


    image_volume = np.stack([pydicom.dcmread(d).pixel_array for d in dcm_files])

    # window level
    dicom_img = pydicom.dcmread(dcm_files[0])
    window_center=50.0
    window_width=400.0
    jpg_scale = 255.0
    lower_limit = window_center-(window_width/2)
    upper_limit = window_center+(window_width/2)

    # dicom_img = dicom.read_file(dcm_file_path, force=True)
    img = image_volume

    rescale_slope = float(dicom_img.RescaleSlope)
    rescale_intercept = float(dicom_img.RescaleIntercept)
    hounsfield_img = (img * rescale_slope) + rescale_intercept

    clipped_img = np.clip(hounsfield_img, lower_limit, upper_limit)

    windowed_img = (clipped_img / window_width) - (lower_limit / window_width)

    # # normalize by image
    for i in range(windowed_img.shape[0]):
        im = windowed_img[i]
        im /= np.max(im)
        windowed_img[i] = im
    # resample image volume to desired dimensions
    output_shape = (windowed_img.shape[0],new_dims[0],new_dims[1])
    im_vol = resize(windowed_img,output_shape)
    # convert contours into masks
    # make empty mask
    mask = np.zeros(output_shape)
    # calculate rescaling factor in each dimension
    x_scale = float(new_dims[1])/float(windowed_img.shape[2])
    y_scale = float(new_dims[0])/float(windowed_img.shape[1])
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
    return im_vol,mask


# set current working directory


os.chdir(os.path.join(root_dir))
output_dir = os.path.join(root_dir, 'output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

patient_dirs = glob(os.path.join(root_dir,'*'))

# take off the first elements which is a spreadsheet
# patient_dirs.pop(0)


for patient_dir in patient_dirs:
    phase_dirs = glob(os.path.join(patient_dir,'*'))

    for phase_dir in phase_dirs:
        if unzipfile:
            path_to_zip_file = glob(os.path.join(phase_dir, '*.zip'))
            path_to_zip_file = path_to_zip_file[0]
            directory_to_extract_to = path_to_zip_file.replace('.zip', '')

            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(directory_to_extract_to)
            zip_ref.close()

            # delete resource fork: weird os x thing
            resource_fork_path = os.path.join(directory_to_extract_to, '__MACOSX')

            if os.path.exists(resource_fork_path):
                shutil.rmtree(resource_fork_path)


        # subject index
        cur_subj = 0

        # find xml files
        xml_files = natsorted(glob(os.path.join(phase_dir, "*.xml")))
        # grab current subject file
        cur_xml_file = xml_files[cur_subj]

        # find dicom files
        temp_dir = glob(os.path.join(phase_dir,'*',''))

        #### Comment out the below for some of the files.
        temp_dir = glob(os.path.join(temp_dir[0],'*',''))
        temp_dir = glob(os.path.join(temp_dir[0],'*',''))
        #### End


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
        image_name = os.path.join(output_dir,'input{:04d}.npy'.format(cur_subj))
        target_name = os.path.join(output_dir,'target{:04d}.npy'.format(cur_subj))
        subject_basename = os.path.basename(os.path.normpath(phase_dir))
        image_name = os.path.join(output_dir,'input_' + subject_basename)
        target_name = os.path.join(output_dir,'target_' + subject_basename)

        if save_files:
            np.save(image_name,imvol)
            np.save(target_name,maskvol)

        print(subject_basename)



    
