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
from segmentation_tools import GetROIcoords, GetImageMaskData


# set desired in-plane dimensions
new_dims = (512,512)

# turn plotting on or off
display_plots = True

# set z-axis orientation to normal or reversed
# z_reversed = True

# set current working directory
root_dir = '/Volumes/Seagate/radiology/research/liversegmentation/MRI/fifty_portal_venous_phases'

os.chdir(os.path.join(root_dir))
output_dir = os.path.join(root_dir, 'output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sub_dirs = sorted(glob(os.path.join(os.getcwd(),'*')))

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
    # temp_dir = glob(os.path.join(data_dir,'*',''))

    # #### Comment out the below for some of the files.
    # temp_dir = glob(os.path.join(temp_dir[0],'*',''))
    # temp_dir = glob(os.path.join(temp_dir[0],'*',''))
    # #### End

    # import pdb; pdb.set_trace()
    # dicom_dirs = natsorted(glob(os.path.join(temp_dir[0],'*','')))
    # # get current subject set of dicom files
    # cur_dcm_files = glob(os.path.join(dicom_dirs[cur_subj], "*.dcm"))

    cur_dcm_files =  sorted(glob('%s/**/*.dcm' % data_dir, recursive=True))

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



    
