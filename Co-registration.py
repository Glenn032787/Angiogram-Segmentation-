import os
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import sys
import scipy.ndimage

print('Getting image')
#Get image path and images
cta_image_path = os.path.join("original_images", 'CTA')
cta_slice_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(cta_image_path)
cta_image = sitk.ReadImage(cta_slice_filenames)
cta_array = sitk.GetArrayFromImage(cta_image)

ncct_image_path = os.path.join("original_images", 'NCCT')
ncct_slice_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(ncct_image_path)
ncct_image = sitk.ReadImage(ncct_slice_filenames)
ncct_array = sitk.GetArrayFromImage(ncct_image)

print('Registering Image... ')
resultImage = sitk.Elastix(cta_image, ncct_image, "bspline")
print('Finish registration')

print('Interpolate')
origin = resultImage.GetOrigin()
direction = resultImage.GetDirection()

original_size = np.array(resultImage.GetSize())
original_spacing = np.array(resultImage.GetSpacing())

# since we want to preserve the whole image region, we can calculate the new size as
# the original size times the spacing ratio rounded to the nearest integer
new_size = np.array(cta_image.GetSize()).astype(np.int).tolist()
new_spacing = original_size * original_spacing / new_size

# set up the resampling filter...
resampler = sitk.ResampleImageFilter()
resampler.SetOutputOrigin(origin)
resampler.SetOutputDirection(direction)
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(new_size)
resampler.SetInterpolator(sitk.sitkBSpline) # use BSpline interpolation

# ...and apply it to the image
resampled_image = resampler.Execute(resultImage)

ncct_BSpline = sitk.GetArrayFromImage(resampled_image)
print('Finish interpolation')


print('Saving Registered Image (CTA image)')
print(f'Original CTA size: {np.shape(cta_array)}')
print(f'New CTA size: {np.shape(cta_array)}')
sitk.WriteImage(cta_image, os.path.join('registered_image', 'cta.nii'))

print('Saving NCCT Image')
print(f'Original NCCT size: {np.shape(ncct_array)}')
print(f'New NCCT size: {np.shape(ncct_BSpline)}')
sitk.WriteImage(resampled_image, os.path.join('registered_image', 'ncct.nii'))
print('Finish Saving Image')
