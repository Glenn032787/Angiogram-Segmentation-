#!/usr/bin/env python
# coding: utf-8

# # CTA NCCT subtration - Angiogram segmentation
# Add images to 'original_image/CTA' and 'original_image/NCCT' \
# Run 'Bspline.py' for coregistration before running this code




get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import sys
from skimage import morphology
from skimage.measure import label
from skimage.segmentation import active_contour
from skimage.filters import threshold_otsu,threshold_multiotsu
from ipywidgets import interact, fixed
import scipy
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly import figure_factory as FF
from plotly.offline import plot, iplot
from collections import Counter
from scipy.ndimage import gaussian_filter
from stl import mesh
from trimesh import Trimesh, remesh, smoothing


# # Get Image and image info

#Get image path and images
#cta_image_path = os.path.join("images_arterial_cta")
#cta_slice_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(cta_image_path)

cta_image_path = os.path.join("registered_image", 'cta.nii')
cta_image = sitk.ReadImage(cta_image_path, sitk.sitkFloat32)
cta_array = sitk.GetArrayFromImage(cta_image)

ncct_image_path = os.path.join("registered_image", 'ncct.nii')
ncct_image = sitk.ReadImage(ncct_image_path, sitk.sitkFloat32)
ncct_array = sitk.GetArrayFromImage(ncct_image)


print(f'CTA Dimention: {cta_array.shape}\nCTA dtype: {cta_array.dtype}\nCTA spacing: {cta_image.GetSpacing()}\n')
print(f'NCCT Dimention: {ncct_array.shape}\nNCCT dtype: {ncct_array.dtype}\nNCCT spacing: {ncct_image.GetSpacing()}\n')




#Show CTA and NCCT images
slide = 10

plt.subplot(1,2,1)
plt.imshow(cta_array[slide], cmap='bone')
plt.title(f'CTA Image - Slide {slide}')
plt.subplot(1,2,2)
plt.imshow(ncct_array[slide], cmap='bone')
plt.title(f'NCCT Image - Slide {slide}')
plt.show()





#Histogram
slide = 15

plt.hist(cta_array[slide].ravel(), bins=100)
plt.title(f'CTA Slide {slide} Histogram')
plt.show()

plt.hist(ncct_array[slide].ravel(), bins=100)
plt.title(f'NCCT Slide {slide} Histogram')
plt.show()


# # Defining Functions




def make_mesh(image, step_size=1):

    print ("Transposing surface")
    p = image.transpose(2,1,0)
    
    print ("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print ("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
    # colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)

def plt_3d(verts, faces):
    print ("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()
    
    
def hist_noise_remove(histogram, threshold):
    hist_edit = histogram
    item = Counter(histogram).items()
    for pixel in item:
        if pixel[1] < threshold:
            hist_edit = hist_edit[hist_edit != pixel[0]]
    return hist_edit


def crop_image(image, num_pixel = 20):
    crop_image = np.ones(image.shape)*np.amin(image)
    crop_image[num_pixel:-num_pixel, num_pixel:-num_pixel] = image[num_pixel:-num_pixel, num_pixel:-num_pixel]
    return crop_image


# # CTA Segmentation



background_images = np.zeros(cta_array.shape)
threshold_list = []
threshold_images = np.zeros(cta_array.shape)

for slide in range(cta_array.shape[0]):

    # Remove background circle
    background_remove = np.copy(cta_array[slide])
    background_remove_mask = (background_remove>-2000).astype(int)
    background_remove[background_remove_mask == 0] = np.amin(background_remove_mask*background_remove)
    threshold = threshold_otsu(background_remove)
    mask = background_remove>threshold
    background_remove[mask==0] = np.amin(background_remove)
    background_images[slide] = background_remove
    
    # Get initial thrshold (Use 2 standard deviation from median)
    hist_background_remove = background_remove.ravel()[background_remove.ravel() != np.amin(background_remove.ravel())]
    hist_background_remove = np.around(hist_background_remove, 1)
    hist_thresholding = hist_noise_remove(hist_background_remove.ravel(), 50) 
    initial_threshold = hist_thresholding.mean() + 2*abs(hist_thresholding.std())
    
    # Threshold image 
    im = background_remove > initial_threshold
    
    threshold_list.append(initial_threshold)
    threshold_images[slide] = im
    
    plt.imshow(im)
    plt.title(f'{slide} --- {initial_threshold}')
    plt.show()
    
print('Finish')




print_image = False
tissue_seg = np.zeros(cta_array.shape)
bone_seg = np.zeros(cta_array.shape)
cta_bone_mask = np.zeros(cta_array.shape)

for slide in range(cta_array.shape[0]):
    k = 0
    ThPrev=np.inf #previous
    ThRev=threshold_list[slide]

    image = background_images[slide]

    #apply threshold
    while ThRev!=ThPrev: #Loop until new threshold is equal to prev threshold

        ThPrev=ThRev #update the previous threshold for comparison in subsequent iterations

        prebonemask = (image>ThRev)
        
        # 3D Connectivity
        # Obtain previous and next slice
        if slide == 0:
            next_mask = (prebonemask + threshold_images[slide+1])
            z_connection = (next_mask == 2).astype(int) 
        elif slide == cta_array.shape[0]-1:
            previous_mask = (threshold_images[slide-1] + prebonemask)
            z_connection = (previous_mask == 2).astype(int) 
        else:
            next_mask = ((prebonemask + threshold_images[slide+1])==2).astype(int)
            previous_mask = ((threshold_images[slide-1] + prebonemask)==2).astype(int)

            z_connection = ((next_mask + previous_mask)==2).astype(int)

        # Find connected pixels between previous current and next slide
        coordinates= np.argwhere(z_connection ==1)      
        im_ff=np.float32(image)
        mask = np.zeros((im_ff.shape[0]+2, im_ff.shape[1]+2), np.uint8)
        for item in range(len(coordinates)):
            cv2.floodFill(im_ff, mask, (coordinates[item][1],coordinates[item][0]), 2)

        # Remove small island based on connectivity 
        nonZ =(im_ff==1)
        Z = (im_ff==2)

        Z_keep = (morphology.remove_small_objects(nonZ,min_size=2, connectivity=1))
        nonZ_keep = (morphology.remove_small_objects(nonZ,min_size=8, connectivity=1))
        bonemask = Z + nonZ_keep 

        # Get Bone and Tissue segment
        boneseg = np.copy(image)
        boneseg[bonemask==0] = np.amin(image)

        tissueseg = np.copy(image)
        tissueseg[bonemask==1] = np.amin(image)

        # Calculate mean signal intensities
        tissue_seg_mask=np.ma.masked_where(tissueseg == np.amin(tissueseg), tissueseg)
        bone_seg_mask=np.ma.masked_where(boneseg==np.amin(boneseg),boneseg) 

        tissue_mean=np.mean(tissue_seg_mask)
        bone_mean=np.mean(bone_seg_mask)

        #Threshold optimization equation
        ThRev=(1+((bone_mean-tissue_mean)/bone_mean))*tissue_mean 


        if print_image:
            fig, axs = plt.subplots (1,4, figsize=(12,4))  
            axs[0].imshow(tissueseg, cmap='bone') 
            axs[1].imshow(tissue_seg_mask,cmap="bone")
            axs[2].imshow(boneseg,cmap="bone")
            axs[3].imshow(bone_seg_mask,cmap="bone") 
            plt.show()

            print(f"Revised Threshold={ThRev}\n\tMean SOFT TISSUE Intensity={MuscSegI} pixels\n\tMean BONE Intensity={FatSegI} pixels\n\n")
        k+=1
        if k==50:
            break
    bone_seg[slide] = boneseg
    tissue_seg[slide] = tissueseg
    cta_bone_mask[slide] = bonemask
    
    plt.imshow(cta_bone_mask[slide])
    plt.show()
    
    if slide == cta_array.shape[0]//2:
        print('50% complete')

print('Finish')


# # NCCT Segmentation



background_images = np.zeros(ncct_array.shape)
threshold_list = []
threshold_images = np.zeros(ncct_array.shape)

for slide in range(ncct_array.shape[0]):
    if (ncct_array[slide] == np.zeros(ncct_array[slide].shape)).all():
        threshold_list.append(0)
        continue 
        
    #Remove background
    background_remove = np.copy(ncct_array[slide])  
    background_remove_mask = (background_remove>-2000).astype(int)
    background_remove[background_remove_mask == 0] = np.amin(background_remove_mask*background_remove)
    background_remove = crop_image(background_remove)
    
    threshold = threshold_otsu(background_remove)
    mask = background_remove>threshold
    background_remove[mask==0] = np.amin(background_remove)
   
    background_images[slide] = background_remove
   
    #Get initial thrshold 
    hist_background_remove = background_remove.ravel()[background_remove.ravel() != np.amin(background_remove.ravel())]
    hist_background_remove = np.around(hist_background_remove, 1)
    
    hist_thresholding = hist_noise_remove(hist_background_remove, 50) #remove pixel intensity with less than 20 pixels
    initial_threshold = hist_thresholding.mean() + 2*abs(hist_thresholding.std())
    
    im = background_remove > initial_threshold
    
    threshold_list.append(initial_threshold)
    threshold_images[slide] = im
    
    plt.imshow(im)
    plt.title(f'{slide} --- {initial_threshold}')
    plt.show()
    
    
print('Finish')





print_image = False
tissue_seg = ncct_bone_mask = bone_seg = np.zeros(ncct_array.shape)

for slide in range(ncct_array.shape[0]):
    k = 0
    ThPrev=np.inf #previous
    ThRev=threshold_list[slide]

    image = background_images[slide]
    
    if (image == np.zeros(image.shape)).all():
        continue
        
    #apply threshold
    while ThRev!=ThPrev: #while new threshold is NOT equal to prev threshold

        ThPrev=ThRev #update the previous threshold for comparison in subsequent iterations

        prebonemask = (image>ThRev)

        #Add slide before and after to find z-connection
        if slide == 0:
            next_mask = (prebonemask + threshold_images[slide+1])
            z_connection = (next_mask == 2).astype(int) 
        elif slide == ncct_array.shape[0]-1:
            previous_mask = (threshold_images[slide-1] + prebonemask)
            z_connection = (previous_mask == 2).astype(int) 
        else:
            next_mask = ((prebonemask + threshold_images[slide+1])==2).astype(int)
            previous_mask = ((threshold_images[slide-1] + prebonemask)==2).astype(int)

            z_connection = ((next_mask + previous_mask)==2).astype(int)

        #NEW: find XY connections to Z connected parts
        coordinates= np.argwhere(z_connection ==1) #put coordinates of Z-connections into a list      
        im_ff=np.float32(image)
        mask = np.zeros((im_ff.shape[0]+2, im_ff.shape[1]+2), np.uint8)
        for item in range(len(coordinates)):
            cv2.floodFill(im_ff, mask, (coordinates[item][1],coordinates[item][0]), 2)

        #Remove small islands for Non-Z parts
        nonZ =(im_ff==1)
        Z = (im_ff==2)

        Z_keep = (morphology.remove_small_objects(nonZ,min_size=2, connectivity=1))
        nonZ_keep = (morphology.remove_small_objects(nonZ,min_size=8, connectivity=1))
        bonemask = Z + nonZ_keep 

        #Get Bone and Tissue segment
        boneseg = np.copy(image)
        boneseg[bonemask==0] = np.amin(image)

        tissueseg = np.copy(image)
        tissueseg[bonemask==1] = np.amin(image)

        #masking the 0's in the image to exclude in the mean calculations
        tissue_seg_mask=np.ma.masked_where(tissueseg == np.amin(tissueseg), tissueseg)
        bone_seg_mask=np.ma.masked_where(boneseg==np.amin(boneseg),boneseg) 

        #Calculate mean signal intensities
        tissue_mean=np.mean(tissue_seg_mask)
        bone_mean=np.mean(bone_seg_mask)

        #Threshold optimization equation
        ThRev=(1+((bone_mean-tissue_mean)/bone_mean))*tissue_mean 


        if print_image:
            fig, axs = plt.subplots (1,4, figsize=(12,4))  
            axs[0].imshow(tissueseg, cmap='bone') 
            axs[1].imshow(tissue_seg_mask,cmap="bone")
            axs[2].imshow(boneseg,cmap="bone")
            axs[3].imshow(bone_seg_mask,cmap="bone") 
            plt.show()

            print(f"Revised Threshold={ThRev}\n\tMean SOFT TISSUE Intensity={MuscSegI} pixels\n\tMean BONE Intensity={FatSegI} pixels\n\n")
        k+=1
        if k==50:
            break
    bone_seg[slide] = boneseg
    tissue_seg[slide] = tissueseg
    ncct_bone_mask[slide] = bonemask
    
    plt.imshow(bonemask)
    plt.title(slide)
    plt.show()

print('Finish')


# # Subtraction - Angiogram segmentation




angiogram = np.zeros(cta_bone_mask.shape)
angiogram_HU = np.zeros(cta_bone_mask.shape)

for i in range(cta_bone_mask.shape[0]):
    # Bone subtraction from NCCT to CTA
    ang = (cta_bone_mask[i] - ncct_bone_mask[i]) == 1
    
    # Remove small objects and morphology close to remove noise
    ang = (morphology.remove_small_objects(ang.astype(bool), min_size=10, connectivity=1)).astype(int)
    ang = np.uint8(ang)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(ang, cv2.MORPH_OPEN, kernel)
    angiogram[i] = ang
    
    angiogram_HU[i] = ang * cta_array[i]
    
    plt.imshow(angiogram_HU[i])
    plt.title(f'Slide {i}')
    plt.show()


# # Generate Angiogram Model




angiogram = (morphology.remove_small_objects(angiogram_HU.astype(bool), min_size= 10, connectivity=1)).astype(int)

angiogram_cut = angiogram[15:-60, :, 150:angiogram.shape[2]-150]

label = measure.label(angiogram_cut)
label_dict = (Counter(np.ravel(label)))

threshold = 300

angio_top = np.zeros(angiogram_cut.shape)

for item in label_dict.items():
    if item[1] >= threshold and item[0] != 0:
        angio_top = angio_top + (label == item[0]).astype(int)

angiogram_crop = np.zeros(angiogram.shape)
angiogram_crop[15:-60, :, 150:angiogram.shape[2]-150] = angio_top
# angiogram_crop = angiogram_crop*angiogram_HU


# # TEST (Trimesh)



for im in angiogram_crop:
    im[0, :] = 1
    im[:, 0] = 1
    
v, f = make_mesh(angiogram_crop[2])
v, f = remesh.subdivide(v, f)

CTmesh = Trimesh()
CTmesh.vertices=v
CTmesh.faces=f

cta_mesh = smoothing.filter_humphrey(CTmesh)

cta_voxel = cta_mesh.voxelized(1)
cta_voxel = cta_voxel.fill()





cta_voxel = cta_voxel.revoxelized((angiogram_crop.shape[2], angiogram_crop.shape[1], 20*10))
cta_bone = cta_voxel.matrix





angiogram_test = np.zeros((cta_bone.shape[2],cta_bone.shape[1],cta_bone.shape[0]))
for i in range(cta_bone.shape[2]):
    a = cta_bone[:,:,i]
    a = np.rot90(a, 3)
    a = np.fliplr(a)
    a[0, :] = 0
    a[:, 0] = 0
    angiogram_test[i] = a
    plt.imshow(a)
    plt.show()




for i in range(cta_bone.shape[2]):
    kernel = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(angiogram_test[i], cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing)
    plt.show()


# # End of test (Trimesh)




#Get original image metadate for copying to new dicom file

cta_image_path = os.path.join('original_images', 'CTA')
cta_slice_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(cta_image_path)
original_cta_image = sitk.ReadImage(cta_slice_filenames)

image_list = []

for slide in cta_slice_filenames:
    image_list.append(sitk.ReadImage(slide))





smooth = cv2.GaussianBlur(angiogram_crop,(3,3),0)
v2, f2 = make_mesh(smooth)
plotly_3d(v2, f2)

sitk_im = sitk.GetImageFromArray(smooth)





new_spacing = (original_cta_image.GetSpacing()[0], original_cta_image.GetSpacing()[1], original_cta_image.GetSpacing()[2])

sitk_im.CopyInformation(original_cta_image)

sitk_im.SetSpacing(new_spacing)
sitk_im.SetDirection(original_cta_image.GetDirection())
sitk_im.SetOrigin(original_cta_image.GetOrigin())


print(f'Spacing: {sitk_im.GetSpacing()}')
print(f'Direction: {sitk_im.GetDirection()}')
print(f'Origin: {sitk_im.GetOrigin()}')
print(f'Depth: {sitk_im.GetDepth()}')




angiogram_im = sitk_im
origin = angiogram_im.GetOrigin()
direction = angiogram_im.GetDirection()

space = angiogram_im.GetSpacing()
spacing = np.array([space[0], space[1], space[2]/2])

original_size = np.array(angiogram_im.GetSize())
original_spacing = np.array(angiogram_im.GetSpacing())

# since we want to preserve the whole image region, we can calculate the new size as
# the original size times the spacing ratio rounded to the nearest integer
new_size = np.floor(original_size * original_spacing / np.array(spacing)).astype(np.int).tolist()

# set up the resampling filter...
resampler = sitk.ResampleImageFilter()
resampler.SetOutputOrigin(origin)
resampler.SetOutputDirection(direction)
resampler.SetOutputSpacing(spacing)
resampler.SetSize(new_size)
resampler.SetInterpolator(sitk.sitkBSpline) # use BSpline interpolation

# ...and apply it to the image
resampled_image = resampler.Execute(angiogram_im)

angiogram_BSpline = sitk.GetArrayFromImage(resampled_image)


# # Generate DICOM Files




#Save as Dicom File
dcm_image = resampled_image

writer = sitk.ImageFileWriter()
# Use the study/seriers/frame of reference information given in the meta-data
# dictionary and not the automatically generated information from the file IO
writer.KeepOriginalImageUIDOn()

first_slice_location = float(image_list[0].GetMetaData('0020|1041'))
slice_thickness = float(image_list[0].GetMetaData('0018|0050'))/2
image_position = (image_list[0].GetMetaData('0020|0032')).split('\\')[:2]
thickness_between_slice = float(image_list[0].GetMetaData('0018|0088'))/2
UID = image_list[0].GetMetaData('0008|0018')[:-1]
Width_Attribute = float(image_list[0].GetMetaData('0018|9306'))/2


for i in range(dcm_image.GetDepth()):
    image_slice = dcm_image[:,:,i]
    original_slice = image_list[10]
    
    for k in original_slice.GetMetaDataKeys():
        image_slice.SetMetaData(k, original_slice.GetMetaData(k))
    
    slice_location = str(first_slice_location+i*slice_thickness)
    current_image_position = image_position[0]+'\\'+image_position[1]+'\\'+slice_location
    
    #Image Properties
    image_slice.SetMetaData('0020|1041', slice_location) #Slice location
    image_slice.SetMetaData('0020|0013', str(i+1)) #Instance Number
    image_slice.SetMetaData('0018|0050', str(slice_thickness)) #Slice Thickness
    image_slice.SetMetaData('0020|0032', current_image_position) #Current Image Position
    image_slice.SetMetaData('0018|0050', str(thickness_between_slice)) #Thickness Between Slices
    image_slice.SetMetaData('0008|0018', UID+str(i)) #SOP instance UID
    image_slice.SetMetaData('0018|9306', str(Width_Attribute)) #Width Attribute
    
    #Image names
    image_slice.SetMetaData('0010|0010', 'Automated Angiogram Segmentation') #Patient Name
    image_slice.SetMetaData('0010|0020', 'Angiogram') #Patient id
    image_slice.SetMetaData('0008|103e', 'Automated Angiogram Segmentation') #Series Description
    
    # Write to the output directory and add the extension dcm if not there, to force writing is in DICOM format.
    writer.SetFileName(os.path.join('angiogram_result', 'dicom', f"angiogram_slice_{i:04d}.dcm"))
    writer.Execute(image_slice)
print('Dicom Image successfully saved')


# # NRRD




#Saving as Nrrd
smooth_image = resampled_image
sitk.WriteImage(smooth_image, os.path.join('angiogram_result','nrrd','Angiogram.nrrd'))

print('NRRD File Saved Succesfully')


# # STL


#Save as STL file
v2, f2 = make_mesh(angiogram_BSpline)

vertices = v2
faces = f2

cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "cube.stl"
cube.save(os.path.join('angiogram_result','stl','angiogram_2.stl'))
print('STL File Saved Succesfully')

