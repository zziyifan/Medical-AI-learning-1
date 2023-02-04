Original
####################################################################################################################
HCP1200 Deep Learning Teaching Dataset

Author: Ali Khan (alik@robarts.ca)

Single 2D axial slice taken from pre-processed T1w data, with masks coming from Freesurfer. Data are in PNG format, with intensities from 0-255 (foreground=255 for masks).
Downsampled and resized to 160x160 pixels, dataset is small enough to easily train models on Google Colab.

915 subjects in training, 197 subjects in test

images: T1w images
brain_masks: binary brain masks
ventricle_masks: binary ventricle masks (lateral ventricles)





Truncated
#################################################################################################################
Just to let the code run, preserved:
3 set of images in training set;
2 set of images in testing set.

potentially others are generated from code, like labels 

