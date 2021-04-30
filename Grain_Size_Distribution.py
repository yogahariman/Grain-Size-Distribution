#%%
# pip install opencv-python
#%matplotib inline
# import imutils
import cv2
import math
import numpy as np
# from scipy.ndimage import label, generate_binary_structure
# pip install scikit-image
from skimage import measure
import matplotlib
matplotlib.use('TKAgg')
# print(matplotlib.rcsetup.all_backends)
# print(plt.get_backend())
import matplotlib.pyplot as plt
#%%
# define pixel resolution micron/pixel 
# this is the spatial resolution of the input 
resolution=0.9243548387096774; # micron/pixel

# Read image
img_original = cv2.imread("images/image_01.png")
#%% convert image rgb to gray scale
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
#%%
# Apply GaussianBlur to reduce image noise if it is required
# img_gaussian=cv2.GaussianBlur(img_gray,(5,5),0) 

# Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
# Use a bimodal image as an input.
# Optimal threshold value is determined automatically.
otsu_threshold, img_threshold = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#%% noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
img_opening = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
# img_closing = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel, iterations=1)

# sure background area
img_sure_bg = cv2.dilate(img_opening,kernel,iterations=5)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(img_opening,cv2.DIST_L2,3)
ret, img_sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
img_sure_fg = np.uint8(img_sure_fg)
img_unknown = cv2.subtract(img_sure_bg,img_sure_fg)

# Marker labelling
ret, img_markers = cv2.connectedComponents(img_sure_fg)
# Add one to all labels so that sure background is not 0, but 1
img_markers = img_markers+1
# Now, mark the region of unknown with zero
img_markers[img_unknown==255] = 0

markers = cv2.watershed(img_original,img_markers)
markers[markers==-1] = 0
markers[markers==1] = 0
# img_original[markers == -1] = [255,0,0]

img_label = measure.label(markers)
props = measure.regionprops(img_label)

#%% grain radius
radius = [resolution*(prop.area/math.pi)**0.5 for prop in props]
plt.figure(figsize=(6.4*2, 4.8*2))
n,bins,patch = plt.hist(radius)
plt.xlabel('Equivalent Particle Radius (micron)')
plt.ylabel('Relative Frequency')
plt.show(block=True)

#%% Plotting
plots = [['Original',img_original,None],
          ['Grayscale',img_gray,'gray'],
          #['Gaussian', img_gaussian, 'gray'],
          ['Threshold', img_threshold, 'gray'],
          ['Opening', img_opening, 'gray'],
          ['Label', img_label, 'jet'],
          #['Contours', img_contours, 'gray'],
          ]

fig, axes = plt.subplots(2, math.ceil(len(plots)/2),  figsize=(6.4*3, 4.8*3))
ax = axes.ravel()
for n, plot in enumerate(plots):
    ax[n].imshow(plot[1], cmap=plot[2])
    ax[n].axis('off')
    ax[n].set_title(plot[0])
fig.tight_layout()
plt.show(block=True)

#%% Plot Centroid
plt.imshow(img_original)
for prop in props:
    plt.plot(*prop.centroid[::-1], marker='x', color='r')
    print(f'Label: {prop.label} >> Object size: {prop.area}')
plt.show(block=True)
#%%
# cv2.imshow("Image", np.uint8(img_markers))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #%%

# img_label = measure.label(img_opening==0)
# props = measure.regionprops(img_label)
# plt.imshow(img_label)

# plt.imshow(img_original)
# for prop in props:
#     plt.plot(*prop.centroid[::-1], marker='x', color='r')
#     print(f'Label: {prop.label} >> Object size: {prop.area}')
# plt.show(block=True)

# # # contours,hierarchy = cv2.findContours(img_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # contours,hierarchy = cv2.findContours(img_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # print("Number of Contours found = " + str(len(contours)))

# # # contours = imutils.grab_contours(contours)
# # # loop over the contours
# # for ind, c in enumerate(contours):
# #     # compute the center of the contour
# #     M = cv2.moments(c)
# #     cX = int(M["m10"] / M["m00"])
# #     cY = int(M["m01"] / M["m00"])
# #     # draw the contour and center of the shape on the image
# #     cv2.drawContours(img_original, [c], -1, (0, 255, 0), 2)
# #     cv2.circle(img_original, (cX, cY), 7, (255, 255, 255), -1)
# #     cv2.putText(img_original, "center", (cX - 20, cY - 20),
# #     	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
# #     # show the image
# #     cv2.imshow("Image", img_original)
# #     cv2.waitKey(0)
# #     print(ind)
# # cv2.destroyAllWindows()


# # # Draw all contours
# # # -1 signifies drawing all contours
# # cv2.drawContours(img_opening, contours, -1, (0, 255, 0), 3)
# # cv2.imshow('Contours', img_original)
# # cv2.waitKey(0)  
# # cv2.imshow('Contours', img_opening)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # # find largest area contour
# # max_area = -1
# # for i in range(len(contours)):
# #     area = cv2.contourArea(contours[i])
# #     if area>max_area:
# #         cnt = contours[i]
# #         max_area = area
# # contours = contours[0] if len(contours) == 2 else contours[1]
# # cv2.drawContours(img_opening, contours, -1, (0,255,0), 2)
# # plt.show(block=True)
# # area=[]
# # for cnts in contours:
# #     area.append(cv2.contourArea(cnts))
# #     # if area < 150:
# #     #     cv2.drawContours(opening, [cnts], -1, (0,0,0), -1)
# # plt.hist(area)
# # plt.show(block=True)
# # props = measure.regionprops(img_opening)
# # area = [prop.area for prop in props]
# # plt.hist(area)
# #%%
# # plots = [['Original',img_original,None],
# #           ['Grayscale',img_gray,'gray'],
# #           ['Gaussian', img_gaussian, 'gray'],
# #           ['Threshold', img_threshold, 'gray'],
# #           ['Opening', img_opening, 'gray'],
# #           #['Closing', img_closing, 'gray'],
# #           #['Contours', img_contours, 'gray'],
# #           ]

# # fig, axes = plt.subplots(2, math.ceil(len(plots)/2),  figsize=(6.4*3, 4.8*3))
# # ax = axes.ravel()
# # for n, plot in enumerate(plots):
# #     ax[n].imshow(plot[1], cmap=plot[2])
# #     ax[n].axis('off')
# #     ax[n].set_title(plot[0])
# # fig.tight_layout()
# # plt.show(block=True)


# # contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # img = cv2.drawContours(res_img, contours, -1, (0,0,0), -1)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# # for c in cnts:
# #     area = cv2.contourArea(c)
# #     if area < 150:
# #         cv2.drawContours(opening, [c], -1, (0,0,0), -1)
        
# # plt.imshow(opening, cmap='gray')
# # plt.show(block=True)

# # #%% convert image gray scale to binary scale
# # # Otsu's thresholding after Gaussian filtering
# # ret, img_thresh = cv2.threshold(img_gray_gaussianimg_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # #%%
# # # kernel = np.ones((3,3),np.uint8) 
# # # img_eroded = cv2.erode(img_thresh,kernel,iterations = 1)
# # # img_dilated = cv2.dilate(img_eroded,kernel,iterations = 1)
# # #%%
# # img_binary = img_thresh == 0
# # # Generate a structuring element that will consider features connected even if they touch diagonally:
# # # s = scipy.ndimage.generate_binary_structure(2,2)
# # # img_labeled, num_features  = scipy.ndimage.label(img_binary, structure=s)
# # img_labeled  = measure.label(img_binary, background=0, connectivity=2)
# # img_labeled_rgb = color.label2rgb(img_labeled, bg_label=0)

# # props = measure.regionprops(img_labeled)
# # area = [prop.area for prop in props]
# # plt.hist(area)
# # #%% Plot Image
# # fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(25.6, 19.2))
# # ax[0, 0].imshow(img_rgb)
# # ax[0, 0].title.set_text('Original')
# # ax[0, 1].imshow(img_gray,'gray')
# # ax[0, 1].title.set_text('Gray Scale')
# # ax[1, 0].imshow(img_thresh,'gray')
# # ax[1, 0].title.set_text('Threshold')
# # # ax[1, 1].imshow(img_eroded,'gray')
# # # ax[1, 1].title.set_text('Erode')
# # # ax[1, 2].imshow(img_dilated,'gray')
# # # ax[1, 2].title.set_text('Dilate')
# # ax[2, 0].imshow(img_binary,'gray')
# # ax[2, 0].title.set_text('Binary')
# # ax[2, 1].imshow(img_labeled)
# # ax[2, 1].title.set_text('Labeled')
# # plt.show()
