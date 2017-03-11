
# coding: utf-8

# In[16]:

###Manual Vehicle Detection


# In[17]:

'''
You'll draw bounding boxes with cv2.rectangle() like this:

cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, thick)
'''


# In[18]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# get_ipython().magic('matplotlib inline')

image = mpimg.imread('bbox-example-image.jpg')


# In[19]:

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

# Here are the bounding boxes I used
bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]


# In[20]:


result = draw_boxes(image, bboxes)
plt.imshow(result)


# In[21]:

# https://github.com/npinto/opencv/blob/master/samples/python2/mouse_and_match.py


# In[25]:

templist = ['cutouts/cutout1.jpg', 'cutouts/cutout2.jpg', 'cutouts/cutout3.jpg',
            'cutouts/cutout4.jpg', 'cutouts/cutout5.jpg', 'cutouts/cutout6.jpg']

method = cv2.TM_CCOEFF_NORMED
bbox_list = []
def find_matches(img, template_list):
    # Iterate over the list of templates
    for temp in template_list:
        tmp = mpimg.imread(temp)
        # Use cv2.matchTemplate() to search the image for each template
        result = cv2.matchTemplate(img, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        img_small_HSV = cv2.cvtColor(tmp, cv2.COLOR_BGR2HLS)

    return bbox_list




# In[ ]:



