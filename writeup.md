
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_nocar.png
[image2]: ./output_images/car_nocar_hog.png
[image3]: ./output_images/hog.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/test1.jpg
[image6]: ./output_images/test2.jpg
[image7]: ./output_images/test3.jpg
[image8]: ./output_images/test5.jpg
[image9]: ./output_images/test6.jpg

[image10]: ./output_images/heat_test1.jpg
[image11]: ./output_images/heat_test2.jpg
[image12]: ./output_images/heat_test4.jpg
[image13]: ./output_images/heat_test5.jpg
[gif1]: ./output_images/heatmap_corner.gif
[gif2]: ./output_images/heatmap_overlay.gif
[gif2]: ./output_images/bounding.gif
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

Codes are in `explore_models.ipynb`, Section 2. Extract Hog.

#### 1. Extract HOG features from the training images.

The code for extracting hog feature from an image is in the function `get_hog_features`, in lines 3 through 17 of the file called `utils.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored HOG in different color spaces.  
![alt text][image3]

I tried different HOG parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Increasing values of the three will reduce number of features. See code cell 10 in `explore_models.ipynb`.

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Choice of HOG parameters.

Cell 9~19 in `explore_models.ipynb`

I tried various combinations of parameters and applied HOG to different combination of color spaces and channels. I found increasing the values of `orientations`, `pixels_per_cell`, and `cells_per_block` will reduce the accuracy of our classifier. Thus I used `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. I tried several combination of channels:

| Space:Channel  |  Accuracy |
|---|---|
| BGR | 0.9634 |
| Y of YUV, Green, V of HSV  |  0.9645 |
| Y of YUV, Green, V of HSV  |  0.9645 |
| Y of YUV, Green, V of YUV  |  0.9764 |
| H of HSV, Green, V of HSV  |  0.9786 |
| H of HSV, Blue, Red V of HSV  |  0.9831 |
| H of HSV, Blue, Green, Red V of HSV  |  0.9831 |
| HSV | 0.9806 |

Since HSV provided good accuracy without adding much computation. I will use HSV as my channel. Since I'm using HSV for lane finding as well. Combining the two will save some time.

#### 3. Trained a classifier with HOG features and color features.
Cell 20~21 in `explore_models.ipynb`

I found including histogram features and spatial features can boost the accuracy to 0.9916. So I choose to use HOG of HSV channels together with histogram features and spatial features to train an SVM model.

Here are the parameters I used:

"orient": 9,
"pix_per_cell": 8,
"cell_per_block": 2,
"spatial_size": (16,16),
"hist_bins": 32

I used `StandardScaler()` to renormalize the data.

Then I trained a linear SVM using 4/5 of the entire dataset.

### Sliding Window Search

Codes are in `detect_image.ipynb`

#### 1. Sliding window search.

Cell 5-6 of `detect_image.ipynb`. Slide window Section.

I decided to search windows at two different scales shown as below:

![alt text][image4]

#### 2. Show some examples of test images.

Ultimately I searched on two scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

For the larger scale window, I reduced the x-overlapping rate to 0.5, and it seems to be enough.

Below are examples of heatmap generated by counting number of windows over each pixels

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]
---

### Video Implementation

Codes are contained in `detect_video.ipynb`

#### 1. Final video.
Here's a [link to my video result](https://youtu.be/0mxZHE_h5mM)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The following codes in `VehicleLaneFinder.pipeline` are used to filter out noisy false positive along time.
```
heat = np.zeros_like(img[:,:,0]).astype(np.float)
# Add heat to each box in box list
heat = add_heat(heat,windows)
self.vehicle_heatmap = np.roll(self.vehicle_heatmap,shift=(0,0,-1))
self.vehicle_heatmap[:,:,-1] = heat

# Apply threshold to help remove false positives
heat = apply_threshold(np.sum(self.vehicle_heatmap,axis=2),6)
```

I recorded the positions of positive detections in each frame of the video.  From the positive detections of the most recent 10 frames, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][gif1]

![alt text][gif2]

### Here the resulting bounding boxes:
![alt text][gif3]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although I have done all suggested tricks (subsample HOG) to improve speed, because computing HOG is quite expensive, the pipeline is still slow.
Also it was time consuming to explore different choices of parameters and features.

A better approach would be a CNN based method such as YOLO. Convolution is faster to compute and involves less manual tuning of parameters.
Another thing I would like to do is to use more data (such as generating non-car images from empty road images) to train the model.
