# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

CarND-Vehicle-Detection is a software pipeline to identify vehicles in a video from a front-facing camera on a car. It uses Histogram of Oriented Gradients (HOG) and OpenCV for feature extraction and trains a classifier using SVM.

[//]: # "Image References"
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/sdcnd_vehicle_detection_gif.gif
[video1]: ./project_video.mp4

![alt text][image7]
(https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Here is video in action:
[![Autonomouse Cars : Vehicle Detection](http://img.youtube.com/vi/CtctLQaF_s4/0.jpg)](https://www.youtube.com/watch?v=CtctLQaF_s4)


Small GIF if you don't want to see video:
![alt text][image8]

The Project Goals
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Dependencies

- Keras
- OpenCV
- Tensorflow
- Python 3.5

## Dataset

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

## Installation

Clone the GitHub repository and use Bundler to install the gem dependencies.

```
$ git clone https://github.com/github/ledbetter.git
$ cd ledbetter
$ bundle install

```

## Usage

For deep learning or support vector machine method, run one of the following:

```
source active carnd-term1
jupyter notebook VehicleDetectionWorkbook.ipynb
```

## Introduction to object detection

Detecting vehicles in a video stream is an object detection problem. An object detection problem can be approached as either a classification problem or a regression problem. As a classification problem, the image are divided into small patches, each of which will be run through a classifier to determine whether there are objects in the patch. Then the bounding boxes will be assigned to locate around patches that are classified with high probability of present of an object. In the regression approach, the whole image will be run through a convolutional neural network to directly generate one or more bounding boxes for objects in the images.

| classification                           | regression                               |
| ---------------------------------------- | ---------------------------------------- |
| Classification on portions of the image to determine objects, generate bounding boxes for regions that have positive classification results. | Regression on the whole image to generate bounding boxes |
| 1. sliding window + HOG 2. sliding window + CNN 3. region proposals + CNN | generate bounding box coordinates directly from CNN |
| RCNN, Fast-RCNN, Faster-RCNN             | SSD, YOLO                                |

## Classifiers

### Support Vector Machine

The Support Vector Machine used in this is scikit-learn's LinearSVC. It runs on modest hardware at about 1.5 seconds/frame and has lots of room for additional performance gains. It was trained on subsets of the GTI and KITTI, as well as manually extracted negative examples from the Udacity set. The latter helped in reducing false positives in areas with a lot of information (trees and complex shadows). It is trained in lines 90-143 of `vehicleDetect_classify.py` and then pickled for reuse. It was trained on 8,792 car and 10,291 non-car images.

### Deep Learning

I am using a fairly slim convolutional neural network that has previously performed well on CIFAR10. It runs at **8fps** on modest hardware. Since all I need here is a binary decision on small images, I expected it to perform reasonably well. 

## Vehicle Labeling

Both SVM and CNN methods use the same approach for finding and labeling vehicles. Sliding windows are used to feed images to the classifier. A heat map is generated over several frames, then gets filtered and labelled.

### Sliding Window Search

To start, I am limiting my search to the right half of the frame, compensating for the lack of data about the road to focus attention properly (my lane finding code was eventually merged with this vehicle detection, but they do not communicate yet). This covers detections in my video and allowed for faster iterations.

I perform a simple sliding window search to extract cropped images for classification. For efficiency, the window locations are computed once on the first frame and then reused (lines 184-199 in `vehicleDetect.py`). I use different sizes of windows at different locations in the image. Per frame, the saved windows are used to extract cropped images via `vehicleDetectUtil.get_window_imgs()` (line 200 in `vehicleDetect.py`) and then sent to the classifier all at once.

The SVM method uses 457 windows (overlap: 0.8), while the CNN detection only uses 76 (overlap: 0.7).

*Sliding windows (CNN). Larger window sizes closer to the bottom and 0.7 overlap. 76 total.*
[![Sliding windows (CNN)](https://github.com/merbar/CarND-Vehicle-Detection/raw/master/output_images/windows1.jpg)](https://github.com/merbar/CarND-Vehicle-Detection/blob/master/output_images/windows1.jpg)

### Heat Map

For each frame, every bounding box that is detected as part of a vehicle by **two** or more bounding boxes adds "heat" to a map. The result is smoothed over eight frames, further thresholded and then fed into scipy's label function to get a single bounding box for each distinct "island" in the heat map (lines 206-221 in `vehicleDetect.py`).

### False Positive Suppression

I am doing two things to prevent false positives:

- As discussed above, the heat map is thresholded twice
- Additionally, I am ignoring bounding boxes with unreasonable aspect ratios that come out of the label function (lines 225-230 in `vehicleDetect.py`)

### Bounding Box Smoothing

I implemented a Car class that keeps track of positions over time (lines 51-87 in `vehicleDetect.py`). Bounding box coordinates are smoothed over six frames. Additionally, I reuse the previous location if the classifier does not detect the car for up to two frames.

*Example frame with labeled image (top left), heat map (top right) and unfiltered sliding-windows detection (bottom right)*
[![Example frame](https://github.com/merbar/CarND-Vehicle-Detection/raw/master/output_images/labeling1.jpg)](https://github.com/merbar/CarND-Vehicle-Detection/blob/master/output_images/labeling1.jpg)

------

## Video Results

- [Deep Learning Solution w/ lane detection](https://youtu.be/IHhLpB5MNTQ) | [w/ additional data](https://youtu.be/DTaLG2DSjyU)
- [SVM solution](https://youtu.be/DOCzH0R3ERc)

------

## Discussion

- Setting up a neural network and training it from scratch for this turned out to be a very straight-forward process that yielded great results fairly quickly.
- The SVM method should be able to run a lot faster. In order to speed it up, I started scaling the image instead of scaling the individual crops - also to set up extracting HOG features from only a single image. In hindsight, this turned out fairly messy and slowed down my iteration time of tweaking the sliding window boundaries due to the unintuitive scaled image sizes. It started to be too time-consuming and I left it where it is right now in order to focus on the deep learning solution.
- Keeping track of individual cars in screen space was impractical to do with just scipy's label method. It does not consistently apply the same label to similar islands in the heat map, so I had to put additional checks in my car class to reset it in case labels flip. A custom label function that gets to draw on more information about the existing car instances and can make reasonable assumptions about how traffic moves should perform well here.
- Neither method fits a tight bounding box around the cars at all times. In order to increase accuracy and speed, I would look into making it a two-step process: A quick initial detection of candidate areas, followed by a very localized sliding-window search. Previously detected cars could be re-used on each frame for additional speedup.
- A search on the entire image using a large neural net would be very interesting to look into more. I tried a pre-trained YOLO-tiny with some success, and it runs very fast (20fps+). It was very sensitive to aspect ratio changes, so it required square inputs and could not be fed the entire image at once. It also failed to detect cars once they did not take up a fairly large part of the frame - meaning it would have required it's own sliding windows extraction. That, in turn, required sending many more images per frame to the classifier. I expected it to perform about as fast as my existing neural net, but I did not have time to test that hypothesis.

##### 



