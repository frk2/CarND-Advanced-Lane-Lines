
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1b]: ./output_images/undistored.jpg "Undistorted"
[image1a]: ./output_images/calibration1.jpg "Distorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

###Camera Calibration
I used a separate file (cam_calibrate.py) just because I wanted a separate usable peice of code and wanted to play around with python. This calibrates the camera and writes the calibration matrix to a pickle file. The next time, the matrix is returned straight from the pickle file instead of calibrating over and over again.

Here is the calibration matrix applied to one of the calibration images themselves:
![alt text][image1a]
![image1b]

##Pipeline

The pipeline reads an image at a time and applies multiple transformations to get it into a state where I can use `np.convolve` and fit a polynomial. This is then cached in a `Line` class which additionally averages the input over 10 images. The Line class is responsible for providing the best fit in cases where the current fit is unsuitable.

##### Undistort
I use the cached calibration matrix and undistort each image such that it looks like:
![alt text][image2]

##### Image transformation
This was quite interesting as different thresholding techniques work for different conditions. I discovered that the s-channel in HLS color space is key to finding the yellow line. The line that does the trick is:
```
combined[((gradx == 1) & (grady == 0)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
```

I'm basically looking for a point where the Y sobel is 0 while the X is 1. The s-channel info is so important that it trumps all the other channels and is hence or'ed alone. Another important observation was that the direction gradient is pretty sensitive to the threshold. From trial and error I arrived at a dir_thresh of `0.7 - 1.1`

![alt text][image3]

#### Perspective Transormation

I used a single function which returns the perspective transform matrix since this is used twice in the code. I took the matrix from the example provided and with a minor tweak it mostly works! I would like to figure out how to do this in a more mathematical way rather than just guess!

```
def getPerspectiveMatrix(img, inverted=False, debug=False):
  img_size = (img.shape[1], img.shape[0])
  srcverts = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 16), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
  dstverts = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]]) 
  
  color = 0.5
  if (debug):
    if (len(img.shape) > 2):
      color = (0,0,255)
    drawpoly(img, srcverts.astype(int), color)

  if (inverted):
    return cv2.getPerspectiveTransform(dstverts, srcverts)
  else:
    return cv2.getPerspectiveTransform(srcverts, dstverts)  

def transformToBEV(img, debug=False):
  
  warped =  cv2.warpPerspective(img, getPerspectiveMatrix(img,debug=debug), (1280,720))
```
I verified the correctness of the transform by enabling 'debug' above which draws the polygon onto the src image. Once warped, this polygon must resemble a rectangle mostly and it does!

![alt text][image4]

##### Finding those lines

I used np.convolve as provided in the lecture slides but I quickly realized that a few optimizations were needed. Firstly, I wanted it to start from the last location of the best fit line. My window searching algo first looks if we have a good fit and starts the search from there if we do:

```
if (self.left_line.best_fit != None):
      fy = np.poly1d(self.left_line.best_fit)
      l_lookupcenter = fy(warped.shape[0]-1)
      lookupmargin = margin*3
      leftxs = max(int(l_lookupcenter - lookupmargin), 0)
      leftxe = min(int( l_lookupcenter + lookupmargin), warped.shape[1]/2)
      l_sum = np.sum(warped[int(3*warped.shape[0]/4):,  leftxs : leftxe ], axis=0)
      l_center = np.argmax(np.convolve(window,l_sum))-window_width/2 + (l_lookupcenter - lookupmargin)
      
    else:
      l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
      l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
```
The other crucial modification was to only mark a window if the convolved sum is greater than some amount. What can happen is that in dark times (pun intended - where we are all black and see no lines) the convolution returns all zeros and the argmax simply picks the first window. This causes a slide of the window by margin units to the left which is completely wrong! It would be way better if the window was simply in the exact same location as the last one if we couldn't find anything. This is accomplished by putting a minimum threshold on the convolved sum:
```
if (np.sum(conv_signal[l_min_index:l_max_index]) > 100.0):
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
```

I'm sure we can do better than some arbitrary minimum - but 100 seemed to work.

![alt text][image5]

#### Curvature
The radius of curvature is calculated
I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
