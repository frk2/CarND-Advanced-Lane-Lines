import cam_calibrate
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

window_width = 50 
window_height = 120 # Break image into 9 vertical layers since image height is 720
margin = 30 # How much to slide left and right for searching
    
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    print (warped.shape)
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
      # convolve the window into the vertical slice of the image
      image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
      conv_signal = np.convolve(window, image_layer)
      # Find the best left centroid by using past left center as a reference
      # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
      offset = window_width/2
      l_min_index = int(max(l_center+offset-margin,0))
      l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
      l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
      # Find the best right centroid by using past right center as a reference
      r_min_index = int(max(r_center+offset-margin,0))
      r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
      r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
      # Add what we found for that layer
      window_centroids.append((l_center,r_center))

    return window_centroids

def colorBinarizeImage(img, s_thresh=(200, 255), sx_thresh=(20, 70)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return color_binary, combined_binary

def binarizeImage(image):
  ksize = 9 # Choose a larger odd number to smooth gradient measurements
  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
  
  s_channel = hsv[:,:,2]
  # Threshold color channel
  s_binary = np.zeros_like(s_channel)
  s_thresh=(200, 255)
  s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
  # cv2.imshow('s_binary', s_binary.astype(np.float32))

  gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(40, 100))

  # cv2.imshow('gradx', gradx.astype(np.float32))
  grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(40, 100))
  # cv2.imshow('grady', grady.astype(np.float32))
  mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(40, 120))
  # cv2.imshow('magthresh', mag_binary.astype(np.float32))
  dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.1))
  # cv2.imshow('dirthresh', dir_binary)
  combined = np.zeros_like(dir_binary)
  combined[((gradx == 1) & (grady == 0)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
  # cv2.imshow('combined', combined)
  return combined

def abs_sobel_thresh(img, orient='x', thresh=(0,255), sobel_kernel=3):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def drawpoly(img, vertices, color):
  #verts = vertices.reshape((-1,1,2))
  cv2.polylines(img, [vertices], True, color)

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
  
  return img, warped
  #cv2.line(img, tuple(srcverts[0]), tuple(srcverts[1]), [0,0,255], 2)

def paintWindows(warped, window_centroids):
  if len(window_centroids) > 0:
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows  
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
      l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
      r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
      # Add graphic points from window mask here to total pixels found 
      l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
      r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channle 
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
  # If no window centers found, just display orginal road image
  else:
      output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

  return output

def fitLines(img, centroids):
  lxpoints = []
  rxpoints = []
  ypoints = []
  for level in range(0, len(centroids)):
    ypoints.append(img.shape[0] - 1 - level * window_height)
    lxpoints.append(centroids[level][0])
    rxpoints.append(centroids[level][1])

  ypoints = np.array(ypoints)
  rxpoints = np.array(rxpoints)
  lxpoints = np.array(lxpoints)

  left_fit = np.polyfit(ypoints, lxpoints, 2)
  right_fit = np.polyfit(ypoints, rxpoints, 2)

  ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
  y_eval = np.max(ploty)
  # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
  # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension

  # Fit new polynomials to x,y in world space
  left_fit_cr = np.polyfit(ypoints*ym_per_pix, lxpoints*xm_per_pix, 2)
  right_fit_cr = np.polyfit(ypoints*ym_per_pix, rxpoints*xm_per_pix, 2)
  # Calculate the new radii of curvature
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

  return left_fit, left_curverad, right_fit, right_curverad

def drawLaneOnImage(img, warped, left_fitx, right_fitx, ploty):
  # Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, getPerspectiveMatrix(img, inverted=True), (img.shape[1], img.shape[0])) 
  # Combine the result with the original image
  result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
  return result;

class ImageProcessor:
  def __init__(self):
    self.mtx, self.dist = cam_calibrate.calibrateCamera('camera_cal/calibration*.jpg')
    
  def processImage(self, img):
    undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    #cv2.imshow('img', undist)
    orig, warped = transformToBEV(binarizeImage(undist))
    #morig, colorwarped = transformToBEV(undist, debug=True)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    #windows = paintWindows(warped * 255, window_centroids)
    left_fit, left_curverad, right_fit, right_curverad = fitLines(warped * 255, window_centroids)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    plotted = drawLaneOnImage(undist, warped, left_fitx, right_fitx, ploty)
    cv2.putText(plotted, 'Curvatute L:{0:.2f} R:{0:.2f}'.format(left_curverad, right_curverad), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), thickness=2 )
    return plotted
    # plt.imshow(plotted)
    # plt.show()
    # plt.imshow(windows)
    # plt.plot(left_fitx, ploty, color='red', linewidth=4)
    # plt.plot(right_fitx, ploty, color='red', linewidth=4)
    # plt.text(1,1, 'Curvature is l: {} r:{}'.format(left_curverad, right_curverad))
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    # cv2.imshow('orig', orig)
    # cv2.imshow('binary', colorwarped)
    # # Display the final results
    # cv2.imshow('boo',windows)
    # cv2.waitKey(40000)
    # cv2.destroyAllWindows()
  
def main():
  #Just try a sample first
  
  processor = ImageProcessor()
  img = cv2.imread('test_images/test5.jpg')

  clip = VideoFileClip('project_video.mp4')
  out_clip = clip.fl_image(processor.processImage)
  out_clip.write_videofile('output.mp4',audio=False)
  return 0
  # my code here




if __name__ == "__main__":
  main()
