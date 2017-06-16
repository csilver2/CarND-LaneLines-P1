# **Finding Lane Lines on the Road** 

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: Result/whiteCarLaneSwitch_r.png "result"

---



## My pipeline has seven steps to follow before we can make lines in a video.

### 1. grayscale, gaussian_blur

First, we edit the functions grayscale, and gaussian_blur, in order to make easier for the canny function to detect edges in images:
```
def grayscale(img):
return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
### 2. Canny

In order to detect the edges, we are using the function canny from the CV library wich detects the edges in the images depending on the contrast or different valiues in a group of pixels.
```
def canny(img, low_threshold, high_threshold): 
return cv2.Canny(img, low_threshold, high_threshold)
```
### 3. Masked Image

The next Function cut the image depending on the vectors given. This limits the work area.
```
def region_of_interest(img, vertices):
    """Applies an image mask."""
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
### 4. Drawing Lines with Hough transform

Using the function HoughLinesP we can find lines connecting the points from the canny function, then draw lines over the image with 
the function cv2.lines.
```
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
    
def draw_lines(img, lines, color=[250, 0, 0], thickness=10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    ```
### 5. Weighted image

Finally, we take both images the one result of the hough_lines function and the image the image from the beginning of the process. with
this code we can draw the lines over the RGB image.
```
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    ```
### 6. Coding the Pipeline

Now we can call all the function and set up the variables and values to test the result in a images or Video. With this code, we should obtain an image that shows the way with all the lines marked on red.

```
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
#img = mpimg.imread('test_images/solidWhiteRight.jpg')

imshape = img.shape

vert = np.array([[[130, 540], [410, 350], [570, 350], [915, 540]]], dtype=np.int32)

#functions
gray = grayscale(img)
blur = gaussian_blur(gray,5)
edges = canny(blur, 100, 200)
mask = region_of_interest(edges, vert)
h_transform = hough_lines(mask, 1, np.pi/180, 30, 20, 20)
lines_edges = weighted_img(h_transform, img,α=0.8, β=1., λ=0)
```

plt.imshow(lines_edges)

### 7. Connect/Average/Extrapolate

To obtain two lines  I edited the function drawing_lines with the following Code:
7.0. creating the values where the variables for (x,y) 
 
 Note: I used the method recommended from udacity help videos for term one projects.

7.1. Iterating the x/ y values on the slope and center equations.

7.2. Separating the lines using the different values given by the slope for each group of lines.

7.3. Then we sum all the data obtained to make an average of the (x,y), slope, and center.

7.4. Now with the mean values of the slope=M, and center points = (x',y'), the code it's almost done, Just need to set up the value
of the (x,y) to use it int general lines form (y-y')=M(x-x'). The easiest values to find is "y's" wich we know are the coordinates in the middle and lower border of the image.

7.3. Finally using this values we can obtaing the x using the form (y-y')=M(x-x') ==> x = (y-y')/m + x'



```
imshape = img.shape

#(7.0.)
    
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    
    left_slope=[]

    left_slope=[] 
    right_slope=[]

    left_center=[]
    right_center=[]

#(7.1.)
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            center = [(x2+x1)/2 ,(y2+y1)/2]
            ymin_global = min(min(y1, y2), ymin_global) # #(7.5.)
 
#(7.2.) 
            if (slope > 0):
                left_slope += [slope] 
                left_center += [center]
            else:
                right_slope += [slope]
                right_center += [center]
                
  #(7.3.)
                
    av_l_slope = np.sum(left_slope)/len(left_slope)
    av_r_slope = np.sum(right_slope)/len(right_slope)
    
    av_l_center = np.divide(np.sum(left_center,axis=0),len(left_center))
    av_r_center = np.divide(np.sum(right_center,axis=0),len(right_center))

#(7.4.)
    if ((len(left_slope) > 0) and (len(right_slope) > 0)):
        upper_left_x = int(((ymin_global - av_l_center[1]) / av_l_slope) + av_l_center[0])
        lower_left_x = int(((ymax_global - av_l_center[1]) / av_l_slope) + av_l_center[0])
        upper_right_x = int(((ymin_global - av_r_center[1]) / av_r_slope) + av_r_center[0])
        lower_right_x = int(((ymax_global - av_r_center[1]) / av_r_slope) + av_r_center[0])
    
        cv2.line(img, (upper_left_x, ymin_global), (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), (lower_right_x, ymax_global), color, thickness)
 ```
This is my result

![alt text][image1]


## Shortcomings
The biggest problem with my project is that doesn't detect curves.

## Possible Improves.
There are too many ways to make lines with this project, and separating the lines with the slope is good when you have straight lines with not vertical or horizontal slope. But in other cases I think should be better to separate lines using coordinates in the image.
