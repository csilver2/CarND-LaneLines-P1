# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

## introduccion. My pipeline has five steps to folow after make lines in the video.

### 1. grayscale, gaussian_blur

first we edit the functions grayscale, and gaussian_blur, in order to make easyer for the canny operation to detect edges in the images:
```
def grayscale(img):
return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
### 2. canny
```
def canny(img, low_threshold, high_threshold): 
return cv2.Canny(img, low_threshold, high_threshold)
```
### 3. Masked Image
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
### 4. 
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
### 5.
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
### 6. 
```
imshape = img.shape
    
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    
    left_slope=[]

    left_slope=[] 
    right_slope=[]

    left_center=[]
    right_center=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            center = [(x2+x1)/2 ,(y2+y1)/2]
            ymin_global = min(min(y1, y2), ymin_global)
            
            if (slope > 0):
                left_slope += [slope] 
                left_center += [center]
            else:
                right_slope += [slope]
                right_center += [center]
                
    av_l_slope = np.sum(left_slope)/len(left_slope)
    av_r_slope = np.sum(right_slope)/len(right_slope)
    
    av_l_center = np.divide(np.sum(left_center,axis=0),len(left_center))
    av_r_center = np.divide(np.sum(right_center,axis=0),len(right_center))

    if ((len(left_slope) > 0) and (len(right_slope) > 0)):
        upper_left_x = int(((ymin_global - av_l_center[1]) / av_l_slope) + av_l_center[0])
        lower_left_x = int(((ymax_global - av_l_center[1]) / av_l_slope) + av_l_center[0])
        upper_right_x = int(((ymin_global - av_r_center[1]) / av_r_slope) + av_r_center[0])
        lower_right_x = int(((ymax_global - av_r_center[1]) / av_r_slope) + av_r_center[0])
    
        cv2.line(img, (upper_left_x, ymin_global), (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), (lower_right_x, ymax_global), color, thickness)
 ```

![image2] (https://csilver2.github.com/CarND-LaneLines-P1/Result/whiteCarLaneSwitch_r.png)
