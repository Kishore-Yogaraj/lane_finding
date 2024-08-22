import cv2
import numpy as np
import matplotlib.pyplot as plt #Library used to display images visually with x and y axis


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Function that takes an image and converts it to another colour space
    #Will put a gaussian blur on the image to reduce noise and detect edges accurately
    #It will use a 5x5 kernel to do this and 0 is the deviation
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #Canny function detects the edges on the image and returns a gradient image
    #50 is the lower threshold and 150 is the upper threshold
    #This means that if the gradient(difference in intensity of 2 pixels) between two pixels is greater than 150
    #It is an edge, if lower than 50 it is not and edge if inbetween it checks to see if the surround 8 pixels are
    #A detected edge, if it is, then it also becomes an edge
    canny = cv2.Canny(blur, 50, 150)
    return canny #We are returned the gradient image with edges detected

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines: #Access each block (there are 9 blocks of (1,4) arrays)
            x1, y1, x2, y2 = line.reshape(4) #Assing point to 1D arrays
            #line() draws a line segment connecting 2 points
            #line(image (graph), point 1, point 2, colour, thickness)
            cv2.line(line_image, (x1, y1), (x2,y2), (255, 0, 0), 10)
    return line_image #return the black image with lines drawn


def region_of_interest(image): #input is the edge detected image
    height = image.shape[0] #find the total heigh of the image (what is the column value at index row 0)
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ]) #Vertices of region of interest - vertices we determined by observation using matplotlib
    mask = np.zeros_like(image) #Creates an array of zeros (all black image) with the same shape as the image's corresponding array

    #Arguments are the masked image, multiple polygons, colour to fill
    cv2.fillPoly(mask, polygons, 255) #Fills array 'mask' with the coordinates of 'triangle' with the colour white

    #Computing the bitwise & of both images takes the bitwise & of each homologous pixel in both arrays,
    #Ultimately masking the canny image to only show the region of interest traced by the polygonal contour of the mask
    #Arguments are edge detected image and the masked image
    masked_image = cv2.bitwise_and(image, mask) #Masks the region we don't want on the edge detected image
    return masked_image #Returns the edge detected image with only region of interest

#Read the image from the file and return it as a multidimensional
#Numpy array containign the relative intensities of each pixel in the test_image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image) #Ensure to make a copy, otherwise changes will be reflected in the original
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

#HoughLinesP() function takes the cropped edge detected image and uses Hough Transform to determine the line of best
#Fit along a series of points
#HoughLinesP(edge detected cropped image, resolution or size of rho(bins in y), resolution or size of theta(bins in x), minimum number of intersections needed to accept a line, min length of line segment that should be detected, min space between two line segments in order to be considered invidual lines)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #Returns an array of line segments with start and end points
# black image with line segments drawn
line_image = display_lines(lane_image, lines)
# addWeighted(original image, pixel intensity (darker), second black image with lines, pixel intensity of second image, some value that will add to each pixel ) fucntion takes the original image and adds the b
#takes each pixel intensity in both images, adds them together (intensity of second image is 0 therefore the coloured image will be displayed), display the coloured image darker and display the lines lighter
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#imshow() takes two arguments ('name of the image when displayed','which image to use' )
cv2.imshow("result", combo_image) #Here we display the grayscale image instead of the original
#Displays image for a specified amount of miliseconds
cv2.waitKey(0)
