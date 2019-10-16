# Shivam Chourey
# shivam.chourey@gmail.com

import cv2
import numpy as np

# 2D Kernel - Size 11*11
GaussianKernel = [
[0.000002   , 0.00001   , 0.000047  , 0.000136  , 0.000259  , 0.00032   , 0.000259  , 0.000136  , 0.000047  , 0.00001   , 0.000002 ],
[0.00001    , 0.000072  , 0.000322  , 0.000939  , 0.001785  , 0.002212  , 0.001785  , 0.000939  , 0.000322  , 0.000072  , 0.00001  ],
[0.000047   , 0.000322  , 0.001443  , 0.004212  , 0.008008  , 0.009921  , 0.008008  , 0.004212  , 0.001443  , 0.000322  , 0.000047 ],
[0.000136   , 0.000939  , 0.004212  , 0.012297  , 0.02338   , 0.028963  , 0.02338   , 0.012297  , 0.004212  , 0.000939  , 0.000136 ],
[0.000259   , 0.001785  , 0.008008  , 0.02338   , 0.044453  , 0.055067  , 0.044453  , 0.02338   , 0.008008  , 0.001785  , 0.000259 ],
[0.00032    , 0.002212  , 0.009921  , 0.028963  , 0.055067  , 0.068216  , 0.055067  , 0.028963  , 0.009921  , 0.002212  , 0.00032  ],
[0.000259   , 0.001785  , 0.008008  , 0.02338   , 0.044453  , 0.055067  , 0.044453  , 0.02338   , 0.008008  , 0.001785  , 0.000259 ],
[0.000136   , 0.000939  , 0.004212  , 0.012297  , 0.02338   , 0.028963  , 0.02338   , 0.012297  , 0.004212  , 0.000939  , 0.000136 ],
[0.000047   , 0.000322  , 0.001443  , 0.004212  , 0.008008  , 0.009921  , 0.008008  , 0.004212  , 0.001443  , 0.000322  , 0.000047 ],
[0.00001    , 0.000072  , 0.000322  , 0.000939  , 0.001785  , 0.002212  , 0.001785  , 0.000939  , 0.000322  , 0.000072  , 0.00001  ],
[0.000002   , 0.00001   , 0.000047  , 0.000136  , 0.000259  , 0.00032   , 0.000259  , 0.000136  , 0.000047  , 0.00001   , 0.000002 ]  ]

# Function to apply 2D Kernel
def GaussianOnPixel(Image, index, itr):
    Intensity = 0;
    w,h = Image.shape[:2];
    
    if(index < 6 or index > w-6 or itr < 6 or itr > h-6):
        Intensity = Image[index][itr]
        # print("Got a free-ride")
        
    else:
        counter = 1
        #print("counter reset")
        for i in range(11):
          for j in range(11):
            Intensity += ( GaussianKernel[i][j] * Image[index-5+i][itr-5+j])
            # print("Internal counter ", counter)
            counter += 1
       
    return Intensity;

	
# 1D kernel - Size 11
GaussianFilter1D = [0.001227,   0.008468,   0.037984,   0.110892,   0.210838,   0.261182,   0.210838,   0.110892,   0.037984,   0.008468,   0.001227];

# Function to apply Vertical Kernel
def VerticalGaussianOnPixel(Image, index, itr):
    Intensity = 0;
    w,h = Image.shape[:2];
    
    if(index < 6 or index > w-6):
        Intensity = Image[index][itr]
        # print("Got a free-ride")
        
    else:
        counter = 1
        #print("counter reset")
        for i in range(11):
           Intensity += ( GaussianFilter1D[i] * Image[index-5+i][itr])
           # print("Internal counter ", counter)
           counter += 1
       
    return Intensity;

# Function to apply Horizontal Kernel	
def HorizontalGaussianOnPixel(Image, index, itr):
    Intensity = 0;
    w,h = Image.shape[:2];
    
    if(itr < 6 or itr > h-6):
        Intensity = Image[index][itr]
        # print("Got a free-ride")
        
    else:
        counter = 1
        #print("counter reset")

        for j in range(11):
           Intensity += ( GaussianFilter1D[j] * Image[index][itr-5+j])
           # print("Internal counter ", counter)
           counter += 1
       
    return Intensity;
    

# File input    
OrigName = "ENTER FILENAME HERE"
Original = cv2.imread(OrigName, cv2.IMREAD_GRAYSCALE)
print("Original Image loaded")

# Get shape of the input image
w,h = Original.shape[:2]

# 2D Gaussian Blur
TwoDResult = np.zeros((w,h));

print("Starting 2D Gaussian")
#count = 1
for index in range(w):
    for itr in range(h):
        # print("External counter", count)
        TwoDResult[index][itr] = GaussianOnPixel(Original, index, itr)
        
# Required for correct display using imshow function
TwoDResult = TwoDResult.astype(np.uint8)		

TwoDName = OrigName[:-4]+"_Gaussian_Blurred_2D.jpg"
cv2.imwrite(TwoDName, TwoDResult)

print("Done with 2D Gaussian")

#1D Gaussian Blur
OneDResultTemp = np.zeros((w,h));
OneDResult = np.zeros((w,h))
print("Starting 1D Gaussian")
# Step 1: Get the vertical filtering done
print("Starting Vertical filtering")
for index in range(w):
    for itr in range(h):
        OneDResultTemp[index][itr] = VerticalGaussianOnPixel(Original, index, itr)
        

# Step 2: Horizontal filtering
print("Starting Horizontal filtering")
for index in range(w):
    for itr in range(h):
        OneDResult[index][itr] = HorizontalGaussianOnPixel(OneDResultTemp, index, itr)
    
print("Done with 1D Gaussian")
# Required for correct display using imshow function
OneDResult = OneDResult.astype(np.uint8)

OneDName = OrigName[:-4]+"_Gaussian_Blurred_1D.jpg"
cv2.imwrite(OneDName, OneDResult)

print("Starting difference")
DiffImage = np.zeros((w,h))

for index in range(w):
   for itr in range(h):
       DiffImage[index][itr] = OneDResult[index][itr] - TwoDResult[index][itr]


# Print the metrics 
print("\n Difference Image Metrics: ")
mean = np.mean(DiffImage)
print("Mean: ", mean)	
   
variance = np.var(DiffImage)
print("Variance: ", variance)

median = np.median(DiffImage)
print("Median: ", median)

DiffName = OrigName[:-4]+"_Difference.jpg";
cv2.imwrite(DiffName, DiffImage)
print("Process complete")

cv2.imshow('Original', Original)
cv2.imshow('2D Result', TwoDResult)
cv2.imshow('1D Result', OneDResult)
cv2.imshow('Difference', DiffImage)       
cv2.waitKey(0)

cv2.destroyAllWindows()
