import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt

# Q2.1.6


def rotTest(opts):
    

    # Read the image and convert to grayscale, if necessary
    
    img= cv2.imread(('../data/cv_cover.jpg'))
    matchlen_arr=[]
    
    for i in range(36):
        
        # Rotate Image
        
        img_rot=scipy.ndimage.rotate(img, i*10)
    
        # Compute features, descriptors and Match features
        
        matches, locs1, locs2 = matchPics(img, img_rot, opts)
        
        # Update histogram
        
        matchlen_arr.append(len(matches))

    #Display histogram
    
    plt.bar(np.arange(36), matchlen_arr)
    plt.xlabel("Rotation angles divided by ten")
    plt.ylabel("Length of matches")
    plt.title("Number of Matches vs rotation angles")
    plt.show


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
