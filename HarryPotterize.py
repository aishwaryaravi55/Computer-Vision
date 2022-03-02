import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions

from matchPics import matchPics
from planarH import computeH_ransac,compositeH
import matplotlib.pyplot as plt

# Q2.2.4

def warpImage(opts):

    cv_desk = cv2.imread('../data/cv_desk.png')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    
    m1=(cv_desk.shape[0])
    n1=(cv_desk.shape[1])
    m2=(cv_cover.shape[0])
    n2=(cv_cover.shape[1])
    
    matches,locs1, locs2=matchPics(cv_cover, cv_desk, opts)
    bestH2to1,_ =computeH_ransac(locs1[matches[:,0], 0:2], locs2[matches[:,1], 0:2],opts)
    
    #new1=cv2.warpPerspective(cv2.transpose(hp_cover), bestH2to1, (m,n))
    new1=compositeH(bestH2to1, cv2.resize(hp_cover,(m2,n2)), cv_desk)
    
    return new1



if __name__ == "__main__":

    opts = get_opts()
    img=warpImage(opts)
    #new_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


