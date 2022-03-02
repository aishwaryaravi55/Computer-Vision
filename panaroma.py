import numpy as np
import cv2
from opts import get_opts
# Import necessary functions

from matchPics import matchPics
from planarH import computeH_ransac,compositeH
import matplotlib.pyplot as plt


# Q4.2

# left=cv2.imread('.../data/pano_left.jpg')
# right=cv2.imread('.../data/pano_right.jpg')

left=cv2.imread('.../data/my_pano_left.jpeg')
right=cv2.imread('.../data/my_pano_right.jpg')

m1=left.shape[0]
n1=left.shape[1]
m2=right.shape[0]
n2=right.shape[1]

diff1=m1-n1

new_img=cv2.copyMakeBorder(right,0, diff1, (round(max(n1,n2))-n2), 0, cv2.BORDER_CONSTANT,0)

opts=get_opts()
matches,locs1, locs2=matchPics(left, new_img, opts)
bestH2to1,_ =computeH_ransac(locs1[matches[:,0], 0:2], locs2[matches[:,1], 0:2],opts)

final_pano=np.maximum(new_img,compositeH(bestH2to1, left, new_img))

plt.imshow(final_pano)
plt.show()