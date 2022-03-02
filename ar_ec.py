
# Import necessary functions

from matchPics import matchPics
from ar import form_ar_app
from planarH import computeH_ransac,compositeH
import matplotlib.pyplot as plt
import time

# Q4.1
# book_vid_src=loadVid('../data/book.mov')
# ar_vid_src=loadVid('../data/ar_source.mov')
# cv_cover=cv2.imread('../data/cv_cover.jpg')
# opts=get_opts()

# vid = cv2.VideoCapture('.../python/ar.avi')

# for frame_num in range(0,100):
#     print("Frame number:", frame_num)
    
#     frame=form_ar_app(opts, cv_cover, ar_vid_src[frame_num], book_vid_src[frame_num])
#     out.write(frame)

# cv2.destroyAllWindows()
# out.release()