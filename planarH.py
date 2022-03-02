import numpy as np
import cv2
import math


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    #assert(x1.shape[0] == x2.shape[0])
   # assert(x1.shape[0]==2)
    m=x1.shape[0]
    
    A=[]
    
    for i in range (m):
        
        xs=x1[i,0]
        ys=x1[i,1]
        xd=x2[i,0]
        yd=x2[i,1]
        
        A.append([-xs, -ys, -1, 0,0,0, xd*xs, xd*ys, xd])
        A.append([0, 0, 0, -xs,-ys,-1, yd*xs, yd*ys, yd])
        
    A=np.asarray(A)
    
    U,S,V=np.linalg.svd(A)
    
    H2to1=(V[-1, :]/ V[-1, -1]).reshape(3,3)

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    
    x1_x_mean=np.mean(x1[:, 0])
    x1_y_mean=np.mean(x1[:, 1])
    x2_x_mean=np.mean(x2[:, 0])
    x2_y_mean=np.mean(x2[:, 1])

    # Shift the origin of the points to the centroid
    
    m1,n1=x1.shape
    m2,n2=x1.shape
    shifted_points_x1=[]
    shifted_points_x2=[]
    
    for i in range(m1):
        
        xdiff=x1[i,0]-x1_x_mean
        xdiff_sq=pow(xdiff,2)
        
        ydiff=x1[i,1]-x1_y_mean
        ydiff_sq=pow(ydiff,2)
        
        shifted_points_x1.append(np.sqrt(xdiff_sq+ydiff_sq))
    
    for i in range(m2):
        
        xdiff=x2[i,0]-x2_x_mean
        xdiff_sq=pow(xdiff,2)
        
        ydiff=x2[i,1]-x2_y_mean
        ydiff_sq=pow(ydiff,2)
        
        shifted_points_x2.append(np.sqrt(xdiff_sq+ydiff_sq))
        
    shifted_points_x1=np.asarray(shifted_points_x1)
    shifted_points_x2=np.asarray(shifted_points_x2)

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    shifted_norm_x1=np.sqrt(2)/np.max(shifted_points_x1)
    shifted_norm_x2=np.sqrt(2)/np.max(shifted_points_x2)
    
    # Similarity transform 1
    
    trans_mat_S1=np.eye(3)
    trans_mat_S2=np.eye(3)
    trans_mat_x1=np.eye(3)
    trans_mat_x2=np.eye(3)
    
    trans_mat_x1[0,2]=-x1_x_mean
    trans_mat_x1[1,2]=-x1_y_mean
    trans_mat_x2[0,2]=-x2_x_mean
    trans_mat_x2[1,2]=-x2_y_mean
    
    for i in range(2):
        trans_mat_S1[i,i]=shifted_norm_x1
        trans_mat_S2[i,i]=shifted_norm_x2
        
    
    T1=np.dot(trans_mat_S1,trans_mat_x1)
    T2=np.dot(trans_mat_S2,trans_mat_x2)
    
    # Similarity transform 2
    
    a1=np.ones((m1, 1))
    a_x1 = np.hstack((x1,a1))
    H_x1=T1@np.transpose(a_x1)
    
    a2=np.ones((m2, 1))
    a_x2 = np.hstack((x2,a2))
    H_x2=T2@np.transpose(a_x2)
    
    # Compute homography
    
    check=computeH(H_x1, H_x2)

    # Denormalization
    
    T2_inv=np.linalg.inv(T2)
    
    H2to1=np.dot(np.dot(T2_inv,check),T1)
    
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol
    
    m1=locs1.shape[0]
    m2=locs2.shape[0]
    
    a1=np.ones((m1, 1))
    H_x1 = np.hstack((locs1,a1))
    
    a2=np.ones((m2, 1))
    H_x2 = np.hstack((locs2,a2))
    inlier_count=0
    maxx=-1
    
    for i in range (0,max_iters):
        
        random_idx=np.random.choice(m1,4)
        
        rand1=locs1[random_idx, :]
        rand2=locs2[random_idx, :]
        
        H_rand=computeH_norm(rand1, rand2)
    
        norm_shape=H_x2.shape[0]
        
        for i in range(0, norm_shape):
            
            keep=np.transpose(H_x1[i])
            keep_calc=np.dot(H_rand, keep)
            
            e_rand=np.linalg.norm([(H_x2[i][0]-(keep_calc[0]/keep_calc[2])),(H_x2[i][1]-(keep_calc[1]/keep_calc[2]))])
            
            if e_rand<=inlier_tol:
                inlier_count=inlier_count+1
                
        if inlier_count > maxx:
            bestH2to1=H_rand
            maxx=inlier_count

    return bestH2to1, maxx


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography
    

    # Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    mask=np.ones(template.shape)

    # Warp mask by appropriate homography
    m=img.shape[0]
    n=img.shape[1]
    
    mask_warp=cv2.transpose(cv2.warpPerspective(cv2.transpose(mask), H2to1, (m,n)))

    # Warp template by appropriate homography
    
    temp_warp=cv2.transpose(cv2.warpPerspective(cv2.transpose(template), H2to1, (m,n)))

    # Use mask to combine the warped template and the image
    
    img[np.nonzero(mask_warp)]=temp_warp[np.nonzero(mask_warp)]
    composite_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return composite_img
