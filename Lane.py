import numpy as np

class Lane:
    def __init__(self, img_size, convert_x, convert_y,
                 smooth_window=5, poly_fit_thres=10, default_lane_width=300):
        self.left_detected = False
        self.right_detected = False
        self.img_size = img_size
        self.width = default_lane_width # Lane width
        # y coodinates, will be used to plot polynomial
        self.ploty = np.linspace(0, img_size[0]-1, img_size[0]).astype('int64')
        self.left_x = np.array([], dtype=np.int64)
        self.left_y = np.array([], dtype=np.int64)
        self.right_x = np.array([], dtype=np.int64)
        self.right_y = np.array([], dtype=np.int64)

        # fitting data
        self.poly_fit_thres = poly_fit_thres # for error checking
        self.left_poly = [] # [c2,c1,c0] coeffecients of the left lane polynomial
        self.right_poly = [] # [c2,c1,c0] coeffecients of the right lane polynomial
        ## err of the poly fitting
        self.left_polyfit_err = 0 # err for left polynomial fitting.
        self.right_polyfit_err = 0 # err for right polynomial fitting.

        ## past_left_fitx and past_right_fitx store fitted lane curve in the past n frame
        ## while n=smooth_window is the smoothing window size
        self.past_left_fitx = np.array([[np.nan]*smooth_window]*img_size[0])
        self.past_right_fitx = np.array([[np.nan]*smooth_window]*img_size[0])
        ## left_fitx and right_fitx store the average result from past_left_fitx and past_right_fitx
        self.left_fitx = None
        self.right_fitx = None

        # constants used to convert from pixels to meters
        self.convert_x = convert_x
        self.convert_y = convert_y

        # canvas to draw the analyzed results
        self.canvas = np.zeros((self.img_size[0],self.img_size[1],3),dtype=np.uint8)

    def reset_canvas(self):
        self.canvas[:,:,:] = 0 # reset to blank
        self.left_x = np.array([], dtype=np.int64)
        self.left_y = np.array([], dtype=np.int64)
        self.right_x = np.array([], dtype=np.int64)
        self.right_y = np.array([], dtype=np.int64)
        self.left_poly = []
        self.right_poly = []

    def polyfit_left(self):
        # Shift past_left_fitx to the left so that the last column can be used to
        # store the new fitted x coordinates (of the left lane polynomial)
        self.past_left_fitx = np.roll(self.past_left_fitx,shift=(0,-1))
        self.past_left_fitx[:,-1] = np.nan
        self.left_detected = False

        if self.left_x.size==0: return False  # if no pixel to fit, return False
        # Fit a second order polynomial
        self.left_poly = np.polyfit(self.left_y, self.left_x, 2)

        # Error check 1:
        ## compute the difference between the x coordinates of the detected lane pixels
        ## and the fitted polynomial
        diff = np.abs((self.left_poly[0]*self.left_y**2 +
                      self.left_poly[1]*self.left_y +
                      self.left_poly[2]).astype('int64')-self.left_x)
        ## remove outliers: if a pixel is located more than the median of the difference
        ## max(.,10): if median is too small, then there is no outlier,
        outlier_mask = (diff<max(np.median(diff),10))
        ## filter out outliers
        self.left_y = self.left_y[outlier_mask]
        self.left_x = self.left_x[outlier_mask]
        diff = diff[outlier_mask]
        ## polyfit_err is the mean err after filtering outliers
        self.left_polyfit_err = diff.mean() if diff.size>0 else self.poly_fit_thres + 1
        if self.left_polyfit_err>self.poly_fit_thres:
            return False

        # compute the fitted curve
        left_fitx = (self.left_poly[0]*self.ploty**2 +
                     self.left_poly[1]*self.ploty +
                     self.left_poly[2]).astype('int64')

        # Error check 2: check left-right curve distance.
        if self.right_fitx is not None:
            lane_checker = self.right_fitx-left_fitx
            mean_width = lane_checker.mean()
            ## if left lane line intersects with right line
            ## or the average lane width is too narrow
            ## then fail the checker
            if (lane_checker<0).any():
                print("Lane intersects")
                return False
            elif mean_width<100:
                print("Lane width failed lane_checker")
                return False
            else:
                self.width = mean_width

        # trim the pixels that run out of the canvas
        left_fitx[left_fitx<0]=0
        left_fitx[left_fitx>self.img_size[0]-1]=self.img_size[0]-1
        self.past_left_fitx[:,-1] = left_fitx
        self.left_fitx = np.nanmean(self.past_left_fitx, axis=1).astype('int64')
        self.left_detected = True
        return True

    def polyfit_right(self):
        # similar to polyfit_left
        self.past_right_fitx = np.roll(self.past_right_fitx,shift=(0,-1))
        self.past_right_fitx[:,-1] = np.nan
        self.right_detected = False
        if self.right_x.size==0: return False
        # Fit a second order polynomial
        self.right_poly = np.polyfit(self.right_y, self.right_x, 2)
        diff = np.abs((self.right_poly[0]*self.right_y**2 +
                      self.right_poly[1]*self.right_y +
                      self.right_poly[2]).astype('int64')-self.right_x)
        outlier_mask = (diff<max(np.median(diff),10))
        self.right_y = self.right_y[outlier_mask]
        self.right_x = self.right_x[outlier_mask]
        diff = diff[outlier_mask]
        self.right_polyfit_err = diff.mean() if diff.size>0 else self.poly_fit_thres + 1

        if self.right_polyfit_err>self.poly_fit_thres:
            return False

        right_fitx = (self.right_poly[0]*self.ploty**2 +
                       self.right_poly[1]*self.ploty +
                       self.right_poly[2]).astype('int64')

        if self.left_fitx is not None:
            lane_checker = right_fitx-self.left_fitx
            mean_width = lane_checker.mean()
            if (lane_checker<0).any() or mean_width<100:
                return False
            else:
                self.width = mean_width

        right_fitx[right_fitx<0]=0
        right_fitx[right_fitx>self.img_size[0]-1]=self.img_size[0]-1
        self.past_right_fitx[:,-1] = right_fitx
        self.right_fitx = np.nanmean(self.past_right_fitx, axis=1).astype('int64')
        self.right_detected = True
        return True

    def analyze(self):
        # Return lane curvature in meter and drifting from lane center

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.left_y*self.convert_y, self.left_x*self.convert_x, 2)
        right_fit_cr = np.polyfit(self.right_y*self.convert_y, self.right_x*self.convert_x, 2)
        # Calculate the new radii of curvature
        y_eval = self.convert_y*self.img_size[0]
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        interection_left = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
        interection_right = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
        center_offset = (interection_left+interection_right)/2 - self.img_size[1]*self.convert_x/2

        return ((left_curverad+right_curverad)/2, center_offset)
