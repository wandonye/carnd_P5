import cv2
import numpy as np
from Lane import Lane

class LaneFinder:
    def __init__(self, original_image_size, mask_vertices, anchor_points, tranformed_image_size,
                 cali_mtx, cali_dist, convert_x, convert_y,
                 window_width=50, window_height=80, margin=100,
                 left_lane_bound=None, right_lane_bound=None,
                 left_lane_pixel_thres = 600, right_lane_pixel_thres=200,
                 smooth_window=5):
        self.original_image_size = original_image_size
        self.lane = Lane(tranformed_image_size, convert_x, convert_y, smooth_window)
        self.height = tranformed_image_size[0]
        self.width = tranformed_image_size[1]
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.level_num = (int)(tranformed_image_size[0]/window_height)
        self.window = np.ones(window_width) # Define window template
        self.channels = []
        [self.idx,self.idy] = np.meshgrid(range(tranformed_image_size[1]),
                                          range(tranformed_image_size[0]))
        self.calibration_matrix = cali_mtx
        self.calibration_dist = cali_dist

        src = np.array(anchor_points, dtype = "float32")
        h_margin = 200
        top_margin = 200
        bottom_margin = 50
        dst = np.array([[h_margin, top_margin],
                        [tranformed_image_size[0] - h_margin, top_margin],
                        [tranformed_image_size[0] - h_margin, tranformed_image_size[1] - bottom_margin],
                        [h_margin, tranformed_image_size[1] - bottom_margin]], dtype = "float32")
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = np.linalg.inv(self.M)

        roi = np.zeros(original_image_size,dtype=np.uint8)
        cv2.fillPoly(roi, np.array([mask_vertices], dtype=np.int32),1)
        roi = cv2.undistort(roi, cali_mtx, cali_dist, None, cali_mtx)
        self.mask = cv2.warpPerspective(roi, self.M, tranformed_image_size, flags=cv2.INTER_LINEAR)

        if left_lane_bound is None:
            left_lane_bound = (int(self.width/8),int(self.width*3/8),int(self.height*3/4), self.height)
        if right_lane_bound is None:
            right_lane_bound = (int(self.width*5/8),int(self.width*7/8),int(self.height*1/8), self.height)

        self.set_lane_initial_detect_range(left_lane_bound, right_lane_bound)

        self.left_lane_pixel_thres = left_lane_pixel_thres
        self.right_lane_pixel_thres = right_lane_pixel_thres
        self.texts = [] # store center offset and curvature, as well as other info for debuging

    def set_lane_initial_detect_range(self, left_lane_bound, right_lane_bound):
        self.lane_bound = [left_lane_bound, right_lane_bound]

    def window_mask(self, center, level):
        output = np.zeros((self.height,self.width))
        output[int(self.height-(level+1)*self.window_height):int(self.height-level*self.window_height),
               max(0,int(center-self.window_width/2)):min(int(center+self.window_width/2),self.width)] = 1
        return output

    def initial_window_finder(self, image, threshold,
                              left_bound, right_bound,
                              upper_bound, lower_bound):
        # Sum the bottom of the image with the given height to get slice
        v_sum = np.sum(image[upper_bound:lower_bound,left_bound:right_bound], axis=0)
        # Find the starting position for a lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template
        conv = np.convolve(self.window,v_sum)
        # the peak should have enough signals, and should stand out from the rests (not noise).
        if conv.max()>threshold:
            center = int(np.argmax(conv)-self.window.size/2 + left_bound)
            return center
        else:
            self.texts.append("No enough pixels detected")

        return -1

    def channel_decompose(self, img, saturation_white_thresh=(0, 2),
                         saturation_yellow_thresh=(100, 180),
                         hue_thresh=(18, 25), value_thresh=(200, 255),
                         component_limit=6, min_area=1000,
                         ksize=15):

        # Convert to HLS color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel>=value_thresh[0])&(v_channel<=value_thresh[1])] = 1
        self.thresholds = [50]

        return [v_binary]

    def prepare_channels(self, img):
        channels = self.channel_decompose(img)
        self.channels = [im*self.mask for im in channels] #ignore outside of masked region
        return len(channels)

    def find_window_per_level(self, image, center, level, threshold):
        # Find the best centroid by using past center as a reference
        min_index = max(center-self.margin, 0)
        max_index = min(center+self.margin, self.width)
        v_projection = np.sum(image[int(self.height-(level+1)*self.window_height):
                                    int(self.height-level*self.window_height),
                                    min_index:max_index],
                              axis=0)
        # convolve the window into the vertical slice of the image
        conv_signal = np.convolve(self.window, v_projection)
        peak_idx = np.argmax(conv_signal)
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        center = peak_idx+min_index-int(self.window_width/2)
        return center

    def get_points_in_window(self, image, level, window_center):
        # return non-zero points for the given window on the given level
        points = np.zeros_like(image)
        if window_center>=0:
            window_mask = self.window_mask(window_center,level)
            points = image*window_mask
        return points

    def init_lane_finder(self, side):
        idx = -1
        center = -1
        # search window center starting from the first channel
        # for the current channel, locate the bin with most pixels within the given bound
        # bounds are different for left lane and right, and are defined in the initializer.
        while center<0 and idx<len(self.channels)-1:
            idx += 1 # move to the next channel if cannot find a window with peak in this channel
            center = self.initial_window_finder(self.channels[idx],self.thresholds[idx],
                                                self.lane_bound[side][0], self.lane_bound[side][1],
                                                self.lane_bound[side][2], self.lane_bound[side][3])

        self.window_centroids[side].append((idx, center))
        points = np.zeros((self.height, self.width))
        if center>=0:
            # when a window with peak is found
            # slice the birdview image into several layers and search for peak in each layer.
            for level in range(1,self.level_num):
                center = self.find_window_per_level(self.channels[idx], center,
                                                    level, self.thresholds[idx])
                level_points = self.get_points_in_window(self.channels[idx], level, center)
                points = np.maximum(points,level_points)
                self.window_centroids[side].append((idx, center))
        if side==0:
            (self.lane.left_y, self.lane.left_x) = np.where(points>0)
            self.lane.polyfit_left()
        else:
            (self.lane.right_y, self.lane.right_x) = np.where(points>0)
            self.lane.polyfit_right()

    def tube_lane_finder(self, img, side):
        tube_mask = np.zeros_like(img)
        if side==0: # left lane
            search_radius = 10
            while (search_radius<self.margin and
                   self.lane.left_x.size<self.left_lane_pixel_thres):
                tube_mask[np.abs(self.idx-np.repeat([self.lane.left_fitx],
                                           self.width,axis=0).T)<search_radius]=1
                l_points = img*tube_mask
                (self.lane.left_y, self.lane.left_x) = np.where(l_points>0)
                search_radius += 10

            return self.lane.polyfit_left()

        else:  # right lane
            search_radius = 10
            while (search_radius<self.margin and
                   self.lane.right_x.size<self.right_lane_pixel_thres):
                tube_mask[np.abs(self.idx-np.repeat([self.lane.right_fitx],
                                               self.width,axis=0).T)<search_radius]=1
                r_points = img*tube_mask
                (self.lane.right_y, self.lane.right_x) = np.where(r_points>0)
                search_radius += 10

            return self.lane.polyfit_right()

    def draw_lane(self):
        # Draw the lane pixels and lane curve
        left_bound_idx, right_bound_idx = None, None
        if self.lane.left_x.size>0:
            # Make left lane pixels red
            self.lane.canvas[self.lane.left_y,self.lane.left_x,2] = 255
        if self.lane.left_detected:
            # Draw polynomial
            self.lane.canvas[self.lane.ploty,self.lane.left_fitx,0:2] = 255
            left_bound_idx = np.repeat([self.lane.left_fitx],
                                        self.width,axis=0).T
        if self.lane.right_x.size>0:
            #right lane pixels green,
            self.lane.canvas[self.lane.right_y,self.lane.right_x,0] = 255
        if self.lane.right_detected:
            self.lane.canvas[self.lane.ploty,self.lane.right_fitx,0] = 255
            self.lane.canvas[self.lane.ploty,self.lane.right_fitx,2] = 255
            right_bound_idx = np.repeat([self.lane.right_fitx], self.width,axis=0).T
        elif left_bound_idx is not None:
            right_bound_idx = left_bound_idx + self.lane.width

        if (left_bound_idx is not None) and (right_bound_idx is not None):
            self.lane.canvas[(self.idx-left_bound_idx>0)&(self.idx-right_bound_idx<0),1] = 50
        return self.lane.canvas

    def draw_result(self, original_img):
        new_img = np.copy(original_img)
        font = cv2.FONT_HERSHEY_DUPLEX
        text_v_position = 70
        for text in self.texts:
            cv2.putText(new_img, text, (40, text_v_position),
                        font, 1.0, (200,255,155), 2, cv2.LINE_AA)
            text_v_position += 30

        return new_img

    def pipeline(self, img):
        # reset information for the current frame
        self.lane.reset_canvas()
        self.texts = []
        self.window_centroids = [[],[]]

        # undistort and apply perspective transformation to get birdview
        warped_img = cv2.undistort(img, self.calibration_matrix, self.calibration_dist,
                                   None, self.calibration_matrix)
        warped_img = cv2.warpPerspective(warped_img, self.M,
                                         (self.height,self.width),
                                         flags=cv2.INTER_LINEAR)

        # apply different filters to create candidate channels
        L = self.prepare_channels(warped_img)

        text = ''
        # If lane-line is detected from the previous frame,
        # then use tube_lane_finder
        if self.lane.left_detected and self.lane.left_fitx is not None:
            channel_left_idx = 0
            while channel_left_idx<L and (not self.tube_lane_finder(self.channels[channel_left_idx],0)):
                channel_left_idx += 1
            text += 'L channel:' + str(channel_left_idx)+' pix:'+ str(self.lane.left_x.size) + ', '
        else: # otherwise search from scratch
            text += 'L init '
            self.init_lane_finder(side=0)

        # same process applied to right lane-line
        if self.lane.right_detected and self.lane.right_fitx is not None:
            channel_right_idx = 0
            while channel_right_idx<L and (not self.tube_lane_finder(self.channels[channel_right_idx],1)):
                channel_right_idx += 1
            text += 'R channel:' + str(channel_right_idx)+' pix:'+ str(self.lane.right_x.size)
        else:
            text += 'R init '
            self.init_lane_finder(side=1)

        if len(text)>0: self.texts.append(text)

        if (self.lane.left_x.size>0) and (self.lane.right_x.size>0):
            # compute curvature and center offset
            (curvature, center_offset) = self.lane.analyze()
            self.texts.append("Curvature: "+"{:04.1f}".format(curvature) + 'm')
            self.texts.append("Distance from Center: "+"{:04.3f}".format(center_offset)+ 'm')

        overlay = cv2.warpPerspective(self.draw_lane(), self.M_inv,
                                      (self.original_image_size[1],self.original_image_size[0]),
                                      flags=cv2.INTER_LINEAR)
        return self.draw_result(cv2.addWeighted(img, 1, overlay, 1, 0.0))
