import os
import cv2
import numpy as np
import sys
from time import time
import logging 
import glob


class DataReaderIMG:

    def __init__(self, img_path='', file_path='', max_frame_id=-1):
        ''' Read img and metadata file (opt) and output the data sequence

        Input:
            - img file path
            - Optional: metadata file path (an npy file)
                    format: [{'frame_id': frame_id, xxx}]
                        xxx means you can put whatever k-v entry
        '''
        if not os.path.exists(img_path):
            print('Cannot find path: %s' % str(img_path))
            return

        self.data = []
        self.frame_id = 0
        self.max_fid = max_frame_id

        self.imgs = sorted(glob.glob(img_path+"/*"))
        # print(self.imgs)

        self.data_ptr = 0
        if file_path:
            if os.path.exists(file_path):
                self.data = np.load(open(file_path, 'rb'), allow_pickle=True)
            else:
                self.log('Cannot load metadata: {}'.format(file_path))

        self.log('init')
        print("img init success")

    def get_data(self):
        '''
        Return:
        - img
        - frame_id
        - meta as a list
        '''
        if self.frame_id >= len(self.imgs):
            return [], self.frame_id, "", []

        fid = self.frame_id
        frame = cv2.imread(self.imgs[fid])
        
        self.frame_id += 1
        return frame, fid, self.imgs[fid], []


    def log(self, s):
        logging.debug('[DataReader] %s' % s)
