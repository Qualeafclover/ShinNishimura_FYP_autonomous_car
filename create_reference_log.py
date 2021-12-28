import cv2
import pandas as pd
from configs import *

from time import perf_counter as tpc

taken, left = 0.0, float('inf')
index_num = 0
total_index = float('inf')
finished = False

def create_reference(RAW_LOG_DIR, DATA_DIR):
    global finished
    finished = False
    df = pd.read_csv(RAW_LOG_DIR)
    df.columns = ['center_dir', 'left_dir', 'right_dir',
                  'wheel_angle', 'forward_throttle', 'backward_throttle', 'speed']
    df['acceleration'] = df['forward_throttle'] - df['backward_throttle']
    del df['forward_throttle'], df['backward_throttle']
    min_steering, max_steering = df['wheel_angle'].quantile(0.01), df['wheel_angle'].quantile(0.99)
    steepest = max(abs(min_steering), abs(max_steering))
    steering_mult = 1/steepest

    df = df.drop(df[df.speed == 0.0].index)
    new_df = pd.DataFrame.from_dict({'center_dir': [],
                                     'wheel_angle': [],
                                     'acceleration': [],
                                     'camera': [],
                                     'flip': [],
                                     'speed': []
                                     })
    process_angle_differences = pd.DataFrame.from_dict({
        'original': [],
        'new': []
    })

    global total_index
    total_index = len(df.index)

    t = tpc()
    for index, row in df.iterrows():
        global index_num, taken
        index_num = index
        taken = tpc() - t

        # print(f'Itering over row: {index}/{len(df.index)}')
        o_center = row['center_dir']
        o_left   = row['left_dir']#[1:]
        o_right  = row['right_dir']#[1:]

        center, center_flip = o_center[:-4]+'_0'+o_center[-4:], o_center[:-4]+'_1'+o_center[-4:]
        left,   left_flip   = o_left[:-4]  +'_0'+o_left[-4:],   o_left[:-4]  +'_1'+o_left[-4:]
        right,  right_flip  = o_right[:-4] +'_0'+o_right[-4:],  o_right[:-4] +'_1'+o_right[-4:]

        for o_img, img, img_flip in (
                (o_center, center, center_flip),
                (o_left,   left,   left_flip),
                (o_right,  right,  right_flip),
        ):
            o_img = cv2.imread(o_img)
            flip = cv2.flip(o_img, 1)
            cv2.imwrite(img, o_img)
            cv2.imwrite(img_flip, flip)

        agp = ANGLE_REDISTRIBUTION
        agm = ANGLE_REDISTRIBUTION
        center_angles = df['wheel_angle'][((index-agm) if index > agm else 0): ((index+agp) if index < (len(df.index)-agp) else -1)]
        actual_angle = row['wheel_angle']
        max_angle = center_angles.max()
        avg_angle = center_angles.mean()
        min_angle = center_angles.min()

        if avg_angle >= 0:
            if actual_angle == max_angle: center_angle = (max_angle*0.95+avg_angle*0.05)
            else: center_angle = (max_angle*0.0+actual_angle*0.9+avg_angle*0.1+min_angle*0.0)
        if avg_angle <= 0:
            if actual_angle == min_angle: center_angle = (min_angle*0.95+avg_angle*0.05)
            else: center_angle = (min_angle*0.0+actual_angle*0.9+avg_angle*0.1+max_angle*0.0)

        process_angle_differences.loc[len(process_angle_differences.index)+1] = [actual_angle, center_angle]

        left_angle   = center_angle
        right_angle  = center_angle

        center_angle_flip = -center_angle
        left_angle_flip   = -left_angle
        right_angle_flip  = -right_angle

        acceleration = row['acceleration']

        for imdir, angle, acc, camera, flip, speed in (
                (center,      center_angle,      acceleration, 'center', False, row['speed']),
                (center_flip, center_angle_flip, acceleration, 'center', True,  row['speed']),
                (left,        left_angle,        acceleration, 'left',   False, row['speed']),
                (left_flip,   left_angle_flip,   acceleration, 'left',   True,  row['speed']),
                (right,       right_angle,       acceleration, 'right',  False, row['speed']),
                (right_flip,  right_angle_flip,  acceleration, 'right',  True,  row['speed']),
        ):
            angle *= steering_mult
            if angle >  1.0: angle =  1.0
            if angle < -1.0: angle = -1.0
            new_df.loc[len(new_df.index)+1] = [imdir, angle, acc, camera, flip, speed]

    new_df.dropna()
    new_df.to_csv(DATA_DIR)
    finished = True