import base64
import cv2
import numpy as np
from configs import *
import tensorflow as tf
from PIL import ImageTk, Image
import random
import copy

from datetime import timedelta
from time import perf_counter as tpc

def static_vars(**args):
    '''Decoration'''
    def decorate(fn):
        for var in args:
            setattr(fn, var, args[var])
        return fn
    return decorate

def base64_to_cvmat(base64_str):
    '''Changing display format'''
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR)

class TrainTimer:
    '''Training timer, not very accurate, might need changing'''
    def __init__(self, *increment):
        self.increments = list(increment)
        self.finished   = [0 for _ in increment]
        self.time_taken = [0.0 for _ in increment]

        self.current    = None
        self.timer  = 0.0

    def start_timer(self):
        self.timer = tpc()

    def increment(self, pos):
        time_taken = tpc() - self.timer
        self.timer = tpc()

        self.time_taken[pos] += time_taken
        self.finished[pos]   += 1

    def time_left(self, timeerror='?:??:??'):
        t = 0.0
        for n in range(len(self.increments)):
            taken = self.time_taken[n]
            left  = self.increments[n] - self.finished[n]
            try: t += taken/self.finished[n]*left
            except ZeroDivisionError: return timeerror
        return str(timedelta(seconds=round(t)))

    def total_time(self):
        return str(timedelta(seconds=round(sum(self.time_taken))))


def camera_angle_offset(x):
    '''Offsetting steering angle based on which camera is being used'''
    angle, camera, flip, speed = x['wheel_angle'], x['camera'], x['flip'], x['speed']
    if camera == 'center': return angle

    if camera == 'left' and flip:
        return angle+CAMERA_ANGLE_VARIATION

    if camera == 'left' and not flip:
        return angle-CAMERA_ANGLE_VARIATION

    if camera == 'right' and flip:
        return angle-CAMERA_ANGLE_VARIATION

    if camera == 'right' and not flip:
        return angle+CAMERA_ANGLE_VARIATION

def train_test_split(X, y):
    '''Train test splitting of X and y datasets'''
    test_samples = round(TEST_PERCENTAGE * len(y[0]))
    test_X, train_X = X[:test_samples], X[test_samples:]
    test_y, train_y = y[:, :test_samples].transpose(), y[:, test_samples:].transpose()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_X, train_y)).shuffle(train_y.shape[0], seed=SEED).batch(BATCH)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_X, test_y)).batch(BATCH)

    return train_ds, test_ds

def img2tk(img):
    '''Change numpy array image to ImageTk photoimage'''
    return ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

def pad(image:np.ndarray, pad_size=1, pad_value=255.0):
    '''Image padding (for display purpose)'''
    under_image = np.full([shape + pad_size*2 for shape in image.shape], fill_value=pad_value)
    under_image[pad_size:image.shape[0]+pad_size, pad_size:image.shape[1]+pad_size] = image
    return under_image

def row_show(images:np.ndarray, image_height=100, max_image=8, pad_value=255.0, layer_name=''):
    '''Showing layer images, probably needs some optimization'''
    for n in range(0, len(images), max_image):
        row_images = images[n:n+max_image]
        if len(range(0, len(images), max_image)) != 1:
            while len(row_images) < max_image:
                if len(images.shape) == 1:
                    row_images = np.concatenate((row_images, np.array([pad_value], dtype=np.float32)))
                else:
                    row_images = np.concatenate((row_images, np.full((1,) + images[0].shape, fill_value=pad_value, dtype=np.float32)), axis=0)
        for new_image in row_images:
            if len(images.shape) == 1:
                new_image = np.full((image_height, image_height), fill_value=new_image, dtype=np.float32)
            else:
                new_image = cv2.resize(new_image, (int(new_image.shape[1]/new_image.shape[0]*image_height), image_height), interpolation=0)
            _, new_image = cv2.threshold(new_image, 255.0, 0.0, cv2.THRESH_TRUNC)
            new_image = pad(new_image)
            try:
                image = np.concatenate((image, new_image), axis=0)
            except NameError: image = new_image
        try:
            output_image = np.concatenate((output_image, image), axis=1)
        except NameError: output_image = image
        del image
    output_image = pad(output_image)
    output_image = output_image.astype(np.uint8)
    return output_image

def train_result(epoch, epoch_len, batch_num, batch_len,
                 loss, steering_accuracy, throttle_accuracy,
                 lr, timer, step):
    '''Print train result'''
    epoch = int(epoch) + 1
    epoch_len = int(epoch_len)

    batch_num = int(batch_num) + 1
    batch_len = int(batch_len)
    batch_left = int(batch_len - batch_num)

    loss = float(loss)
    steering_accuracy = float(steering_accuracy)
    throttle_accuracy = float(throttle_accuracy)

    print(f'EPOCH: {epoch}/{epoch_len}\n'
          f'Step Number: {step}\n'
          f'Batch Number: {batch_num}/{batch_len}\n'
          f'[{loading_bar(batch_num, batch_left)}]\n'
          f'Estimated Time Left: {timer}\n'
          f'Learning Rate: {lr:.3}\n'
          f'Loss:         {loss:.3}\n'
          f'Steering Error: {round(steering_accuracy*25, 3)}deg\n'
          f'Throttle Error: {throttle_accuracy:.3}\n\n')


def test_result(epoch, epoch_len,
                loss, steering_accuracy, throttle_accuracy, timer):
    '''Print test result'''
    epoch = int(epoch) + 1
    epoch_len = int(epoch_len)

    loss = float(loss)
    steering_accuracy = float(steering_accuracy)
    throttle_accuracy = float(throttle_accuracy)

    print('TEST RESULT\n'
          +'='*64+'\n'
          f'EPOCH: {epoch}/{epoch_len}\n'
          f'Estimated Time Left: {timer}\n'
          f'Loss:  {loss:.3}\n'
          f'Steering Error: {round(steering_accuracy*25, 3)}deg\n'
          f'Throttle Error: {throttle_accuracy:.3}\n'
          +'='*64+'\n\n')

def loading_bar(done, left, done_i='=', left_i='-', fold=64, insert=']\n['):
    '''Create text based loading bar'''
    s = done_i*done + left_i*left
    return (''.join(l + insert * (n % fold == fold-1) for n, l in enumerate(s)))
