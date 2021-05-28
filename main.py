import tensorflow as tf

import tkinter as tk
import tkinter.ttk as ttk

import shutil
import time
import numpy as np
import pandas as pd
import threading
import os
import win32gui
import signal
import psutil
import cv2
import ctypes

from model import *
from drive import Simulator, PIController
from utils import *
from tkinter import filedialog
from create_reference_log import create_reference
from configs import *
from datetime import timedelta

# Threading the simulator so that it can be debugged easily
class Application(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        os._exit(1)
        self.root.quit()

    def run(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.root.title('Self-Driving-AI Interface')
        self.root.geometry('900x450')
        self.root.protocol('WM_DELETE_WINDOW', self.callback)
        self.root.wm_attributes('-transparentcolor', '')
        self.root.wm_attributes('-topmost', True)
        self.root.attributes('-alpha', 0.8)

        self.root = SelfDriving(master=self.root)
        self.root.mainloop()


class SelfDriving(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.root_w, self.root_h = 900, 450
        self.root = tk.Canvas(self.master, width=self.root_w, height=self.root_h, background='grey',
                              borderwidth=0, highlightthickness=0)

        # Start connection to the udacity simulator, whether it is open or not.
        sim = Simulator(self.controller, show=False)
        sim.init()
        run = threading.Thread(target=sim.run)
        run.start()

        # Which layers to look out for when displaying individual layers.
        self.look_for = {
            'process': {'image_height': 100, 'max_image': 3},

            'conv1a': {'image_height': 40, 'max_image': 8},
            'conv1b': {'image_height': 40, 'max_image': 8},
            'pool1c': {'image_height': 40, 'max_image': 8},

            'conv2a': {'image_height': 35, 'max_image': 8},
            'conv2b': {'image_height': 35, 'max_image': 8},
            'conv2c': {'image_height': 20, 'max_image': 10},

            # 'flatten': {'image_height': 10, 'max_image': 24},
            'dense1': {'image_height': 25, 'max_image': 8},
            'dense2': {'image_height': 75, 'max_image': 4},
            'dense3': {'image_height': 150, 'max_image': 1},
        }

        # Some other initialization stuff
        self.layer_reference = (None,) + tuple(layer for layer in self.look_for)
        self.layer_shown = 0
        self.file_dir = ''
        self.found = False
        self.running_tensorflow = False
        self.model_running = False
        self.main_menu()

    def main_menu(self):
        self.del_all()

        if self.found:
            self.button = tk.Button(
                self.root.pack(),
                text='End Simulation',
                width=30,
                command=self.kill_sim
            ).place(relx=0.2, rely=0.1, anchor=tk.CENTER)
        else:
            self.button = tk.Button(
                self.root.pack(),
                text='Start Simulation',
                width=30,
                command=self.start_sim
            ).place(relx=0.2, rely=0.1, anchor=tk.CENTER)

        self.button = tk.Button(
            self.root.pack(),
            text='Select work file',
            width=30,
            command=self.get_workfile
        ).place(relx=0.2, rely=0.25, anchor=tk.CENTER)

        self.button = tk.Button(
            self.root.pack(),
            text='Reset work file',
            width=30,
            command=self.del_workfile
        ).place(relx=0.2, rely=0.4, anchor=tk.CENTER)

        self.button = tk.Button(
            self.root.pack(),
            text='Load file for training',
            width=30,
            command=self.train_1
        ).place(relx=0.2, rely=0.7, anchor=tk.CENTER)

        if self.model_running:
            self.button = tk.Button(
                self.root.pack(),
                text='Stop model',
                width=30,
                command=self.stop_model
            ).place(relx=0.2, rely=0.8, anchor=tk.CENTER)

            self.button = tk.Button(
                self.root.pack(),
                text='Previous layer',
                width=20,
                command=self.layer_prev
            ).place(relx=0.55, rely=0.90, anchor=tk.CENTER)

            self.button = tk.Button(
                self.root.pack(),
                text='Next layer',
                width=20,
                command=self.layer_next
            ).place(relx=0.85, rely=0.90, anchor=tk.CENTER)
        else:
            self.button = tk.Button(
                self.root.pack(),
                text='Run trained model',
                width=30,
                command=self.run_model
            ).place(relx=0.2, rely=0.85, anchor=tk.CENTER)

    def layer_next(self):
        if self.layer_shown != (len(self.layer_reference)-1):
            self.layer_shown += 1

    def layer_prev(self):
        if self.layer_shown != 0:
            self.layer_shown -= 1

    def place_img(self):
        if self.model_running:
            try:
                self.imgtk = img2tk(self.img)
                self.master.place_slaves()[1].image = self.imgtk
                self.label = tk.Label(
                    self.root.pack(),
                    image=self.imgtk,
                    borderwidth=0
                ).place(relx=0.7, rely=0.45, anchor=tk.CENTER)
                self.del_last(spl=tk.Label, last_num=3)
                self.del_last(spl=tk.Label, last_num=4)

            except AttributeError: pass
            self.after(50, self.place_img)

    def place_acceleration(self):
        self.scale = tk.Scale(
            self.root.pack(),
            orient=tk.VERTICAL,
            length=150,
            from_=30.0, to=0.0,
            resolution=0.1,
            variable=self.speed
        ).place(relx=0.4, rely=0.5, anchor=tk.CENTER)

    def del_workfile(self):
        if self.file_dir != '':
            image_dir = os.path.join(self.file_dir, 'IMG')
            model_dir = os.path.join(self.file_dir, MODEL_SAVE)
            raw_log_dir = os.path.join(self.file_dir, RAW_LOG_NAME)
            reference_dir = os.path.join(self.file_dir, REFERENCE_NAME)
            shutil.rmtree(image_dir)
            shutil.rmtree(model_dir)
            os.remove(raw_log_dir)
            os.remove(reference_dir)

    def get_workfile(self):
        self.file_dir = filedialog.askdirectory(title='Select file location')

    # Stops the running model.
    def stop_model(self):
        self.model_running = False
        self.running_tensorflow = False
        self.main_menu()

    # Runs the model, prediction not done here though. It is on the threaded part.
    def run_model(self):
        self.speed = tk.DoubleVar()
        self.speed.set(0.0)
        if not self.running_tensorflow:
            if self.file_dir != '':
                self.model = tf.keras.models.load_model(os.path.join(self.file_dir, MODEL_SAVE))
                self.running_tensorflow = True
                self.model_running = True

                self.main_menu()
                self.place_acceleration()
                self.place_img()

    # Phase 1 of training, it creates the reference log from the driving log.
    # If the reference log exists, it skips to Phase 2.
    def train_1(self):
        if not self.running_tensorflow:
            if self.file_dir != '':
                self.running_tensorflow = True
                try:
                    self.train_3()
                except FileNotFoundError:
                    if os.path.exists(os.path.join(self.file_dir, RAW_LOG_NAME)):
                        reference = threading.Thread(target=create_reference,
                            args=(os.path.join(self.file_dir, RAW_LOG_NAME), os.path.join(self.file_dir, REFERENCE_NAME)))
                        reference.start()
                        self.loading_2()

    # Phase 1 of training, this is basically the progress bar for the reference log creation.
    def loading_2(self):
        from create_reference_log import index_num, total_index, taken, finished

        if type(self.master.place_slaves()[1]) == ttk.Progressbar:
            self.master.place_slaves()[1].value = index_num+1
        else:
            self.progress = ttk.Progressbar(
                self.root.pack(),
                orient=tk.HORIZONTAL,
                length=350,
                mode='determinate',
                value=index_num+1, maximum=float(total_index)
            ).place(relx=0.7, rely=0.75, anchor=tk.CENTER)

        left = total_index - index_num
        try: est = str(timedelta(seconds=round(taken / index_num * left)))
        except ZeroDivisionError: est = '?:??:??'

        if type(self.master.place_slaves()[1]) == tk.Label:
            self.master.place_slaves()[1].text = \
                f'Itering over row: {index_num+1}/{total_index}\n'\
                f'Time taken: {str(timedelta(seconds=round(taken)))}\n'\
                f'Estimated time left: {est}'
        else:
            self.text = tk.Label(
                self.root.pack(),
                text=f'Itering over row: {index_num+1}/{total_index}\n'
                     f'Time taken: {str(timedelta(seconds=round(taken)))}\n'
                     f'Estimated time left: {est}',
                bg='grey'
            ).place(relx=0.7, rely=0.85, anchor=tk.CENTER)

        if finished:
            self.train_3()
        else:
            self.after(50, self.loading_2)

    # Phase 2 of training, it... trains the model. Yeah.
    def train_3(self):
        self.main_menu()

        df = pd.read_csv(os.path.join(self.file_dir, REFERENCE_NAME), index_col=0)
        df = df.sample(frac=1)
        df = df.drop_duplicates(subset=['wheel_angle'])
        df = df.to_dict(orient='list')

        X = np.array([cv2.imread(n) for n in df['center_dir']])
        y = np.array([df['wheel_angle']])
        train_ds, test_ds = train_test_split(X, y)

        trainer = threading.Thread(target=train_model,
            args=(create_model(), train_ds, test_ds, os.path.join(self.file_dir, MODEL_SAVE), False))
        trainer.start()
        self.loading_4()

    # Phase 2 of training, this is basically the progress bar for the model training.
    def loading_4(self):
        from model import timer, finished

        if type(self.master.place_slaves()[1]) == ttk.Progressbar:
            self.master.place_slaves()[1].value = sum(timer.finished)
        else:
            self.progress = ttk.Progressbar(
                self.root.pack(),
                orient=tk.HORIZONTAL,
                length=350,
                mode='determinate',
                value=sum(timer.finished), maximum=sum(timer.increments)
            ).place(relx=0.7, rely=0.75, anchor=tk.CENTER)

        if type(self.master.place_slaves()[1]) == tk.Label:
            self.master.place_slaves()[1].text = \
                 f'Performing actions: {sum(timer.finished)}/{sum(timer.increments)}\n'\
                 f'Time taken: {timer.total_time()}\n'\
                 f'Estimated time left: {timer.time_left()}'
        else:
            self.text = tk.Label(
                self.root.pack(),
                text=f'Performing actions: {sum(timer.finished)}/{sum(timer.increments)}\n'
                     f'Time taken: {timer.total_time()}\n'
                     f'Estimated time left: {timer.time_left()}',
                bg='grey'
            ).place(relx=0.7, rely=0.85, anchor=tk.CENTER)


        if finished:
            self.main_menu()
            self.running_tensorflow = False
        else:
            self.after(50, self.loading_4)

    # In short, the part that will be looped when the program is connected to the Udacity simulator.
    @static_vars(speed_controller=PIController())
    def controller(self, data):
        if (data is None) or (not self.model_running): return 0, 0
        image = base64_to_cvmat(data["image"])

        if self.layer_reference[self.layer_shown] is None:
            self.img = image

        image = np.expand_dims(image, 0)

        for layer in self.model.layers:
            image = layer(image)
            if layer.name == self.layer_reference[self.layer_shown]:
                output = image.numpy()[0]
                show_image = []
                for n in range(output.shape[-1]):
                    show_image.append(output[..., n]*255)
                self.img = row_show(np.array(show_image), layer_name=layer.name, **self.look_for[layer.name])

        result = image.numpy()
        steering = result[0][0]

        if float(data['speed']) >= self.speed.get():
            throttle = 0.0
        else: throttle = 1.0
        if self.speed.get() == 0.0:
            steering = 0.0

        return steering, throttle

    # It is what it says, it starts the Udacity simulation.
    def start_sim(self):
        self.found = False
        win32gui.EnumWindows(self.callback, None)
        if not self.found:
            os.startfile(SIMULATION_EXE)
            self.found = True
        self.main_menu()

    # It is what it says, it kills the Udacity simulation.
    def kill_sim(self):
        self.found = False
        win32gui.EnumWindows(self.callback, None)
        if self.found:
            os.kill(self.pid, signal.SIGTERM)
            self.found = False
        self.main_menu()


    # Stuff to find the PID value of the Udacity simulator.
    def get_pid(self):
        for pros in psutil.process_iter(['name']):
            if pros._name == PROCESS_NAME:
                return pros.pid

    # Stuff to find the PID value of the Udacity simulator.
    def callback(self, hwnd, extra):
        GetWindowText = ctypes.windll.user32.GetWindowTextW
        GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW

        length = GetWindowTextLength(hwnd)
        buff   = ctypes.create_unicode_buffer(length + 1)
        GetWindowText(hwnd, buff, length + 1)
        if buff.value == BUFF_VALUE:
            self.pid = self.get_pid()
            self.found = True

    # Delete all items
    def del_all(self):
        for slave in self.master.place_slaves():
            slave.destroy()

    # Detete last item placed, or under some special circumstances.
    def del_last(self, spl=None, last_num=0):
        if spl is None:
            self.master.place_slaves()[last_num].destroy()
        else:
            if type(self.master.place_slaves()[last_num]) == spl:
                self.master.place_slaves()[last_num].destroy()


if __name__ == '__main__':
    app = Application()
