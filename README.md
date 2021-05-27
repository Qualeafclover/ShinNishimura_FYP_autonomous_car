# NYP FYP Autonomous driving vehicle By Shin Nishimura

Tested versions:
Python 3.8 on Windows 10

# Quick start:

Step 1:

Download Udacity's car simulation, version 2 at: https://github.com/udacity/self-driving-car-sim

Extract the downloaded folder and get the directory of where the executable is located.

Directory example: 'C:/Users/8051/Desktop/190595E/(testing)beta_simulator_windows/beta_simulator.exe'

Step 2:

Make sure you have installed git. If you have not, install it here: https://git-scm.com/downloads

Open your command prompt and go to the directory where you want to clone this repository.

Then, run the following command:

`git clone https://github.com/Qualeafclover/ShinNishimura_FYP_autonomous_car.git`

Step 3:

Open the cloned repository in Pycharm. Make sure to install the dependencies from requirement.txt

Step 4: 

Create a new folder, preferably inside the cloned repository. This will be used to store the data the AI will be using.

(Don't name the folder 'logs')

Step 5:

Go to configs.py and replace the variable string of SIMULATION_EXE with the executable directory obtained at Step 1.

Step 6:

Run main.py and click on 'Start Simulation'. This will cause the car simulation software to start.

Enter the training mode, and choose where to save the images you have recorded by clicking R.

Navigate to the folder you have created on Step 4 and select it as the save location.

Click R again to start the recording, and drive around the track. The reccomended number of laps to do is 1.

To mark the end of your recording, click R one las time.

Step 7:

After the software collects its data, click on 'Select work file'

Then, click on 'Load file for training'.

The 2 step training process will then start. You will know it is over after 2 progress bars go by.

Step 8:

Return to the main menu of the Udacity car imulator. Then, click 'Autonomous Mode'

Click on 'Run Trained Model' on the UI, then the model trained on Step 7 start.

Step 9:

Use the scale widget to control the speed of the car. 

The car should start driving.
