# Udacity simulation executable location
SIMULATION_EXE = 'C:/Users/8051/Desktop/190595E/(testing)beta_simulator_windows/beta_simulator.exe'
# Buff value when getting simulation PID
BUFF_VALUE = 'self_driving_car_nanodegree_program'
# Process name for launching/stopping simulation
PROCESS_NAME = 'beta_simulator.exe'

# Driving log name
RAW_LOG_NAME = 'driving_log.csv'
# Model name
MODEL_SAVE = 'model.tf'

# Angle averaging distribution on data
ANGLE_REDESTRIBTION = 3
# Driving log reference name
REFERENCE_NAME = 'reference_log.csv'

# Steering angle variation for different camera angles
CAMERA_ANGLE_VARIATION = 2.0 / 25.0

# Image cropping for model input
CROP_LEFT   = 0.00
CROP_RIGHT  = 0.00
CROP_TOP    = 0.35
CROP_BOTTOM = 0.10

# Image resize for model input
RESIZE = (109, 253)

# Image pixel value rescaling
VALUE_RESCALE_MIN = 0.0
VALUE_RESCALE_MAX = 1.0

# Seed for model creation
SEED            = None
# Train Test Split, test percentage
TEST_PERCENTAGE = 0.0

# Batch, epoch and warmup epoch for model training
BATCH 	= 8
EPOCHS        = 10
WARMUP_EPOCHS = 1

# Learning rate for model
LEARNING_RATE_MAX = 0.0003
LEARNING_RATE_MIN = LEARNING_RATE_MAX/100

# Tensorboard usage
USE_TENSORBOARD = True
