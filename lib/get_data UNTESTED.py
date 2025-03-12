from pupil_labs.realtime_api.simple import discover_one_device
from lib.calibration import calibration
import numpy as np
import csv
import shutil
from datetime import datetime
import os
import classification


def getData(calib:bool=True)-> np.array :
    '''
    getting data from eye tracker
    inputs: bool calib, if true, then calibration for monitor location is done, if false, then camera resolution will be used
    returns: np array
    '''
    if not calib:
        print("Looking for the next best device...")
        device = discover_one_device(max_search_duration_seconds=20)
        if device is None:
            print("No device found.")
            raise SystemExit(-1)
        corner=np.array([[0,0],[1599,0],[1599,1199],[0,1199]]) #if not calibrated, then it will treat the whole camera as the screen
        screen_length=np.array([[1600,1200]])
    
    try:
        x_compiled =np.array([]) #collection of all x during runtime
        y_compiled =np.array([]) #collection of all x during runtime
        while True:
            received_x=device.receive_gaze_datum().x
            x_compiled[-1]=received_x
            if corner[0,0] < received_x < corner[1,0]:
                x_normalized=received_x-corner[0,0]
                x_normalized=x_normalized/screen_length[0,0]
                if x_normalized>1:
                    y_normalized=-1
                    x_normalized=-1
            else:
                x_normalized=-1
                y_normalized=-1

            received_y=device.receive_gaze_datum().y
            y_compiled[-1]=received_y
            if corner[0,1] < received_y < corner[3,1]:
                y_normalized=received_y-corner[0,0]
                y_normalized=y_normalized/screen_length[0,1]
                if y_normalized>1:
                    y_normalized=-1
                    x_normalized=-1
            else:
                y_normalized=-1
                x_normalized=-1
            
            timestamp=device.receive_gaze_datum().timestamp_unix_seconds

            screen_pointList=(x_normalized,y_normalized)
            print(screen_pointList)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping...")
        device.close()  # explicitly stop auto-update
        classification.classify_velocity(x_compiled,y_compiled,timestamp)

        
        
       