from pupil_labs.realtime_api.simple import discover_one_device
from lib.calibration import calibration
import numpy as np
import csv
import shutil
from datetime import datetime
import os
from lib.classification import classify

def getData(calib:bool=True)-> np.array :
    '''
    getting data from eye tracker
    inputs: bool calib, if true, then calibration for monitor location is done, if false, then camera resolution will be used
    returns: np array (gaze_x,gaze_y,pupil_diam_left,pupil_diam_right)
    '''
    if not calib:
        print("Looking for the next best device...")
        device = discover_one_device(max_search_duration_seconds=20)
        if device is None:
            print("No device found.")
            raise SystemExit(-1)
        corner=np.array([[0,0],[1599,0],[1599,1199],[0,1199]]) #if not calibrated, then it will treat the whole camera as the screen
        screen_length=np.array([[1600,1200]])
        #here
        with open('screenCornerCalib.txt', mode='a', newline='') as file:
            writer = csv.writer(file)
        
        # Append values row by row
            for row in corner:
                writer.writerow(row)
        #here
    else:
        corner,device = calibration()
        screen_length=(corner[1,:]-corner[3,:]).reshape(1, 2)
        screen_length=np.abs(screen_length)
        #here
        with open('screenCornerCalib.txt', mode='a', newline='') as file:
            writer = csv.writer(file)
        
        # Append values row by row
            for row in corner:
                writer.writerow(row)
        #here
    
    try:
        data_array=np.empty((0, 5)) #gaze_x,gaze_y,pupil_diam_left,pupil_diam_right
        
        while True:
            current_data=np.array[device.receive_gaze_datum().x,device.receive_gaze_datum().y,device.receive_gaze_datum().pupil_diameter_left,device.receive_gaze_datum().pupil_diameter_right,device.receive_gaze_datum().timestamp_unix_ns]
            data_array = np.vstack([data_array, current_data])
            received_x=device.receive_gaze_datum().x
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
            if corner[0,1] < received_y < corner[3,1]:
                y_normalized=received_y-corner[0,0]
                y_normalized=y_normalized/screen_length[0,1]
                if y_normalized>1:
                    y_normalized=-1
                    x_normalized=-1
            else:
                y_normalized=-1
                x_normalized=-1
            #here
            
            with open('x_normalized.txt', mode='a', newline='') as file:
                writer = csv.writer(file)
                for num in [x_normalized]:
                    writer.writerow([num])
            #here
            #here
            with open('y_normalized.txt', mode='a', newline='') as file:
                writer = csv.writer(file)
                for num in [y_normalized]:
                    writer.writerow([num])
            #here           
            screen_pointList=(x_normalized,y_normalized)
            print(screen_pointList)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping...")
        device.close()  # explicitly stop auto-update
        np.savetxt("data.csv", data_array, delimiter=",", fmt="%d")
        classify(data_array)
        dataset_folder = datetime.today().strftime("%d.%m.%y")
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        shutil.move('screenCornerCalib.txt', os.path.join(dataset_folder, 'screenCornerCalib.txt'))
        shutil.move('x_normalized.txt', os.path.join(dataset_folder, 'x_normalized.txt'))
        shutil.move('y_normalized.txt', os.path.join(dataset_folder, 'y_normalized.txt'))
        shutil.move(dataset_folder, os.path.join('test dataset', dataset_folder))
        # device.streaming_stop()  # optional, if not called, stream is stopped on close
       