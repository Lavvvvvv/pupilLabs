from pupil_labs.realtime_api.simple import discover_one_device
import time
import numpy as np
import multiprocessing

"""
Calibration, to set the eye tracking boundaries to the screen, 
does not track the screen, so coordinates are static

Deprecated: function not needed anymore, this was in development when real time was needed

"""


def averaging(queue):
    
    coordinate_npnx2 = queue  # wait for a value from the main proces
    sum=np.sum(coordinate_npnx2, axis=0, keepdims=True)
    point_quantity=coordinate_npnx2.shape[0]
    result=sum/point_quantity
    return result
        

def calibration()->np.ndarray:
    '''Calibrating using 4 corners of the screen
    participants should look at each corner of the screen for 5 seconds and move on, sequence should be top left, top right, bottom right, bottom left
    return: np.ndarray [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], device
    '''
    try:
        #getting eye tracker
        print("Looking for the next best device...")
        device = discover_one_device(max_search_duration_seconds=20)
        if device is None:
            print("No device found.")
            raise SystemExit(-1)



        # device.streaming_start()  # optional, if not called, stream is started on-demand
        calib_duration=4.0 #data set duration that will be averaged to get a corner point
        calib_buffer =1.0 # buffer time to let the eye settle to a point
        first_corner=True
        start_time=time.time()
        updated_time=time.time()
        corner_coordinate=np.array([[0,0]])

        for i in range(4):
            print('Calib start')
            x_collection=np.array([[0]])
            y_collection=np.array([[0]])
            
            start_time=time.time() #time when calibration for one point is started

            while updated_time-start_time<=calib_duration:
                if first_corner:
                    first_corner=False
                    time.sleep(calib_buffer)
                else:
                    elapsed_time=float(time_before-time_before)
                    calib_buffer_needed=calib_buffer-elapsed_time #calculating time needed to rest including the time to process the averaging
                    time.sleep(calib_buffer_needed)

                #taking account to how much time needed to process the data    
                time_before=time.time()
                x_collection= np.append(x_collection,[[device.receive_gaze_datum().x]],axis=0) #1600x1200
                y_collection= np.append(y_collection,[[device.receive_gaze_datum().y]],axis=0)
                updated_time=time.time()
                
            x_collection=x_collection[1:,:]
            
            y_collection=y_collection[1:,:]
            collection=np.hstack([x_collection,y_collection])
            collection=averaging(collection)
            corner_coordinate=np.vstack([corner_coordinate,collection])
            
        corner_coordinate=corner_coordinate[1:,:] 
        print('Corner coordinates are ',corner_coordinate)
        return corner_coordinate,device
    except Exception as e:
        print(f"An error occurred: {e}",'\n calibration not finalized',)





