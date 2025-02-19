import multiprocessing
import time
from pupil_labs.realtime_api.simple import discover_one_device
import numpy as np





def code_B(queue):
    """Worker process B: receives values from the main process via queue."""
    while True:
        value = queue.get()  # wait for a value from the main process
        if value is None:    # sentinel received: time to exit
            print("Process B exiting.")
            break
        # process the received value
        print("Process B received:", value)
        # Simulate some work with a sleep (optional)
        time.sleep(0.4)

def code_C(queue):
    """Worker process C: receives values from the main process via queue."""
    while True:
        value = queue.get()  # wait for a value from the main process
        if value is None:    # sentinel received: time to exit
            print("Process C exiting.")
            break
        # Process the received value
        print("Process C received:", value)
        # Simulate some work with a sleep (optional)
        time.sleep(0.4)

if __name__ == '__main__':
    # Look for devices. Returns as soon as it has found the first device
    print("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=20)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)
    # Create separate queues for the two worker processes
    queue_B = multiprocessing.Queue()
    queue_C = multiprocessing.Queue()

    # Start the two worker processes
    process_B = multiprocessing.Process(target=code_B, args=(queue_B,))
    process_C = multiprocessing.Process(target=code_C, args=(queue_C,))
    
    process_B.start()
    process_C.start()

    # Initializing some values to be extracted
    X=np.array([])
    Y=np.array([])

    #initializing time constant
    last_sent=time.time()
    batch_interval=0.5 #gather data for this amount of time
    try:
        
        # Main process loop: runs continuously until a keyboard interrupt
        while True:
            data=device.receive_gaze_datum()
            current_time = time.time()
            X=np.append(X,data.x)
            Y=np.append(Y,data.y)
            if current_time - last_sent >= batch_interval:
                result=np.stack((X,Y), axis=1)
                # Send the result to both worker processes
                queue_B.put(result)
                queue_C.put(result)
                X=np.array([])
                Y=np.array([])
                


            
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating processes...")
        queue_B.put(None)
        queue_C.put(None)
    finally:
        process_B.join()
        process_C.join()
        print("All processes have terminated.")
