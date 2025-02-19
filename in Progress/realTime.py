import multiprocessing
import time

def code_B(queue):
    """Worker process B: receives values from the main process via queue."""
    while True:
        value = queue.get()  # wait for a value from the main process
        if value is None:    # sentinel received: time to exit
            print("Process B exiting.")
            break
        # Process the received value
        print("Process B received:", value)
        # Simulate some work with a sleep (optional)
        time.sleep(1)

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
        time.sleep(1)

if __name__ == '__main__':
    # Create separate queues for the two worker processes.
    queue_B = multiprocessing.Queue()
    queue_C = multiprocessing.Queue()

    # Start the two worker processes.
    process_B = multiprocessing.Process(target=code_B, args=(queue_B,))
    process_C = multiprocessing.Process(target=code_C, args=(queue_C,))
    
    process_B.start()
    process_C.start()

    try:
        counter = 0
        # Main process loop: runs continuously until a keyboard interrupt.
        while True:
            counter += 1
            result = f"Result {counter}"
            print("Main process produced:", result)
            # Send the result to both worker processes.
            queue_B.put(result)
            queue_C.put(result)
            # Sleep a bit before producing the next result.
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating processes...")
        # Send a sentinel (None) to each worker process so they can exit.
        queue_B.put(None)
        queue_C.put(None)
    finally:
        # Wait for the worker processes to finish.
        process_B.join()
        process_C.join()
        print("All processes have terminated.")
