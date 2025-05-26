"""
callable = classify(gaze)
param : gaze nx15 numpy array from data_extractor
leftEyePointUser(x,y,z),rightEyePointUser(x,y,z),leftEyeOrigin(x,y,z),rightEyeOrigin(x,y,z), pupilDiamLeft,pupilDiamRight,timestamp

return : classification nx2 numpy array containing (timestamp,"fixation" or "saccade")
the return values are the "special events" that happen, so it only shows when the events start

DEPRATED: function not needed anymore, this was in development when real time was needed
"""
import numpy as np
from lib.depr.utils_DEPR import discrete_to_continuous, continuous_to_discrete, _get_time
import csv

def classify_velocity(x, y, time, threshold=None, return_discrete=False):
    """I-VT velocity algorithm from Salvucci & Goldberg (2000).
    
    One of several algorithms proposed in Salvucci & Goldberg (2000),
    the I-VT algorithm classifies samples as saccades if their rate of
    change from a previous sample exceeds a certain threshold. I-VT 
    can separate between the following classes:
    ```
    Fixation, Saccade
    ```
    
    For reference see:
    
    ---
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
    and saccades in eye-tracking protocols. In Proceedings of the 
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    ---
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze 
        arrays or a float/int that represents the sampling rate.
    threshold : float
        The maximally allowed velocity after which a sample should be 
        classified as "Saccade". Threshold can be interpreted as
        `gaze_units/s`, with `gaze_units` being the spatial unit of 
        your eyetracking data (e.g. pixels, cm, degrees). If `None`,
        `mad_velocity_thresh` is used to determine a threshold.
        Default=`None`.
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.
        
    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event 
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
        """
    # process time argument and calculate sample threshold
    times, sfreq = _get_time(x, time, warn_sfreq=True)
    
    # find threshold if threshold is None
    if threshold == None:
        threshold = mad_velocity_thresh(x, y, times)
    # express thresh in terms of freq
    sample_thresh = threshold / sfreq
    
    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([vels, [0.]])
    
    # define classes by threshold
    classes = np.empty(len(x), dtype=object)
    classes[:] = "Fixation"
    classes[vels > sample_thresh] = "Saccade"
    
    # group consecutive classes to one segment
    segments = np.zeros(len(x), dtype=int)
    for idx in range(1, len(classes)):
        if classes[idx] == classes[idx - 1]:
            segments[idx] = segments[idx - 1]
        else:
            segments[idx] = segments[idx - 1] + 1
    
    # return output
    if return_discrete:
        segments, classes = continuous_to_discrete(times, segments, classes)     
    return segments, classes

def mad_velocity_thresh(x, y, time, th_0=200, return_past_threshs=False):
    """Robust Saccade threshold estimation using median absolute deviation.
    
    Can be used to estimate a robust velocity threshold to use as threshold
    parameter in the `classify_velocity` algorithm.
    
    Implementation taken from [this gist] by Ashima Keshava.
    [this gist]: https://gist.github.com/ashimakeshava/ecec1dffd63e49149619d3a8f2c0031f
    
    For reference, see the paper:
    
    ---
    Voloh, B., Watson, M. R., König, S., & Womelsdorf, T. (2019). MAD 
    saccade: statistically robust saccade threshold estimation via the 
    median absolute deviation. Journal of Eye Movement Research, 12(8).
    ---
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze 
        arrays or a float/int that represents the sampling rate.
    th_0 : float
        The initial threshold used at start. Threshold can be interpreted 
        as `gaze_units/s`, with `gaze_units` being the spatial unit of 
        your eyetracking data (e.g. pixels, cm, degrees). Defaults to 200.
    return_past_thresholds : bool
        Whether to additionally return a list of all thresholds used 
        during iteration. Defaults do False.
        
    Returns
    -------
    threshold : float
        The maximally allowed velocity after which a sample should be 
        classified as "Saccade". Threshold can be interpreted as
        `gaze_units/ms`, with `gaze_units` being the spatial unit of 
        your eyetracking data (e.g. pixels, cm, degrees).
    past_thresholds : list of float
        A list of all thresholds used during iteration. Only returned
        if `return_past_thresholds` is True.
        
    Example
    --------
    >>> threshold = mad_velocity_thresh(x, y, time)
    >>> segments, classes = classify_velocity(x, y, time, threshold)
    """
    # process time argument and calculate sample threshold
    times, sfreq = _get_time(x, time, warn_sfreq=True)
    # get init thresh per sample
    th_0 = th_0 / sfreq
    
    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([[0.], vels])
    
    # define saccade threshold by MAD
    threshs = []
    angular_vel = vels
    while True:
        threshs.append(th_0)
        angular_vel = angular_vel[angular_vel < th_0]
        median = np.median(angular_vel)
        diff = (angular_vel - median) ** 2
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        th_1 = median + 3 * 1.48 * med_abs_deviation
        # print(th_0, th_1)
        if (th_0 - th_1) > 1:
            th_0 = th_1
        else:
            saccade_thresh = th_1
            threshs.append(saccade_thresh)
            break
    
    # revert units
    saccade_thresh = saccade_thresh * sfreq
    threshs = [i * sfreq for i in threshs]
    
    if return_past_threshs:
        return saccade_thresh, threshs
    else:
        return saccade_thresh  

def classify(gaze):
    """
    param : gaze nx15 numpy array from data_extractor
    leftEyePointUser(x,y,z),rightEyePointUser(x,y,z),leftEyeOrigin(x,y,z),rightEyeOrigin(x,y,z), pupilDiamLeft,pupilDiamRight,timestamp
    return : classification nx2 numpy array containing (timestamp,"fixation" or "saccade")
    """
    
    gaze_right_eye_x=gaze[:,0]
    gaze_right_eye_y=gaze[:,1]
    device_time_stamp=gaze[:,14]

    time,classification= classify_velocity(gaze_right_eye_x, gaze_right_eye_y, device_time_stamp,return_discrete=True)
    
    print("classified successfully")
    return time,classification
