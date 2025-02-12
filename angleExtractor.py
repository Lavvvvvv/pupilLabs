import final.data_extractor as extr
import numpy as np
import final.classification as clas
import math


gaze=extr.getData() #leftEyePointUser(x,y,z),rightEyePointUser(x,y,z),leftEyeOrigin(x,y,z),rightEyeOrigin(x,y,z), pupilDiamLeft,pupilDiamRight,timestamp
class_left,class_right=clas.classify(gaze)

device_timestamps=gaze[:,14]
discrete_timestamps= class_left[0]#timestamp from descrete classification in classification.py, indicates when the fixation or saccade start
class_left=class_left[1] #list
firstTime= True
result=[]
'''
see formula 2 https://confluence.iabg.de/display/HSI/Pipeline+and+To-dos

x_p1, y_p1, z_p1 = coordinate of gaze point 1 in user coordinates
x_p2, y_p2, z_p2 = coordinate of gaze point 2 in user coordinates
x_o1, y_o1, z_o1 = coordinate of gaze origin 1 in user coordinates
x_o2, y_o2, z_o2 = coordinate of gaze origin 2 in user coordinates
t1=start time
t2=end time
'''
saccade_duration=np.array(([]))
for i in range(timestamps.size-1):
    label=class_left[i,:]
    current_time=timestamps[i]
    if i==0: #idk why first few values are always zero
            continue
    
    if label[1]=='Saccade' :
        match firstTime:
            case True:
                saccade_duration=np.append(saccade_duration,label[0])
                p1=np.array[gaze[i,0:5]] #1st row, xyzleftrightpoint, 2nd row xyzoriginleftrightpoint
                o1=np.array[gaze[i,6:11]]
                t1=current_time
                firstTime= False
                

            case False:
                p2=np.array[gaze[i,0:5]]
                o2=np.array[gaze[i,6:11]]
                t2=current_time
                

                
        
    elif label[1]=='Fixation' :
        
        if firstTime ==False:
            
                #calculate with formula
                directionVector1=p1-o1 #vector from origin to point in state 1
                directionVector2=p2-o2 #vector from origin to point in state 1
                magnitude_xy1=np.linalg.norm(directionVector1[:2]) #magnitude of xy direction vector1
                magnitude_xy2=np.linalg.norm(directionVector2[:2]) #magnitude of xy direction vector1
                z1=directionVector1[2]
                z2=directionVector2[2]
                detlta_time=t2-t1
                tanquotient1=magnitude_xy1/z1
                tanquotient2=magnitude_xy2/z2
                delta_angle=math.atan(tanquotient2)-math.atan(tanquotient1)
                angular_rate=delta_angle/detlta_time
                firstTime=True #reset first time to look for saccades again
                result.append((t1,t2,angular_rate))
  
        

        

