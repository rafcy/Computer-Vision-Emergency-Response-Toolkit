**Submission For "Where is the Missing Person" Contest**
Febuary 2020
Rafael Makrigiorgis - makrigiorgis.rafael@ucy.ac.cy
Panayiotis Kolios - kolios.panayiotis@ucy.ac.cy


*Works in any python environment (Windows/Linux platforms).*

**Package requirments:**
Opencv, numpy, imutils

**Description:**

* We have 3 main functions, one for RGB Video detection,  one for RGB image detection and the last one for thermal video motion detection. 
Set from the main function to True the only one you wish to run. 

* For the RGB video/image functions we have trained a tinyyolov3 network using aproximately 2k images from the IPSAR Dataset. These files are needed and they are already configured as global variables. Basically these functions use opencv's neural network library to read the network and output the possible detections based on the confidence set from the user. The output detections are then drawn using rectangles.

* For the thermal video detections we have used a motion detection algorithm. At first we are finding differences between frames and then we calculate the contours of these differences.
- If the contours are a lot it means it has a lot of image movement so the code ignores it. Also it resets all the boxes saved if such movement was done 10 times.
- If it has a number of contours below a certain threshold the motion detection starts. First time initializing all the contours found on the image using bounding boxes.
-Then, on following frames, it checks the IOU score of each new contour box compared with all the previous boxes. If an IOU score is above a certain threshold it matches the box with the same ID and saves it into a list so it can track it.
- If the saved boxes for each box id are more than 5, it calculates the euclidean distance of its trajectory. If the euclidean distance is above a certatin threshold, it means it is moving and it draws the latest box and its trajectory.
-In the meanwhile, if a saved box is not detected for 20 frames, it resets the boxes of its ID so it will not match it with something else. 

**Notes**
* resize = True: you resize the input frames in order to increase performance of detections (most desirable for video detections).
You can set the desired image width/height from global variables.

* save_results = True: you always save your results to Detections/images or Detection/videos (depends on the detection) folders inside your current directory.
* *For the Video detections, the video frames capture is done using threading (recv_thread() function) in order to increase the performance significantly.*

**#unctions included:**
- Finding IOU of 2 boxes: bb_iou()
- Finding euclidean distance of 2 points:  MidPoint()



**HOW TO USE:**
- Install required packages
- Set as True for the use case you want to use (video RGB/thermal, RGB image)
- Configure NN config/weights/label files for RGB detections
- Change the image resize width/height (if resize = True)
- Change video/image file_name accordingly from global variables
- Run python missing_person_full.py 

**References:**
Ž. Marušić, D. Božić-Štulić, S. Gotovac and T. Marušić, "Region Proposal Approach for Human Detection on Aerial Imagery," 2018 3rd International Conference on Smart and Sustainable Technologies (SpliTech), Split, 2018, pp. 1-6.
