# OBJECT RECOGNITION PIPELINE FOR SMART

# FARMING

## Sruthin Reddy Swathi Meka Sourav Kumar Mounika Chenna

## Venkata Akhila

## September 1, 2019

```
Abstract
In this Global era, Smart farming is becoming a more prominent technique, winding up with
more spotlight on innovative technologies as well as image and video processing techniques. The
identification of individual cow plays a significant role in the success of Dairy Farming by de-
tecting and localizing the cow. Previously, many systems have followed by sensor technologies
(RFID) to identify and detect the cow position, but this technology has some limitations and
doesn’t fulfil all the levels of required categories such as good health and physical status. Later,
people used to capture the movements of the cow in a digital camera then, machine learning and
pattern recognition techniques have used to recognize patterns and regularities in the captured
images/videos. Based on those technologies, a lot of pre-trained object recognition models have
evolved. In our project, we consider a data set with cow images, among them we extract some
random images (jpg), regarded as input for three pre-trained models such as YOLO, SSD and
Mask RCNN. The output (bounding boxes, coordinates )that are predicted by three models are
compared with ground-truth. We calculate the individual models performance in terms of time
and accuracy using the evaluation metric called intersection over union (IOU). The main aim of
our project is to select the best model based on accuracy and average time. In extension, we apply
segmentation on the output of the best model to separate each object in an image, then classify
those objects to form a cow dataset. In the future, we can use this dataset to train the models for
better accuracy.
Keywords: Smart farming,YOLO, SSD, Mask RCNN, RFID, IOU, Segmentation, Classification
and Object detection.
```

## Download Source Code : https://1drv.ms/u/s!An5NTTrGR0UVkHUHROkP27Emg9m2?e=cTgRDg

## 1 Introduction

Smart Farming is the latest innovation to overcome the problems of traditional farming and to improve
the quantity and nature of agricultural products. Today farmers are following modern technologies
such as communication technologies, the Internet of Things and their related engineering management
procedures to fulfil all the requirements of smart farming. By utilizing these procedures, farmers can
efficiently monitor the prerequisites of each animal. Latest dairy forms use electronic equipment for
monitoring and collecting the data of cows present in the farm. Identification of each individual cow
plays an important role to implement an automatic monitoring system.

Most existing approaches mount sensor devices on the body of the cow for recognition. Apparently in
this procedure sensors can produce strain on the cow’s health and also requires the high attention of
labour to audit the sensor data. To overcome these problems, we set up a camera on a dairy farm. This
camera records continuous observations of the cows. From stored data, we extract images comprising
of multiple angles such as the full face, right side, left side, back, front, corner of each cow and we
utilize these images to recognize a cow. After extracting the images, we perform the preprocessing
step. Through preprocessing we organize all required images from different folders into a single folder
along with assigned labels with the help of C# code.

We consider three pre-trained models namely Yolo, SSD, mask-RCNN for object detection with the
help of machine learning libraries such as Numpy, Pandas, OpenCV. The main task of our project is to
compare three pre-trained models in terms of average time and accuracy. We give preprocessed images
folder as input to three pre-trained models. These models give predicted bounding boxes around the
object in the images along with class labels. From the models output, we also get the information of
Total objects detected, time is taken for detecting each object and total time taken for detecting all
objects. We compare the bounding boxes which were predicted by three models with Ground truth
bounding box. Ground truth can be obtained by manual annotation. For manual annotation, we used
Dark Label tool. This tool gives annotated images with a class label along with coordinates. Now in
the evaluation part, we compare predicted bounding boxes by models with ground truth using eval-
uation metric called Intersection Over Union (IOU). From IOU scores we calculate precision, recall,
accuracy scores.

This paper is organised as follows Section 2 explains system architecture, pre-processing, the working
method and implementation of Yolo, SSD, Mask-RCNN and ground truth. Section 3 explains the
evaluation and comparison of three models.Section 4 explains additional work Section 5 describes
conclusion and future work.

## 2 System Architecture

```
Figure 1: System Architecture
```

Initially, we set up a digital camera on a dairy farm to capture the movements of a cow. Then we
extract some sample images with excessive visual quality. As our images are in various folders, we do
preprocess using machine learning libraries and make them as a single folder to simplify the selection
process. Our main aim is to compare three pre-trained models (YOLO, SSD, Mask R-CNN), we make
a ground truth by annotating the extracted images and assigning a label manually using a dark label
(image annotation) tool. On another side, we process YOLO, SSD, Mask R-CNN by taking extracted
images as input then, we will get three outputs from these three models as Localization and Bounding
boxes with labels. Now we compare these three models with ground truth by using IOU(intersection
over union) as a metric while we select one among three models based on speed and accuracy. Besides

we perform segmentation and classification on the output of the selected model to form a dataset.

### 2.1 Pre-Processing

Image pre-processing is the term for the operations on the images at the lowest level of abstraction.
The main aim of this method is an improvement of image data that overcome unwanted distortions
or enhance some image features like brightness, contrast and intensity for analysing tasks. Moreover,
it improves the image data and also performs the classification, feature extraction and transform the
images for further processing [14]. For our project, we were supplied by thousands of images in three
main folders namely camera1, camera 2, camera 3. In these 3 folders, there are so many different sub-
folders and these subfolders also contain sub subfolders, so we cannot give these folders and subfolders
directly as input to pre-trained models.

A solution for this, we downloaded 3 main folders into our local machines. After downloading, we
wrote a programming code which takes multiple folders as input and aligns all the images from different
folders to a single folder and this code also assign continuous label numbers for the images starting
from 1 till the end of images. We gave camera 1, camera 2, camera 3 as input to the code. We created
the ”All Images” folder and we gave this folder as the destination folder. When we execute the code,
it takes all images from 3 main folders and it places all the images in destination folder( i.e All images
folder) with the label assigned to each image like PIC0001.JPG, PIC0002.JPG,...so on, till the end of
images. We copied nearly 40,000 images from 3 main folders to a single folder. From these 40,
images, we selected 320 images and gave these 320 images as input to our pre-trained models. This is
a pre-processing step of our project.

### 2.2 YOLO

As the name implies, YOLO looks the whole image only once for object detection. It averts predicting
several times for different regions in an image. One of the special features of YOLO is speed enhance-
ment for detecting objects. Yolo is a single-stage detector, it predicts numerous bounding boxes as
well as class probabilities for predicted boxes. The working method of Yolo is it first splits the image
into grid cells of size SxS, each grid cell implies N bounding boxes and confidences. For individual
grid, perform image classification for clasifying the objects and localization to locate the objects in
an image. Initially, it determines the existence of an object in the grid and the center coordinates,
width, height can be calculated relative to the appropriate grid cell. Forward and backward propa-
gation can be performed for training purposes. Implementation of forward propagation followed by
moving an image to model for the accomplishment of testing the model. If two or more gids have same
object, calculate the midpoint of the object and determine the grid which contains the midpoint of
the object and then accommodate the object into that grid. The probability of arising the objects in
the same grid cell can be minimized by incrementing grids. The confidences values imply how confi-
dent the model is that the existence of an object in the box as well as precision of the predicted box [1].

Each individual boundary box consists five predictions that are center coordinates(x,y), width(w),
height(h) and confidence values. These probabilities are conditional on the grid cell holding an object.
Another special feature of YOLO is it produces less number of errors in contrast to Fast R-CNN.
Yolo is extremely generalizable [4], it is less probable to disrupt when applied to advanced domains
or sudden inputs. It also averts false positives. Yolo demands powerful spatial restrictions on bound-
ing boxes predictions. By considering the individual grid cell, it only has one class and predicts two
boxes.So,this spatial restriction diminishes the capacity of the model for predicting adjacent objects.

Implementation

For detecting cows in an image, we consider YOLO as one of the object detecting algorithm or pre-
trained model for object detection using OpenCV machine library. First, we trained YOLO on the
COCO dataset [7] .COCO is Common Objects in Context as it consists of 80 labels, 80,000 training

images, and 40,000 validation images[2]. Then we load our class labels and set random color values for
each class label. Later we derive the paths to the yolo weights and configuration files through-loading
yolo from disk using OpenCV’s DNN function called CV2.dnn.readNetfromDarkNet. Then input im-
age can be loaded and extract its dimensions and discover the names of the output layer from the
model. We then perform a forward pass in yolo and provide bounding boxes with a high probability
area. Then analyse the total time taken by yolo. To visualize our results, we initialize the lists of an
area of interest boxes, confidences, classes. Boxes represent the bounding boxes around the object,
confidence values indicate that yolo assigns to an object. We gave confidence value as 0.5, we filter out
the objects that do not meet the value and classID’s are detected object’s class label. Then perform
iterations overall an area of interest and area of detected area.

Through the predicted confidence value more than user input we filter out the weak predictions and
scale the bounding box coordinates to display box properly. The dimensions and coordinates of the
individual predicted bounding boxes can be obtained. Predicted bounding box coordinates by yolo is in
the form of (centerX, centerY, width, height) [3]. With the help of center coordinates, the top and left
corner of the bounding box coordinates can be derived. Finally, the lists of boxes, confidences, classID’s
will be updated. To suppress the weak and overlapping bounding boxes, Non-maxima suppression
can be performed as it preserve the bounding boxes that have high confidence values. Non-maxima
suppression also provides that we do not have any unnecessary or extraneous bounding boxes. Now
considering at least one area of interest predicted, iterate over the indexes that are provided by non-
maxima suppression.With the help of random color values, produce the bounding box as well as text
on an image and the resulting image can be displayed. We consider 320 images as input for yolo. We
compared the predicted bounding box with the ground truth boundary box that is obtained through
manual annotation. We used Intersection Over Union [11] which is an evaluation metric of accuracy
to compare the predicted boundary box with ground truth. Intersection Over Union is the ratio of
the area of overlap to the area of union. We also calculated the average time taken by yolo for 320
annotated images as input.

```
Figure 2: Object Detection using Yolo
```

### 2.3 SSD

The main motivation behind object detection is to identify objects present in the image and draw a
predicted bounding box around the objects present in the image. Single Shot Detector is one of the ob-
ject detection algorithms available in the current day world. SSD is the first one-shot detector among
all object detection models, which achieve an accuracy almost to accuracy of two-stage detectors [13]
and still, SSD can work in real-time. To match the accuracy with two-staged detectors, many changes
have been made in the structure of one-shot detectors by overcoming the issues of SSD and also added
an extra stage of refinement in single-shot detection pipeline, but still many users utilize SSD only as
starting point. Default bounding boxes represent bounding boxes according to their positions, size.
SSD has 8732 default boxes.

SSD architecture has been distributed into three different parts, namely Base networks, Extra feature
layers and prediction Layers. Base networks are considered as initial layers of any standard objection
detection network and these base networks of SSD are pre-trained n coco dataset. SSD is based on
VGG-16 neural networks type. at the final step, fully connected layers are assumed as convolutional
layers. Feature map of size 19 x 19 x 1024 is the result of base networks in SSD. At the highest point
of base systems, additional 4 convolutional layers are joined with the end goal that the size of feature
maps will be diminished till feature map of size 1 x 1x 256 is acquired. Final part i.e prediction layers
are very important in the architecture of SSD. For classification scores and bounding box coordinates,
multiple feature scales are used rather than single feature maps. This is where default boxes and fea-
ture map is utilized. Most importantly the convolutional layers are used for bounding box predictions.
The main advantage of SSD is, it bridges the gap between speed and accuracy trade-off. [12]

Implementation

SSD is one of the three object detection models which were used in this project. SSD is a pre-trained
model which is trained on coco data set with 90 class labels. To start the coding part, first import
the necessary packages like Numpy, OS, OpenCV. Now we import our SSD model weights named
”MobileNetSSDdeploy.caffemode” into the code. Then we load our class labels and set random colour
values for each class labels. Later we derive the paths to the SSD weights and configuration files
through-loading SSD from disk using OpenCV’s DNN function called CV2.dnn.readNetfromCaffe. Af-
ter loading the model weights, we take variables total objects detected, total time taken, average and
assign zero as the initial value to the variables. After assigning initial values, now we concentrate on
bounding boxes, SSD in which we design input bounding boxes for the image view with 512\*512 pixels.

The main aim of these bounding boxes is to represent where the object present in the image. Now pass
these designed bounding boxes into the network and obtain the predicted and area detections. Then
Analyse the total time taken by SSD. To visualize our results, we initialize the lists of an area of interest
boxes, confidences, class IDs. Boxes represent the bounding boxes around the object, confidence values
indicate that SSD assigns to an object. After obtaining confidence values, we get model prediction
confidence which is associated with high probability predicted bounding boxes. We take confidence
value as 0.5. now we filter out weak detections based on confidence value. When model predicted
confidence values are greater than the minimum model predicted confidence values, weak detections
are eliminated and this step scales the bounding box coordinates to display the bounding box correctly.

Extract the coordinates and dimensions of the bounding box.SSD returns the Bounding box coordinates
in the form of (centre X, centre Y, width, height). With the help of centre coordinates, the top and
left corner of the bounding box coordinates can be derived. Finally, the lists of boxes, confidences,
class IDs will be updated. still here, we have an output of the code with a predicted bounding box
around the object with the label assigned in the image. Execute ”cv2.imshow” command to display
the image with a predicted bounding box with the assigned label. As the output of SSD we get timing
information, the number of objectives detected and the total number of objects detected along with
the image with predicted bounding box.

```
Figure 3: Object Detection using SSD
```

### 2.4 MASK R-CNN

Mask R-CNN method preciesly detects objects in an image although giving high-quality segmentation
[6] mask for each image. The main aim of this method to develop a framework, for instance segmenta-
tion. Instance segmentation is demanding because it requires the correct detection of all objects in an
image and accurately segmenting each object. To support instance segmentation combining object de-
tection and semantic segmentation [8]. Object detection follows the classical computer approach that
detects the exact shape and gives the type of the object. It is a challenging task for classifying each
object in an image and localizing the objects based on the bounding boxes. Segmentation generates a
pixel -wise mask for each object in the photography without differentiating object instance. Generally,
Mask R-CNN comes from the extension of faster R-CNN and also fully convolutional neural networks
[9] for semantic segmentation and object detection.

Implementation

To train the Mask R-CNN model coco dataset has used. The data set includes a total of [7] 80,
classes and one background class that can detect and segment from an input image. We included all
the label names in a file of coco label. Then import required libraries, it is the first step for Mask
R-CNN. Import OS, which provides functionalities for interacting with other systems. NumPy as np
means that submodule, which refers to the parent module name. Time to handle various operations in

a model and CV2 to find various versions of the packages in search engine. Random contains a variety
of things for number generation.

Next, give the path to our coco class label and determine the pre-trained weights of the path to our
model, which is trained by the coco data set. Assigning the color values for each class label. Then
load Mask R-CNN from the disk of OpenCVs DNN function called CV2.dnn.reasdNet from Darknet.
It requires configuration path and weights path which are established via command line. Then give
input image path for instance segmentation on the image that is having some special dimensions like
height and width. To pass the bounding box through the network and obtain the area detections and
predictions. Set the timing and volume information for Mask R-CNN, that took 6f seconds.

Class ID, confidence and boxes are established for the output layer of Mask R-CNN. Extract pixel-wise
segmentation for each object, resize the mask such that the same size of the bounding boxes and finally
create a binary mask. Extracts [10] the regions of interest (ROI)of the image and visualize extracted
mask itself. Randomly select a color that will be used to visualize this particular instance segmentation
then create a transparent overlay by blending the randomly selected color with the ROI. Draw the
bounding box of the instance on the image. To compare the bounding boxes and ground truth to the
manual annotation. Intersection over union (IOU) [11] is calculated based on the ground truth and
overlapping bounding boxes. We take different cow input images like 1 and 320 for evaluating output,
accuracy speed and a time. Mask-RCNN model predicting the objects much better than other models
like YOLO and SSD. The main drawback of this model takes more time for predicting the objects.

```
Figure 4: Object Detection using Mask R-CNN
```

### 2.5 Ground Truth

Ground-Truth indicates the real output of an algorithm on an input. As our project is to compare
pre-trained (Object Recognition) models, we annotate the images by manual using DarkLabel (tool)
for ground truth because it provides an exact position, height, width, and a bounding box with labels.

```
Figure 5: Dark Label 1.
```

2.5.1 Features:

1. Support for different formats(jpg, bmp, png, ...) of images (our input images are in jpg format).
2. Various bounding box settings and label settings are available.
3. Automatic labeling function (automatically assign unique labels based on class ).
4. support for different types of data (video/images).
5. labeling by image tracker. [5]

2.5.2 Output

Dark Label provide output in required formats as it contains 63 types of output formats in terms of
ID (id of object), height(h), width(w), iname(image name), left and top positions of the bounding
box (X,Y), center coordinates(cx, cy) and n (number of objects). We have selected output format as
”iname,h[,x,y,w,h,label]”. Along with these formats, it is also provide bounding boxes as Figure:5.

```
Figure 6: Dark Label Output
```

## 3 Evaluation

The task of evaluating the performance of RCNN, SSD, and YOLO models with time and accuracy
define better choices of options provided for object detection models. The general way to represent
any object in an image or video by plotting a box i.e bounding box. The bounding box is an attribute
of (x, y) coordinates of the object in the image. The coordinate values are detected by the predicted
model which uniquely explains the region of the object in an image. The bounding box is attached
with a class label for representing the object in which class it belongs to.

Manually annotate the image with the help of an open-source tool to describe the actual and ideal
ground truth for an object. The relation between the area covered by the predicted bounding box and
ground truth helps the system to explain the accuracy. Extending the concept of accuracy with the
help of the metric called Intersection over Union (IOU) and it is a simple approach for determining the
accuracy with the help of the area covered by both bounding box and ground truth. For considering
the IOU value, the threshold value of 0.50 is taken as a margin value while indicating the accuracy.
To obtain a good prediction, we compare the threshold value with all the above values.

IOU metrics is considered with the help of basic concepts:
1.True Positive (TP): It detects cow correctly with the IOU≥threshold.
2.False Positive (FP): It detects wrongly the cow correctly with the IOU<threshold.. The bounding
box mask has no association with the ground truth.
3.False Negative (FN): detecting object wrongly. The bounding box mask matched the object where
ground truth has not been marked.
4.True Negative (TN): We are not considering this concept as it shows the misdetection of the
object. There can be possibilities the bounding box will not detect any object in an image.
Out of many objects, considering cow as a focus area. Taking all the provided images and processing
it with naming them in a serial wise.

```
Analysis with 320 annotated images:
```

```
Table 1: Time vs Object Detection for all the annotated images
```

Table 1 describes the total number of objects detected and the total time taken by three models(YOLO,
SSD, Mask R-CNN) for 320 images.

Analyse the IOU and accuracy with the total number of objects followed by three steps.
Step 1: Calculating IOU value as per number of objects

```
Figure 7: Comparision of models with Ground Truth
```

From Figure 7 we observed that YOLO, SSD predicted only one among three objects whereas Mask
R-CNN has predicted two objects while compared to the ground truth.

IOU Scores obtained from 3 models of Figure 7:

IOU for MASKRCNN: 0.8501310675223719 , for Image object:
IOU for SSD: 0.03862310120115347 , for Image object:
IOU for Yolo: 0.8568236915910064 , for Image object:
IOU for MASKRCNN: 0.8272939182030091 , for Image object:
IOU for SSD: 0.9472588480222068 , for Image object:
IOU for Yolo: 0.7427270192397063 , for Image object:
IOU for MASKRCNN: 0.0 , for Image object:
IOU for SSD: 0.0 , for Image object:
IOU for Yolo: 0.0 , for Image object:

This data has been analysed by our code ImageShowComparisonYoloRCNNSDD.ipynb file.

Step 2:Calculating Accuracy and Average IOU for figure 7

```
Figure 8: Intersection Over Union [11]
```

```
Table 2: Accuracy and Average IOU
```

Table 2: TP, FP, FN and Accuracy has been calculated with considering the figure 7. While processing
the figure 7 where the Yolo detect 2, SSD detects 1 and mask R-CNN detect 2 and then TP, FP, FN
and Accuracy has been calculated with ground truth which detects 3 objects.

Step 3: Finding precision, recall and accuracy using above formulas for all annotated images (320).

```
Table 3: Precision, Recall and Accuracy for 320 Images
```

Finally, summing up and following the 1,2 steps for every annotated images we gathered the values
for all the models.

```
Figure 9: Performance Comparison of the Three Models
```

Summing it all 320 annotated images and following the step 1, which help us to analysis the IOU and
computing the sum of all IOU for the models and steps 2, computes accuracy with the same approach
as for figure 7; finally, we conclude our result as: Yolo is much better than others as compare with
parameter IOU and Accuracy.

## 4 Additional Work

In additional work, we perform segmentation and classification on objects that are predicted by the
best model based on the model’s performance. In the evaluation section, we selected yolo as the best
model. By considering the output of yolo, we segment the output images using OpenCV. Then we
obtain individual objects. Later, we perform binary classification using keras.io to classify the objects
as a cow and other. Through this classification, we build cow dataset for future implementation.

## 5 Conclusion and Future work

We considered pretrained models such as YOLO, SSD, and Mask RCNN for object detection. Then we
compared the output of three models with ground truth, obtained through manual annotation. While
we process those models with 320 images, we calculate the time and accuracy of the three models. SSD
has predicted 101 objects in 16.15 sec, YOLO has predicted 200 objects in 45.38 sec and Mask RCNN
predicted 862 objects in 20711.7 sec. From above all observations, Mask RCNN requires a longer time
compared to others, SSD is fast but performs worse for low-resolution images, and YOLO performs
more accurate by taking 0.226 seconds per object. So, we conclude that Mask RCNN and SSD are
good on average but cannot outperform the YOLO in terms of Accuracy (using IOU), but the use of
low-resolution feature maps diminishes accuracy. In the future, our project may upgrade to work on
images with any kind of resolution and if we train the YOLO with our classified cow dataset or any
other required dataset, we can get good accuracy.

## References

[1] https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-
python/

[2] https://www.cnblogs.com/jins-note/p/9952737.html

[3] https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

[4] https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-
492dc

[5] https://darkpgmr.tistory.com/

[6] https://www.learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/

[7] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick.
Microsoft COCO: Com- mon objects in context. In ECCV, 2014. 2, 5

[8] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In
CVPR, 2015. 1, 3, 6

[9] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with
region proposal networks. In NIPS, 2015. 1, 2, 3, 4, 7

[10] R. Girshick. Fast R-CNN. In ICCV, 2015. 1, 2, 3, 4, 6

[11] https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

[12] https://medium.com/inveterate-learner/real-time-object-detection-part-1-understanding-ssd-
65797a5e675b/

[13] https://www.groundai.com/project/dual-refinement-network-for-single-shot-object-detection/

[14] Image Processing, Analysis and Machine Vision by Milan Sonka PhD, Vaclav Hlavac PhD and
Roger Boyle DPhil, MBCS, CEng
