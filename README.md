**Read me:**

**1.** Build a RTSP streaming module, which can take the following links rtsp://[freja.hiof.no:1935/rtplive/\_definst\_/hessdalen03.stream](http://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream)

or

rtsp://[wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny\_115k.mov](http://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov)

or any suitable web url and iterate it on the front end into the multiple of n.

- Tocomplete this task we have used Open CV as it is not possible to directly render RTSP url in HTML tags.
- I can across [https://stackoverflow.com/questions/1735933/streaming-via-rtsp-or-rtp-in-html5](https://stackoverflow.com/questions/1735933/streaming-via-rtsp-or-rtp-in-html5) where it says that we can use a Javascript library of some sort (unless you want to get into playing things with flash and silverlight players) to play streaming video. {IMHO} and many other options are also available like ffmpeg, but I have not tried it because of time limitation. Even I am wondering what could be the optimal solution.

**2.** That is if the number we enter is 4, the UI should display as below ;(Also Attached)

![](RackMultipart20210514-4-kyikin_html_6f443dba342df600.png)

- I have created one UI as per the provided picture that include
  - search bar
  - navigation bar i.e. menu
  - content box i.e. RTSP view

- We can render the number of camera view with the help of search bar.

![UI](https://github.com/kamini019/RTSP/1.png)

**3**. Dockerize the project as well as expose it to localhost.

- To be honest, I don&#39;t recommend using public configured Docker system instead of it we can have a official image of the software and then install all the required packages and then use it.
- But for time sake, I have tries to use Open-CV and alpine os which created by some other guy and I am facing version issues.
- I have created Docker file where I am using the python:3.10.0a1-alpine3.12 instead of Linux, show that our application will be light weight.

**4.** Store the information in mongo or on postgres.

- I have created a cloud MongoDB database, we can using connect mongocompass

mongodb+srv://admin:admin@cluster0.vdibn.mongodb.net/test

- I have created a table named as RTSP and storing all the input value.
- I have used djongo to connect to mongoDB because I wanted to use Django ORM.

**5.** Added advantage would be using any publicly available ML models such as yolo for object detection on the live cameras.

- I have added yolo coco model, and I have created yolo help class.

- Its takes lots of time to process the data, So, I have commented it out. File : rtsp\_opencv.py line number 23