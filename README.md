# Human-Activity-Extractor
High level python script that looks at a video or folder of videos and for each video,outputs a single video containing only segments containing people. The main motivation behind this was to be able to remove large amounts of non-useful video from CCTV camera footage.

This project was created as a result of a burglary at a close relative's house. Full story [here](https://www.linkedin.com/posts/shailesh-sridhar-441baa156_tldr-there-was-a-burglary-at-a-close-relatives-activity-6798319156354789376-w1hV)

Huge thanks to humandecoded, whose repository (https://github.com/humandecoded/People-Detector) provided the base structure and code.

![ezgif com-gif-maker(1)](https://user-images.githubusercontent.com/29563101/117713270-11e01700-b1f3-11eb-9572-c1ec582666d3.gif)

The source video for the above demo gif was taken from the public [DCSASS dataset](https://www.kaggle.com/mateohervas/dcsass-dataset)

 

## Requirements 
* First, activate your Python 3.7 virtual env.  Then:
```
pip install --upgrade pip
pip install tensorflow
pip install cvlib
pip install opencv-python
pip instal tqdm
```

When you first run this script it will reach out and download the pre-trained YOLO model as well.

After that it's as simple as:
* `python extract-human-segments.py -d <path to folder>`
* or
* `python extract-human-segments.py -f <path to file>`

There are a number of optional flags outlined below.

The default ML model is 'yolov4'. This model is big and CPU intensive. The `--tiny_yolo` flag will give you a smaller model that is faster but less accurate.
In my experience tiny yolo is generally adequate,as the code pads all human containing segments with a small amount of extra video, generally making up for any missed detections. Experimentation with the user's specific conditions is advised, however.


The `--frames n`  flag sets the program to examine every `n`th frame. The default is every 10th frame.

The `--confidence n`  flag will adjust the confidence threshold for YOLO that trips a detection alert to `n`. The default is 65.

The `--gpu` flag tells the code whether to use the gpu configured on the system. If no gpu is found, falls back to cpu.

The `--time_padding` flag sets the number of seconds of video to be added before and after a human detected segment.Default is 10

The `--draw_boxes` flag determines whether the bounding boxes for the detections are to be included in the processed video



When run, the script creates a folder `../output`, containing a folder or each video analyzed. Each such folder contains a 'processed_video.avi' , which  contains all human containing segments identified within the video, and a 'snapshots' folder, containing snapshots of frames(.jpeg format) where humans were found, along with a bounding box and confidence score. 











