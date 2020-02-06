# YOLOv3-PyTorch

**[Ayoosh Kathuria](https://blog.paperspace.com/tag/series-yolo/) has an amazing tutorial that guides you in coding up a YOLO v3 detection model in PyTorch from pretty much scratch, using only the official YOLO v3 config file and weights.** <br>

As part of a [revisit to an older vehicle-detection project](https://github.com/Anner-deJong/CarND-Vehicle-Detection-DL-approach), I went through all of the tutorial and created this repository that can also be used stand-alone or in any other context than the vehicle-detection project. <br>

I did this as an exercise for myself, so simply taking over the code line by line wouldn't make sense. I mostly read the tutorial in terms of theory and descriptions, and limited my exposure to the tutorial's code. <br>
As such, this repository has the same structure as the original, and performs the same preprocessing, inference and post-processing on an input image. <br>

At the same time however, many functions are implemented with either different syntax, taking/returning differently stored data, or simply following a different line of reasoning. Some functions might be better (I actually made a few small pull requests), some might be worse. I did not compare the two final results in terms of e.g. inference time.

## Set up:
* This repository was made with Pytorch `torch=0.4.1`, `matplotlib=3.0.0` and `opencv=3.4.2`
* Before you can run the Yolov3 network you need to replace the weights placeholder inside `/Yolov3/cfg_weights_utils/` with the actual weights. Download them directly [here](https://pjreddie.com/media/files/yolov3.weights) or get them yourself from the [official website](https://pjreddie.com/darknet/yolo/).

## Use case examples
There are two notebooks listed below to help you get on your way. Two remarks first before starting out:
* The implementation does **not** support training, but only inference (the model loads in pre-trained weights).
* The wrapper class YOLOv3() is a very high abstraction and somewhat limited <br>

**1. Yolov3_as_repo_example.ipynb** <br>
If you want an example how the Yolov3 folder functions as an abstract imported class for inference, with minimal required code.

**2. Yolov3_notebook.ipynb** <br>
If you want to understand the code behind the model, or play with/tweak its functions in a notebook environment. It has all the same building blocks as the Yolov3/darknet.py file. (Which the notebook thus no longer needs to import. It still does requires the entire Yolov3/cfg_weights_utils folder).


## Room for improvement

* Some parts of the original repository are not included, most significantly a command line tooling including argument parsing
* Image resolution is hardcoded
* Detection rendering:
    * Bounding boxes arent clipped
    * Detected class text description sizes are hardcoded
* There is a whole bunch of different optimizations that could still be applied (especially the end was a bit rushed-through):
    * The Non Maximum Suppression function; IoU can be done in parallel as a matrix
    * The image loading and (pre)processing pipeline; currently no batches
