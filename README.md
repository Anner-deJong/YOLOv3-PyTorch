# YOLOv3-PyTorch
# to be updated later

# copy but not a copy from:

# inference, no training
# lot of optimization still to be done

# Set up and prerequisites

# include weight downloader

# include example the other repo


# kind of rushing through in the end, so especially the Non Maximum Suppression function can be optimized (IoU can be done in parrallel as a matrix, img pipeline (currently no batches)

# wrapper class YOLOv3 is very high abstraction and somewhat limited class

# I did this as an exercise for myself, so simply transcribing all code wouldn't make sense. This repository has the same structure as the original, and performs the same preprocessing, inference and post-processing on an input image. However, many functions are implemented with either different syntax, taking/returning differently stored data, or simply follow a different line of reasoning. Some functions might be better (I actually made a few small pull requests), some might be worse. I did not implement timing in order to compare.

# some parts of the original repo are not included, most significantly a command line tooling including argument parsing etc
# bounding boxes arent clipped
# image text annotation sizes are hardcoded
# hardcoded img resolution