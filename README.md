# pytorch-EDI(event based deblur method)

A simple implementation of the paper "Bringing a Blurry Frame Alive at High Frame-Rate with an Event Camera"(2019) by Pan et al.

## requirements
 pytorch
 
 numpy 
 
 opencv-python 
 
## usage

### data preparation
Given raw event data,run  `python process.py` to convert event into event frames shaped (T, H, W).

### optimization

Run `train.py` to estimate the constant c of event camera.

## result
A nice estimate for constant c is about 0.34 specifically for the example data.

Visualization of the deblurred result of the example data.

