# pytorch-EDI(event based deblur method)

A simple pytorch implementation of the paper "[Bringing a Blurry Frame Alive at High Frame-Rate with an Event Camera](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Bringing_a_Blurry_Frame_Alive_at_High_Frame-Rate_With_an_CVPR_2019_paper.pdf)"(2019) by Pan et al.

official repository is [here](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera)
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
A nice estimate for constant c is about 0.34 specifically for the [example data](https://github.com/XiangZ-0/EVDI).

Visualization of the deblurred result of the example data.

![blurred](https://github.com/lbh666/pytorch-EDI/blob/main/blur1.png) ![result](https://github.com/lbh666/pytorch-EDI/blob/main/result.png)
![blurred](https://github.com/lbh666/pytorch-EDI/blob/main/blur2.png) ![result](https://github.com/lbh666/pytorch-EDI/blob/main/result2.png)
![](https://github.com/lbh666/pytorch-EDI/blob/main/cube.gif)
