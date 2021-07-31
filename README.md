# MFNet
Source code for "MFNet: Multi-filter Directive Network for Weakly Supervised Salient Object Detection", accepted in ICCV-2021, poster.

Yongri Piao, Jian Wang, Miao Zhang and Huchuan Lu.  IIAU-OIP Lab.

![image](https://github.com/DUTyimmy/MFNet/blob/main/fig/overall%20framework.png)

## Prerequisites
### environment
  - Windows 10
  - Torch 1.8.1
  - CUDA 10.0
  - Python 3.7.4
  - other Prerequisites can be found in requirments.txt 

### training data
link: https://pan.baidu.com/s/1omTCChQFWwNFhQ79AVD8rg.    code: oipw

### testing datasets
link: https://pan.baidu.com/s/1PBzDP1Hnf3RIvpARmxn2yA.    code: oipw

## Training
### 1st training stage
Case1: please refer to :

Case2: We also upload ready-made pseudo labels in **Training data**, you can directly use our offered two kinds of pseudo labels for convenience. CAMs are also presented if you needed.

### 2nd training stage

#### 1, setting the training data to the proper root as follows:

```
MF_code -- data -- DUTS-Train -- image -- 10553 samples

                -- ECSSD (not necessary) 
                
                -- pseudo labels -- label0_0 -- 10553 pseudo labels
                
                                 -- label1_0 -- 10553 pseudo labels
```
#### 2, training
```Run main.py```

## Testing
```Run test_code.py```

You need to configure your desired testset in ```--test_root```.  Here you can also perform PAMR and CRF on saliency maps for a furthur refinements if you want, by setting ```--pamr``` and ```--crf``` to True. Noting that the results in our paper do not adopt these post-process for a fair comparison.

## Contact us
If you have any questions, please contact us [jiangnanyimi@163.com].

## Acknowledge
Thanks to pioneering works:

  - [IRNet](https://github.com/jiwoon-ahn/irn): Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, CVPR2019.
  - [MSW](https://github.com/zengxianyu/mws/tree/new) Multi-source weak supervision for saliency detection, CVPR2019
