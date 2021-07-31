# MFNet
source code for "MFNet: Multi-filter Directive Network for Weakly Supervised Salient Object Detection", accepted in ICCV-2021, poster.

Yongri Piao, Jian Wang, Miao Zhang and Huchuan Lu.  IIAU-OIP Lab.

![image](https://github.com/DUTyimmy/MFNet/blob/main/fig/overall%20framework.png)

## Prerequisites
  - Windows 10
  - Torch 1.8.1
  - CUDA 10.0
  - Python 3.7.4
  - other Prerequisites can be found in requirments.txt 

## Training data
link: https://pan.baidu.com/s/1omTCChQFWwNFhQ79AVD8rg code:oipw

## Testing data


## 1st training stage (pseudo labels)
please refer to :

We also upload ready-made pseudo labels in **Training data**, you can directly use our offered two kinds of pseudo labels for convenience. CAMs are also presented if you needed.

## 2nd training stage

### 1, Setting the training data to the proper root:

```
MF_code -- data -- DUTS-Train -- image -- 10553 samples

                -- ECSSD (not necessary) 
                
                -- pseudo labels -- label0_0 -- 10553 pseudo labels
                
                                 -- label1_0 -- 10553 pseudo labels
```
