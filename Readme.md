# CEUSegNet
IEEE ISBI 2022 paper: [CEUSegNet: A Cross-Modality Lesion Segmentation Network for Contrast-Enhanced Ultrasound](https://ieeexplore.ieee.org/abstract/document/9761594), a cooperation achievement by researchers from Institute of Automation CAS and Lanzhou University Second Hospital.

![overview](https://user-images.githubusercontent.com/57392333/176909137-4ea310ab-7e16-4ae4-b7bc-1d26620ef496.jpg)

## Highlights
- Contrast-Enhanced Ultrasound (CEUS) usually presents two modalities on video frames at the same time, i.e. ultrasound part and contrast-enhanced part.
- We can determine a rough location for lesion on ultrasound part and then finely sketch the region of interest on contrast-enhanced part.
- In this way, a video segmentation task can be converted into a frame segmentation task.

## Demos
![result](https://user-images.githubusercontent.com/57392333/176909977-20e9755b-1fe8-4b88-a278-1a635d1f3779.jpg)

Our work can achieve a comparable performance with clinicians on breast lesion and cervical lymphadenopathy segmentation task. More details can refer to our paper.

## Inference-time
| Input size | Time (ms) | MACs(G) | Params(M) |
| :--------: | :--: | :---: | :----: |
|128 * 128   | 20.82±2.62 | 12.74 | 9.281 |
|375 * 375 (origin)   | 68.51±0.41 | 108.51 | 9.281 |

## Acknowledgement
Our code is based on [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). Thanks!
