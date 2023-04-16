## FE-GAN Features
* Provides competitive performance for underwater image enhancement  
* Provides good inference speed: 200+ FPS on NVIDIA GeForce 2060
* Improves the accuracy of downstream tasks such as underwater object detection

## Installation
> python=3.8  
> pytorch=1.9.1  
> yaml=0.2.5  
> numpy=1.23.2  
> pillow=8.4.0  
> pandas=1.3.4

## Datasets
* EUVP dataset
  * EUVP dataset: [https://irvlab.cs.umn.edu/resources/euvp-dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset)
  * Paper: [Fast Underwater Image Enhancement for Improved Visual Perception](https://ieeexplore.ieee.org/document/9001231)
  * Code: [FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN)
* UGAN dataset
  * UGAN dataset: [https://irvlab.cs.umn.edu/resources](http://irvlab.cs.umn.edu/resources/)
  * Paper: [Enhancing Underwater Imagery Using Generative Adversarial Networks](https://arxiv.org/pdf/1801.04011.pdf)
  * Code: [Underwater-Color-Correction](https://github.com/cameronfabbri/Underwater-Color-Correction)
* UIEB dataset
  * UIEB dataset: [https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)
  * Paper: [An Underwater Image Enhancement Benchmark Dataset and Beyond](https://ieeexplore.ieee.org/abstract/document/8917818/)
  * Code: [Water-Net_Code](https://github.com/Li-Chongyi/Water-Net_Code)
* NYU-v2 dataset
  * NYU-v2 dataset:
    > [Type-I](https://pan.baidu.com/s/13k3qNGG93aFwdthjRtxi3Q)
    > [Type-IA](https://pan.baidu.com/s/13lRAbZYyYLyb-Z8qcpW-4Q)
    > [Type-IB](https://pan.baidu.com/s/12qXACo20C6ee9bViItZAFA)
    > [Type-II](https://pan.baidu.com/s/1iZM9md_mdeHXqw3XchvKHg)
    > [Type-III](https://pan.baidu.com/s/1fIKVcvU6jg5Mi0Sw-k8VmA)
    > [Type-1](https://pan.baidu.com/s/1V10iXd9QnFbevm17Ua0jwQ)
    > [Type-3](https://pan.baidu.com/s/1DEI4T700jmU-cUYgAxRQAw)
    > [Type-5](https://pan.baidu.com/s/1jlPodNRPqySGnFxr7-qRRg)
    > [Type-7](https://pan.baidu.com/s/12l0gCsPYOtEx7hCvp9C-fw)
    > [Type-9](https://pan.baidu.com/s/1IPKimxXA1CsX3wjRE4VYNQ)
  * Paper: [Underwater scene prior inspired deep underwater image and video enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0031320319303401)

## Usage
* Download the datasets, setup data-paths in the **config** files
* Use the **Enhancement_train.py** file to train the model
* Use the **Enhancement_test.py** file to evaluate the model
* Pretrained_model is provided, please click [here](https://drive.google.com/drive/folders/1PmTX1_W6_7pFo-vAF0M5h-5B83MSVdGU) to download, then put the file in **"./Pytorch/pretrain_models"**
* The evaluation code is provided in **Evaluation.py** file

## Some results
* EUVP dataset
![image](https://github.com/Jhan960102/FE-GAN-master/blob/main/FE-GAN/imgs/EUVP-2.PNG)

## Citation
If you find the code and datasets helpful in your resarch or work, please cite the following papers:
```
 @article{islam2019fast,
     title={Fast Underwater Image Enhancement for Improved Visual Perception},
     author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
     journal={IEEE Robotics and Automation Letters (RA-L)},
     volume={5},
     number={2},
     pages={3227--3234},
     year={2020},
     publisher={IEEE}
 }
 @article{li2019underwater,
  title={An underwater image enhancement benchmark dataset and beyond},
  author={Li, Chongyi and Guo, Chunle and Ren, Wenqi and Cong, Runmin and Hou, Junhui and Kwong, Sam and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4376--4389},
  year={2019},
  publisher={IEEE}
}
@inproceedings{fabbri2018enhancing,
  title={Enhancing underwater imagery using generative adversarial networks},
  author={Fabbri, Cameron and Islam, Md Jahidul and Sattar, Junaed},
  booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7159--7165},
  year={2018},
  organization={IEEE}
}
@article{li2020underwater,
  title={Underwater scene prior inspired deep underwater image and video enhancement},
  author={Li, Chongyi and Anwar, Saeed and Porikli, Fatih},
  journal={Pattern Recognition},
  volume={98},
  pages={107038},
  year={2020},
  publisher={Elsevier}
}
```
