# Learning Adaptive Classifiers Synthesis for Generalized Few-Shot Learning
The code repository for ["Learning Adaptive Classifiers Synthesis for Generalized Few-Shot Learning"](https://arxiv.org/abs/1906.02944)

```
@article{DBLP:YeHZS2019Learning,
      author    = {Han-Jia Ye and
                   Hexiang Hu and
                   De-Chuan Zhan},
      title     = {Learning Adaptive Classifiers Synthesis for Generalized Few-Shot Learning},
      journal   = {CoRR},
      volume    = {abs/1906.02944},
      year      = {2019}
    }
```

## Generalized Few-Shot Learning
Object recognition in the real-world requires handling long-tailed or even open-ended data. An ideal visual system needs to recognize the populated head visual concepts reliably and meanwhile efficiently learn about emerging new tail categories with a few training instances. Class-balanced many-shot learning and few-shot learning tackle one side of this problem, by either learning strong classifiers for head or learning to learn few-shot classifiers for the tail. We investigate the problem of generalized few-shot learning (GFSL) --- a model during the deployment is required to learn about tail categories with few shots and simultaneously classify the head classes.

<img src='imgs/gfsl.png' width='640' height='200'>

## Adaptive Classifiers Synthesis
We propose the ClAssifier SynThesis LEarning (Castle), a learning framework that learns how to synthesize calibrated few-shot classifiers in addition to the multi-class classifiers of head classes with a shared neural dictionary, shedding light upon the inductive GFSL. Furthermore, we propose an adaptive version of Castle (ACastle) that adapts the head classifiers conditioned on the incoming tail training examples, yielding a framework that allows effective backward knowledge transfer. As a consequence, ACastle can handle GFSL with classes from heterogeneous domains.

<img src='imgs/architecture.png' width='640' height='200'>

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images into the folder data/[name of the dataset, miniimagenet or cub]/images

- Pre-trained weights: please download the [pre-trained weights](https://drive.google.com/open?id=14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE) of the encoder if needed. The pre-trained weights can be downloaded in a [zip file](https://drive.google.com/file/d/1XcUZMNTQ-79_2AkNG3E04zh6bDYnPAMY/view?usp=sharing).

## Dataset

### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation, respectively. The auxiliary images for evaluating SEEN categories could be downloaded from [here](https://drive.google.com/drive/folders/1gk0AsYUrN9NtLIDE9AKaZeVYzftpo6Ol?usp=sharing). 

### TieredImageNet Dataset
[TieredImageNet](https://github.com/renmengye/few-shot-ssl-public) is a large-scale dataset  with more categories, which contains 351, 97, and 160 categoriesfor model training, validation, and evaluation, respectively. The dataset can also be download from [here](https://github.com/kjunelee/MetaOptNet).
We only test TieredImageNet with ResNet backbone in our work. The auxiliary images for evaluating SEEN categories could be downloaded from [here](https://drive.google.com/drive/folders/19SGTIAnTmVohdSNN6Q9G5Qn3bVChOIMr?usp=sharing).

## Acknowledgment
We thank following repos providing helpful components/functions in our work.
- [FEAT](https://github.com/Sha-Lab/FEAT)
