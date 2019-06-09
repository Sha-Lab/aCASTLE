# CASTLE
The code repository for ["Learning Classifier Synthesis for Generalized Few-shot Learning"]()

### Prerequisites

The following packages are required to run the scripts:

- [PyTorch-0.4 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

### Dataset

#### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation respectively.

## .bib citation
If this repo helps in your work, please cite the following paper:

```
    @article{DBLP:YeHZS2019Learning,
      author    = {Han-Jia Ye and
                   Hexiang Hu and
                   De-Chuan Zhan and
                   Fei Sha},
      title     = {Learning Classifier Synthesis for Generalized Few-Shot Learning},
      journal   = {CoRR},
      volume    = {abs/xxxx.xxxx},
      year      = {2019}
    }
```

## Acknowledgment
We thank following repos providing helpful components/functions in our work.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)
- [MatchingNet](https://github.com/gitabcworld/MatchingNetworks)
- [PFA](https://github.com/joe-siyuan-qiao/FewShot-CVPR/)
