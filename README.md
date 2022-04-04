# *GMCNet:* Robust Partial-to-Partial Point Cloud Registration in a Full Range
<p align="center"> 
<img src="images/intro.png">
</p>

## [GMCNet]
This repository contains the PyTorch implementation of the paper:

**Robust Partial-to-Partial Point Cloud Registration in a Full Range**

[[arxiv](https://arxiv.org/abs/2111.15606)]

> Point cloud registration for 3D objects is very challenging due to sparse and noisy measurements, incomplete observations and large transformations. In this work, we propose Graph Matching Consensus Network (GMCNet), which estimates pose-invariant correspondences for fullrange 1 Partial-to-Partial point cloud Registration (PPR). To encode robust point descriptors, **1)** we first comprehensively investigate transformation-robustness and noiseresilience of various geometric features. **2)** Then, we employ a novel Transformation-robust Point Transformer (TPT) modules to adaptively aggregate local features regarding the structural relations, which takes advantage from both handcrafted rotation-invariant (RI) features and noise-resilient spatial coordinates. **3)** Based on a synergy of hierarchical graph networks and graphical modeling, we propose the Hierarchical Graphical Modeling (HGM) architecture to encode robust descriptors consisting of i) a unary term learned from RI features; and ii) multiple smoothness terms encoded from neighboring point relations at different scales through our TPT modules. Moreover, we construct a challenging PPR dataset (MVP-RG) with virtual scans. Extensive experiments show that GMCNet outperforms previous state-of-the-art methods for PPR. Notably, GMCNet encodes point descriptors for each point cloud individually without using crosscontextual information, or ground truth correspondences for training.

<p align="center"> 
<img src="images/hgm.png">
</p>

### Installation
Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), and then use the following command:
```
git clone https://github.com/paul007pl/GMCNet
cd GMCNet; source setup.sh;
```
If your connection to conda and/or pip is unstable, it is recommended to follow the setup steps in `setup.sh`.


### Data
Please download our prepared data ([Dropbox](https://www.dropbox.com/sh/tdfs406baoyugda/AADe8GV3w7CaORUDO6nCnRSra?dl=0) to the folder `data`.


### Usage
+ To train a model: run `python train.py -c *.yaml`, e.g. `python train.py -c pcn.yaml`
+ To test a model: run `python test.py -c *.yaml`, e.g. `python test.py -c pcn.yaml`
+ Config for each algorithm can be found in `cfgs/`.
+ `run_train.sh` and `run_test.sh` are provided for SLURM users. 


## [License]
Our code is released under Apache-2.0 License.


## [Acknowledgement]
