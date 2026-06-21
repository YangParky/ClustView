<<<<<<< HEAD
<!--
 * @Date: 2023-10-18 20:23:35
 * @Author: Xiao Xioayang
 * @LastEditors: Xiao Xioayang
 * @LastEditTime: 2026-06-21 13:58:24
-->

# ClustView
This repository contains the official code release of [**ClustView: Point Clustering and Depth View Fusion for Point Cloud Analysis**](https://www.sciencedirect.com/science/article/pii/S0957417426022116?dgcid=author) (ESWA 2026).


## Introduction
In this work, we propose ClustView, an end-to-end framework that effectively fuses point cloud data with multi-view features for enhanced 3D shape recognition. 

<img src="./figure/ClustView.png" align="center" width="100%">



## 1. Installation
```
# clone this repo
git clone https://github.com/YangParky/ClustView.git
cd ClustView

# create a conda env
conda create -n ClustView -y python=3.7 numpy=1.20 numba
conda activate ClustView

# install PyTorch and libs 
# please install compatible PyTorch and CUDA versions
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install glob, h5py, sklearn, einops, hydra-core=1.1, tqdm, warmup-scheduler, deepspeed

# install the pointnet++ library cuda extensions
pip install pointnet2_ops/.
```


## 2. Data Preparation
When you first run the command for training, the datasets will be automatically downloaded and saved in `data/`.
- **ModelNet40** -->`data/modelnet40_ply_hdf5_2048/`
- **ScanObjectNN** -->`data/h5_files/`


## 3. Training
### Step 1: Check config file 
You can modify settings in `config/cls.yaml`.

Make sure the `eval` is set to False.

> We support [wandb](https://wandb.ai/site) for collecting results online. Just set `wandb.use_wandb=True` if use wandb. 
> Please check the [official wandb doc](https://docs.wandb.ai/) for more details. 

### Step 2: Train ClustView
- Classification on ModelNet40

    ```
    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python main_cls.py data=modelnet40
    ```

- Classification on ScanObjectNN

    ```
    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python main_cls.py data=scanobjectnn
    ```

## 4. Evaluation
- To evaluate a trained-model, please set `eval=True` in `config/train.yaml` and run `python main.py data=${dataset}`
Or you can override values in the loaded config from the command line:
    
    ```
    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python main_cls.py data=modelnet40 eval=True
    ```

- You can evaluate by voting:

    ```
    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python vote_cls.py
    ```

## Visualization
### Dependency
- [Mitsuba](https://www.mitsuba-renderer.org/)
Please refer to the following github repository for point cloud rendering code: [PointFlowRenderer](https://github.com/zekunhao1995/PointFlowRenderer)


## Citation
If you entrust our work with value, please consider giving a star ⭐ and citation:
```
@article{xiao2026clustview,
  title={ClustView: Point Clustering and Depth View Fusion for Point Cloud Analysis},
  author={Xiao, Xiaoyang and Chen, Yuanbo and Yao, Runzhao and Tian, Zhiqiang and Jiang, Jue and Zheng, Xinhu and Guo, Long and Du, Shaoyi},
  journal={Expert Systems with Applications},
  pages={133302},
  year={2026},
  publisher={Elsevier}
}
```


## Acknowledgement
Our code is mainly based on the following open-source projects. Many thanks to the authors for their wonderful works.
[PointNet2](https://github.com/erikwijmans/Pointnet2_PyTorch),
[Point-Transformers](https://github.com/qq456cvb/Point-Transformers),
[DGCNN](https://github.com/AnTao97/dgcnn.pytorch), 
[CurveNet](https://github.com/tiangexiang/CurveNet), 
[PointMLP](https://github.com/ma-xu/pointMLP-pytorch), 
[PAConv](https://github.com/CVMI-Lab/PAConv), 
[PointNeXt](https://github.com/guochengqian/pointnext)
[PointCont](https://github.com/yahuiliu99/PointConT).

## License
This repository is released under MIT License.
=======
# ClustView
ClustView
>>>>>>> 211c2fe21f664684f68e84abd97db67f3f4c57c6
