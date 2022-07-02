# Local Augmentation for Graph Neural Networks

This repository contains an implementation of ["Local Augmentation for Graph Neural Networks"](https://arxiv.org/pdf/2109.03856.pdf).

## Dependencies
- CUDA 10.2.89
- python 3.6.8
- pytorch 1.9.0
- pyg 2.0.3

## Usage
- For semi-supervised setting, run the following script
```sh
cd Citation
bash semi.sh
```

- For full-supervised setting, run the following script
```sh
cd OGB
# If you want to pre-train the generative model, run the following command:
python cvae_generate_products.py --latent_size 10 --pretrain_lr 1e-5 --total_iterations 10000 --batch_size 8192
# Train downstream GNNs
bash full.sh
```

## Citation
```
@inproceedings{liu2021local,
  title={Local augmentation for graph neural networks},
  author={Liu, Songtao and Ying, Rex and Dong, Hanze and Li, Lanqing and Xu, Tingyang and Rong, Yu and Zhao, Peilin and Huang, Junzhou and Wu, Dinghao},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
