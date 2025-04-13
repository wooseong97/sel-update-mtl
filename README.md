# Selective Task Group Updates for Multi-Task Optimization [ICLR 2025]

**Official PyTorch implementation of [*Selective Task Group Updates for Multi-Task Optimization*](https://arxiv.org/abs/2502.11986) [ICLR 2025].**

Wooseong Jeong & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

![Selective Task Group Visualization](figures/overview.png) 

Multi-task learning enables the acquisition of task-generic knowledge by training
multiple tasks within a unified architecture. However, training all tasks together
in a single architecture can lead to performance degradation, known as negative
transfer, which is a main concern in multi-task learning. Previous works have
addressed this issue by optimizing the multi-task network through gradient manipulation or weighted loss adjustments. However, their optimization strategy
focuses on addressing task imbalance in shared parameters, neglecting the learning of task-specific parameters. As a result, they show limitations in mitigating
negative transfer, since the learning of shared space and task-specific information
influences each other during optimization. To address this, we propose a different
approach to enhance multi-task performance by selectively grouping tasks and
updating them for each batch during optimization. We introduce an algorithm that
adaptively determines how to effectively group tasks and update them during the
learning process. To track inter-task relations and optimize multi-task networks
simultaneously, we propose proximal inter-task affinity, which can be measured
during the optimization process. We provide a theoretical analysis on how dividing
tasks into multiple groups and updating them sequentially significantly affects
multi-task performance by enhancing the learning of task-specific parameters. Our
methods substantially outperform previous multi-task optimization approaches and
are scalable to different architectures and various numbers of tasks.


## 1. Dataset
We use the same datasets as [InvPT](https://github.com/prismformore/Multi-Task-Transformer/tree/main/InvPT): **NYUD-v2** and **PASCAL-Context**.

You can download them from the following links:

- [NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c)  
- [PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab)

After downloading, extract the datasets using:

```bash
tar xfvz NYUDv2.tar.gz
tar xfvz PASCALContext.tar.gz
```

## 2. Build environment

## 3. Train and Evaluate


## Installation



### Experimental Results on the Taskonomy Dataset using ViT-L
The best results in each column are shown in **bold**, while convergence failures are indicated with a dash.

| Task          | DE       | DZ       | EO         | ET         | K2         | K3   | N       | C       | R       | S2      | S2.5    | Δₘ (↑)             |
|---------------|----------|----------|------------|------------|------------|------|---------|---------|---------|---------|---------|--------------------|
| **Metric**    | L1 ↓     | L1 ↓     | L1 ↓       | L1 ↓       | L1 ↓       | L1 ↓ | L1 ↓    | RMSE ↓  | L1 ↓    | L1 ↓    | L1 ↓    |                    |
| Single Task   | 0.0155   | 0.0160   | 0.1012     | 0.1713     | 0.1620     | 0.082| 0.2169  | 0.7103  | 0.1357  | 0.1700  | 0.1435  | -                  |
| GD            | 0.0163   | 0.0167   | 0.1211     | 0.1742     | 0.1715     | 0.093| 0.2333  | 0.7527  | 0.1625  | 0.1837  | 0.1487  | -8.65 ± 0.229       |
| GradDrop      | 0.0168   | 0.0172   | 0.1229     | 0.1744     | 0.1727     | 0.091| 0.2562  | 0.7615  | 0.1656  | 0.1862  | 0.1511  | -10.81 ± 0.377      |
| MGDA          | -        | -        | -          | -          | -          | -    | -       | -       | -       | -       | -       | -                  |
| UW            | 0.0167   | 0.0151   | 0.1212     | 0.1728     | 0.1712     | 0.089| 0.2360  | 0.7471  | 0.1607  | 0.1829  | 0.1538  | -7.65 ± 0.087       |
| DTP           | 0.0169   | 0.0153   | 0.1213     | **0.1720** | 0.1707     | 0.089| 0.2517  | 0.7481  | 0.1603  | 0.1814  | 0.1503  | -8.16 ± 0.081       |
| DWA           | 0.0147   | 0.0155   | 0.1209     | 0.1725     | 0.1711     | 0.089| 0.2619  | 0.7486  | 0.1613  | 0.1845  | 0.1543  | -7.92 ± 0.077       |
| PCGrad        | 0.0161   | 0.0165   | 0.1206     | 0.1735     | 0.1696     | 0.090| 0.2301  | 0.7540  | 0.1625  | 0.1830  | 0.1483  | -7.72 ± 0.206       |
| CAGrad        | 0.0162   | 0.0166   | 0.1202     | 0.1769     | 0.1651     | 0.091| 0.2565  | 0.7653  | 0.1661  | 0.1861  | 0.1571  | -10.05 ± 0.346      |
| IMTL          | 0.0162   | 0.0165   | 0.1206     | 0.1741     | 0.1710     | 0.090| 0.2268  | 0.7497  | 0.1617  | 0.1832  | 0.1543  | -8.03 ± 0.179       |
| Aligned-MTL   | 0.0150   | 0.0155   | **0.1135** | 0.1725     | **0.1630** |**0.086**|0.2513| 0.8039  | 0.1646  | 0.1800  | **0.1438**| -6.22 ± 0.285       |
| Nash-MTL      | -        | -        | -          | -          | -          | -    | -       | -       | -       | -       | -       | -                  |
| FAMO          | 0.0157   | 0.0155   | 0.1211     | 0.1730     | 0.1702     | 0.090| 0.2433  | 0.7479  | 0.1610  | 0.1823  | 0.1527  | -7.58 ± 0.211       |
| **Ours**      | **0.0140** | **0.0145** | 0.1136 | 0.1735     | 0.1679     | 0.087| **0.2029**|**0.7166**|**0.1500**|**0.1769**|0.1469  | **-1.42 ± 0.208**   |


## Acknowledgements
This repository incorporates experimental settings and codes from the following prior works:

- **[InvPT: Inverted Pyramid Multi-task Transformer for Dense Scene Understanding](https://arxiv.org/abs/2203.07997)**  
  by Hanrong Ye and Dan Xu, *ECCV 2022*  
  GitHub: [https://github.com/prismformore/Multi-Task-Transformer](https://github.com/prismformore/Multi-Task-Transformer)

- **[MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning](https://arxiv.org/abs/2001.06902)**  
  by Simon Vandenhende, Stamatios Georgoulis, and Luc Van Gool, *ECCV 2020*  
  GitHub: [https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)

> We sincerely thank the authors for open-sourcing their codebases. 


## Contact
Wooseong Jeong: stk14570@kaist.ac.kr

## Citation
If you use this work in your research, please cite it as:

```bibtex
@article{jeong2025selective,
  title   = {Selective Task Group Updates for Multi-Task Optimization},
  author  = {Jeong, Wooseong and Yoon, Kuk-Jin},
  journal = {arXiv preprint arXiv:2502.11986},
  year    = {2025}
}


