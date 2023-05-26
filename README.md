# dtac-dev
The is the repo of **D**istributed **T**ask-**a**ware **C**ompression (dtac). 

Link to paper: [Task-aware Distributed Source Coding under Dynamic Bandwidth](https://arxiv.org/abs/2305.15523)
## Table of Contents
- [TLDR](#TLDR)
- [Results](#results)
- [Installation](#installation)
  - [Packages](#packages)
  - [Dataset](#dataset)
- [Usage](#usage)
- [Citation](#citation)

## TLDR
We design a distributed compression framework which learns low-rank task representations and efficiently distributes bandwidth among sensors to provide a trade-off between performance and bandwidth.

## Abstract
<details>
<summary>Click to expand</summary>
Efficient compression of correlated data is essential to minimize communication overload in multi-sensor networks. In such networks, each sensor independently compresses the data and transmits them to a central node due to limited communication bandwidth. A decoder at the central node decompresses and passes the data to a pre-trained machine learning-based task to generate the final output. Thus, it is important to compress features that are relevant to the task. Additionally, the final performance depends heavily on the total available bandwidth. In practice, it is common to encounter varying availability in bandwidth, and higher bandwidth results in better performance of the task. We design a novel distributed compression framework composed of independent encoders and a joint decoder, which we call neural distributed principal component analysis (NDPCA). NDPCA flexibly compresses data from multiple sources to any available bandwidth with a single model, reducing computing and storage overhead. NDPCA achieves this by learning low-rank task representations and efficiently distributing bandwidth among sensors, thus providing a graceful trade-off between performance and bandwidth. Experiments show that NDPCA improves the success rate of multi-view robotic arm manipulation by 9% and the accuracy of object detection tasks on satellite imagery by 14% compared to an autoencoder with uniform bandwidth allocation.
</details>

## Results
![results](./plots/results.png "results")
**Top:** Performance Comparison for 3 different tasks. Our method achieves equal or higher performance than other methods. 
**Bottom:** Distribution of total available bandwidth (latent space) among the two views for NDPCA (ours). The unequal allocation highlights the difference in the importance of the views for a given task.

## Installation
### Packages
For the installation of the required packages, see the [setup.py](setup.py) file or simply run the following command to install the required packages in [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```

Then to activate the dtac environment, run:
```bash
pip install -e .
```

### Dataset
#### Locate and lift
The locate and lift experiment needs the gym package (see [setup.py](setup.py) or [requirements.txt](requirements.txt)) and mujoco. To install mujoco, see [install mujoco](https://github.com/openai/mujoco-py). \
To collect the demonstration dataset used for training of RL agent and the autoencoders, run the following command:
```bash
python collect_lift_hardcode.py
```

#### Airbus
The airbus experiment needs the airbus dataset. To download the dataset, see [Airbus Aircraft Detection](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset). \
After downloading the dataset, place the dataset in the "./airbus_dataset
 folder and run "./airbus_scripts/aircraft-detection-with-yolov8.ipynb". \
Then, put the output of the notebook in the following folder
"./airbus_dataset/224x224_overlap28_percent0.3_/train" and "./airbus_dataset/224x224_overlap28_percent0.3_/val". \
Finally, augment the dataset with mosaic at "./mosaic/":
```bash
python main.py --width 224 --height 224 --scale_x 0.4 --scale_y 0.6 --min_area 500 --min_vi 0.3 --path
```

## Usage

### dtac package
The dtac package contains the following models:
* ClassDAE.py: Class of Autoencoders
* DPCA_torch.py: Fuctions of DPCA

and other common utility functions.

### Locate and lift
To train an RL agent, run the following command:
```bash
python train_behavior_cloning_lift.py -v
```
where -v is the views of the agent: "side", "arm", or "2image".

To train the lift and locate NDPCA, run the following command:
```bash
python train_awaDAE.py -args
```
See the -args examples in the main function of [train_awaDAE.py](PnP_scripts/train_awaDAE.py) file.

To evaluate autoencoder models, run the following command in the "./PnP_scripts" folder:
```bash
python eval_DAE.py -args
```
See the -args examples in the main function of [eval_DAE.py](PnP_scripts/eval_DAE.py) file.

### Airbus
To train the object detection (Yolo) model, run the following command:
```bash
python airbus_scripts/yolov1_train_detector.py
```

To train the Airbus NDPCA, run the following command in the "./airbus_scripts" folder:
```bash
python train_od_awaAE.py -args
```
See the -args examples in the main function of [train_od_awaAE.py](airbus_scripts/train_od_awaAE.py) file.

To evaluate autoencoder models, run the following command in the "./airbus_scripts" folder:
```bash
python dpca_od_awaDAE.py -args
```
See the -args examples in the main function of [dpca_od_awaDAE.py](airbus_scripts/dpca_od_awaDAE.py) file.

## Citation
If you find this repo useful, please cite our paper:
```
@misc{li2023taskaware,
      title={Task-aware Distributed Source Coding under Dynamic Bandwidth}, 
      author={Po-han Li and Sravan Kumar Ankireddy and Ruihan Zhao and Hossein Nourkhiz Mahjoub and Ehsan Moradi-Pari and Ufuk Topcu and Sandeep Chinchali and Hyeji Kim},
      year={2023},
      eprint={2305.15523},
      archivePrefix={arXiv},
      primaryClass={cs.IT}
}
```
