
# SimMTM (NeurIPS 2023)

This is the codebase for the paper: [SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling](https://arxiv.org/abs/2302.00861)


## Architecture

<p align="center">
<img src=".\figs\overview.png" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of SimMTM.
</p>

The reconstruction process of SimMTM involves the following four modules: masking, representation learning, series-wise similarity learning and point-wise reconstruction.

### Masking

We can easily generate a set of masked series for each sample by randomly masking a portion of time points along the temporal dimension.

### Representation Learning

After the encoder and projector layer, we can obtain the point-wise representations and series-wise representations.

### Series-wise Similarity Learning

To precisely reconstruct the original time series, we attempt to utilize the similarities among series-wise representations for weighted aggregation, namely exploiting the local structure of the time series manifold.

### Point-wise Reconstruction

Based on the learned series-wise similarities, we aggregate the point-wise representation of its own masked series and other series to reconstruct the original time series.


## Get Started

1、Prepare Data. 

All benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a238e34ff81a42878d50/?dl=1), and arrange the folder as:

```plain
SimMTM/
|--SimMTM_Forecast
    |-- dataset/
        |-- ETT-small/
            |-- ETTh1.csv
            |-- ETTh2.csv
            |-- ETTm1.csv
            |-- ETTm2.csv
        |-- weather/
            |-- weather.csv
        |-- ...
    |-- ...
|--SimMTM_Class
    |-- dataset/
        |-- SleepEEG/
            |-- train.pt
            |-- val.pt
            |-- test.pt
        |-- FD-B/
            |-- ...
        |-- EMG/
            |-- ...
    |-- ...
```

2、Forecasting

We provide the forecasting experiment coding in `./SimMTM_Forecast` and experiment scripts can be found under the folder `./scripts`. To run the code on ETTh2, just run the following command: 

```bash
cd ./SimMTM_Forecast
# pre-training
sh ./scripts/pretrain/ETT_script/ETTh2.sh
# fine-tuning
sh ./scripts/finetune/ETT_script/ETTh2.sh
```

3、Classification

We also provide the classification experiment coding in `./SimMTM_Class`. When we want to pre-train a model on SleepEEG and fine-tune it on Epilepsy, please run:

```bash
cd ./SimMTM_Class
python ./code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy 
```

4、We also provide some [checkpoints](https://cloud.tsinghua.edu.cn/f/466995bb5f924f55a6da/?dl=1) and you can tune them directly on target datasets.

## Main Results

<p align="center">
<img src=".\figs\mainresult.png" alt="" align=center />
<br><br>
</p>

SimMTM (marked by red stars) can simultaneously cover high-level and low-level tasks for in- and cross-domain settings and outperforms other baselines significantly, highlighting the advantages of SimMTM in task generality. More results can be found in our paper.

## Citation
If you find this repo useful, please cite our paper.

```plain
@inproceedings{dong2023simmtm,
  title={SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling},
  author={Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Contact

If you have any questions, please contact [djx20@mails.tsinghua.edu.cn](mailto:djx20@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Time-Series-Library

https://github.com/mims-harvard/TFC-pretraining/tree/main

Thanks to [vincentsham](https://github.com/vincentsham/simmtm/blob/main/experiments_simmtm-BeijingPM25Quality.ipynb) for reproducing our code.