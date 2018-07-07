## DVAE# : Discrete Variational Autoencoders with Relaxed Boltzmann Priors

DVAE# is the state-of-the-art deep learning framework for training deep generative models with Boltzmann priors. This
repository offers the Tensorflow implementation of DVAE# that can be used to reproduce all the results presented in 
the DVAE# paper (Table 1 and Table 2) for both binarized MNIST and OMNIGLOT. We have implemented the following
training frameworks in this repo:

1. [DVAE](https://arxiv.org/abs/1609.02200) [1]: The original discrete variational autoencoders (DVAE) proposed by 
Jason Rolfe. This framework introduces the spike-and-exponential smoothing for relaxing binary latent variables
and optimizes the variational lower bound in training.
2. [DVAE++](https://arxiv.org/abs/1802.04920) [2]: This paper introduces the overlapping smoothing transformations and shows that these transformations can
be used for training discrete variational autoencoder with a directed prior as well as an undirected prior. This repo
only contains undirected priors that were used in DVAE#.
3. [DVAE#](https://arxiv.org/abs/1805.07445) [3]: This paper generalizes overlapping transformations to distributions in which the inverse CDF
cannot be computed analytically. Moreover, it shows how a continuous relaxation of Boltzmann machines can be formed
using the generalized overlapping smoothings as well as the Gaussian integral trick. The continuous relaxations
permit training with importance-weighted bounds.

For sampling from Boltzmann priors, persistent contrastive divergence (PCD) and
population annealing (PA) algorithms are used in this repo. We rely on the
sampling library QuPA which was recently released by [Quadrant](http://quadrant.ai). 
You can have access to this library [here](https://try.quadrant.ai/qupa).

<br/>

## Running the Training/Evaluation Code
The main train/evaluation script can be run locally using the following command: 

```bash
python run.py \
    --log_dir=${PATH_TO_LOG_DIR} \
    --data_dir=${PATH_TO_DATA_DIR}
```

If you don't have the datasets locally, the scripts will download them automatically to the data directory.

The following flags are introduced in order to run the settings reported in Table 1 of DVAE#:
1. `--dataset` specifies the dataset used for the experiment. Currently, we support `omniglot` and `binarized_mnist`.
2. `--baseline` sets the type of objective function which is used for training. This corresponds to different columns 
in Table 1. You can use `dvae_spike_exp` for DVAE (spike and exp), `dvaepp_exp` for DVAE++ (exponential), 
`dvaepp_power` for DVAE++ (power), `dvaes_gi` for DVAE# (Gaussian Integral), `dvaes_gauss` for DVAE# (Gaussian), 
`dvaes_exp` for DVAE# (exponential), `dvaes_unexp` for DVAE# (uniform+exponential), `dvaes_power` for DVAE# (power).
3. `--struct` sets the structure used in the encoder and decoder. You can choose one from
`1-layer-lin`, `1-layer-nonlin`, `2-layer-nonlin`, `4-layer-nonlin`.
4. `--k` specifies the number of samples used for estimating the variational bound in the case of DVAE/DVAE++ 
and the importance weighted bound in the case of DVAE#. 
5. `--qupa` is a boolean flag that enables sampling using QuPA. Setting this flag to `False` will use PCD sampling.
6. `--beta` sets the beta hyperparameter for different smoothing transformation. Not setting this hyperparameter will
 rollback to the beta parameter used in our experiments.

Example:
```bash
python run.py \
    --log_dir=${PATH_TO_LOG_DIR} \
    --data_dir=${PATH_TO_DATA_DIR} \
    --dataset=binarized_mnist \
    --baseline=dvaes_exp \
    --struct=4-layer-nonlin \
    --k=5 \
    --qupa=True
```
<br />

## Running Tensorboard

You can monitor the progress of training and the performance on the validation and test datasets using tensorboard.
Run the following command to start tensorboard on the log directory:

```bash
tensorboard --logdir=${PATH_TO_LOG_DIR}
```

<br />

## Prerequisites
Make sure that you have:
* Python (version 2.7 or higher)
* Numpy
* Scipy
* [QuPA](https://try.quadrant.ai/qupa) 
* Tensorflow (The version should be compatible with QuPA)

<br/>

## Change Logs


##### July 13, 2018

First release including DVAE, DVAE++, and DVAE#. With both PCD and population annealing sampling.


<br/>

## Citation

if you use this code in your research, please cite us:
```
@article{vahdat2018dvae,
  title={DVAE\#: Discrete Variational Autoencoders with Relaxed {B}oltzmann Priors},
  author={Vahdat, Arash and Andriyash, Evgeny and Macready, William G},
  journal={arXiv preprint arXiv:1805.07445},
  year={2018}
}
```

<br/>

## References

[1] Discrete Variational Autoencoders, by Jason Tyler Rolfe, ICLR 2017, [paper](https://arxiv.org/abs/1609.02200). <br/>
[2] DVAE++: Discrete Variational Autoencoders with Overlapping Transformations by Arash Vahdat, William G. Macready, 
Zhengbing Bian, Amir Khoshaman, Evgeny Andriyash, ICML 2018, [paper](https://arxiv.org/abs/1802.04920). <br/>
[3] DVAE#: Discrete Variational Autoencoders with Relaxed Boltzmann Priors by Arash Vahdat*, Evgeny Andriyash*, 
William G. Macready, arXiv 2018, [paper](https://arxiv.org/abs/1805.07445).
