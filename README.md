# Practical Full Resolution Learned Lossless Image Compression


### Fabian Mentzer, Eirikur Agustsson, Michael Tschannen, Radu Timofte, Luc Van Gool
ETH Zurich

_CVPR'19 (oral presentation)_

<div align="center">
  <img src='figs/teaser.png' width="50%"/>
</div>


## [[Paper]](https://arxiv.org/abs/1811.12817) [[Citation]](#citation)

### Abstract

We propose the first practical learned lossless image compression system, L3C, and show that it outperforms the
popular engineered codecs, PNG, WebP and JPEG 2000.
At the core of our method is a fully parallelizable hierarchical probabilistic model for adaptive entropy coding which is optimized end-to-end for the compression task.
In contrast to recent autoregressive discrete probabilistic models such as PixelCNN, our method i) models the image distribution jointly with learned auxiliary representations instead of exclusively modeling the image distribution in RGB space, and ii) only requires three forward-passes to predict all pixel probabilities instead of one for each pixel.
As a result, L3C obtains over two orders of magnitude speedups when sampling compared to the fastest PixelCNN variant (Multiscale-PixelCNN).
Furthermore, we find that learning the auxiliary representation is crucial and outperforms predefined auxiliary representations such as an RGB pyramid significantly.


## Prerequisites for Code

Clone the repo and create a conda environment as follows:

```
conda create --name l3c_env python=3.7 pip --yes
conda activate l3c_env
```

We need PyTorch, CUDA, and some PIP packages:

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r pip_requirements.txt
```

To test our entropy coding, **you must also install torchac**, as [described below](#the-torchac-module-fast-entropy-coding-in-pytorch).

##### Notes
- We tested this code with Python 3.7 and PyTorch 1.1
- The training code also works with PyTorch 0.4, but for testing, we use the `torchac` module, which
needs PyTorch 1.0 or newer to build, [see below](#the-torchac-module-fast-entropy-coding-in-pytorch).
- The code relies on `tensorboardX==1.2`, even though TensorBoard is now part of PyTorch (since 1.1)

## Released models

We release the following trained models:


|     | Name | Training Set | ID  | Download Model |
| --- | ---- | ------------ | --- | --------- |
| Main Model | L3C | [Open Images](#prepare-open-images-for-training) | `0524_0001` | [L3C.tar.gz](http://data.vision.ee.ethz.ch/mentzerf/l3c_models/L3C.tar.gz) |
| Baseline | RGB Shared | Open Images | `0524_0002` | [RGB_Shared.tar.gz](http://data.vision.ee.ethz.ch/mentzerf/l3c_models/RGB_Shared.tar.gz) |
| Baseline | RGB | Open Images | `0524_0003` | [RGB.tar.gz](http://data.vision.ee.ethz.ch/mentzerf/l3c_models/RGB.tar.gz) |
| Main Model | L3C | [ImageNet32](http://image-net.org/download-images)    | `0524_0004` | [L3C_inet32.tar.gz](http://data.vision.ee.ethz.ch/mentzerf/l3c_models/L3C_inet32.tar.gz) |
| Main Model | L3C | [ImageNet64](http://image-net.org/download-images)    | `0524_0005` | [L3C_inet64.tar.gz](http://data.vision.ee.ethz.ch/mentzerf/l3c_models/L3C_inet64.tar.gz) |

See [Evaluation of Models](#evaluation-of-models) to learn how to evaluate on a dataset.

You can train them yourself using the following commands, after preparing the data as shown in
[Prepare Open Images Train](#prepare-open-images-for-training):

| Model | Train with the following flags to `train.py`|
| --- | --- |
| L3C               | `configs/ms/cr.cf configs/dl/oi.cf log_dir` |
| RGB Shared        | `configs/ms/cr_rgb_shared.cf configs/dl/oi.cf log_dir` |
| RGB               | `configs/ms/cr_rgb.cf configs/dl/oi.cf log_dir` |
| L3C ImageNet32    | `configs/ms/cr.cf configs/dl/in32.cf -p lr.schedule=exp_0.75_e1 log_dir` |
| L3C ImageNet64    | `configs/ms/cr.cf configs/dl/in64.cf -p lr.schedule=exp_0.75_e1 log_dir` |

Each of the released models were trained for around 5 days on a Titan Xp.

*Note*: We do not provide code for multi-GPU training. To incorporate `nn.DataParallel`, the code must be changed
slightly: In `net.py`, `EncOut` and `DecOut` are `namedtuple`s, which is not supported by `nn.DataParallel`.

### Results

When preparing this repo, we found that removing one approximation in the loss originally introduced by the PixelCNN++
 code slightly improved the final bitrates of L3C, while performance of the baselines got slightly worse.

<!-- TODO: add CR link -->

The code contains the loss without the approximation.
We note that [arXiv v2](https://arxiv.org/abs/1811.12817) is the same as CVPR Camera Ready version, and the results in there where obtained
with the approximation.

However, if you re-train with the provided code, you'll get the new results.
For clarity, we compare the new results as obtained by the **released code**, with the results in the **Camera Ready**:

| Model | Released [bpsp OI] | Camera Ready [bpsp OI] |
| ----- | ------------- | ----------------- |
| L3C   | 2.578         | 2.604 |
| RGB Shared | 2.948    | 2.918 |
| RGB   | 2.832         | 2.819 |

Here, _bpsp OI_ means bit per sub-pixel on Open Images Test.

We did not re-train the ImageNet32 and ImageNet64 models.


## Details about Code

### Experiments

Whenever `train.py` is executed, a new _experiment_ is started.
Every experiment is based on a specific configuration file for the network, stored in _configs/ms_ and
another file for the dataloading, stored in _configs/dl_.
An experiment is uniquely identified by the **log date**, which is just date and time (e.g. `0506_1107`).
The config files are parsed with the [parser from `fjcommon`](https://github.com/fab-jul/fjcommon#configparserpy),
which allow hiararchies of configs.
Additionally, there is `global_config.py`, to allow quick changes by passing
 additional parameters via the `-p` flag, which are then available _everywhere_ in the code, see below.

When an experiment is started, a directory with all this information is created in the folder passed as
`LOG_DIR_ROOT` to `train.py` (see `python train.py -h`).

For example, running

```bash
python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir -p upsampling=deconv
```

results in a folder `log_dir`, and in there another folder called

```
0502_1213 cr oi upsampling=deconv
```

Checkpoints (weights) will be stored in a subfolder called `ckpts`.

This experiment can then be evaluated simply by passing the log date to `test.py`, in addition to some image folders:


```bash
python test.py logs 0502_1213 data/openimages_test,data/raise1k
```

where we test on images in `data/openimages_test` and `data/raise1k`.

To use another model as a pretrained model, use `--restore` and `--restore_restart`:

```bash
python train.py configs/ll/cr.cf configs/dl/oi.cf logs --restore 0502_1213 --restore_restart
```

### Naming of code vs. paper

<div align="center">
  <img src='figs/arch_detail.png' width="99%"/>
</div>


| Name in Paper | Symbol | Name in Code | Short | Class |
| ----- | -------| -----| ----- | -----|
| Feature Extractor | `E` | Encoder | `enc` | `EDSRLikeEnc`
| Predictor | `D` | Decoder | `dec` | `EDSRLikeDec`
| Quantizer | `Q` | Quantizer | `q` | `Quantizer`
| Final box, outputting pi, mu, sigma |  | Probability Classifier | `prob_clf` | `AtrousProbabilityClassifier`

See also the notes in `src/multiscale_network/multiscale.py`.

### Structure of the code

The code is quite modular, as it was used to experiment with different things. At the heart is the
`MultiscaleBlueprint` class, which has the following main functions: `forward`, `get_loss`, `sample`. It is used by the
`MultiscaleTrainer` and `MultiscaleTester`. The network is created by `MultiscaleNetwork`, which pulls together all
the PyTorch modules needed. The discretized mixture of logistics loss is in `DiscretizedMixLogisticsLoss`, which is
usally referred to as `dmll` or `dmol` in the code.

For bitcoding, there is the `Bitcoding` class, which uses the `ArithmeticCoding` class, which in turn uses my
`torchac` module, written in C++, and described below.

<div align="center">
  <img src='figs/l3c_code_outline.jpg' width='60%'/>
</div>

### The `torchac` module: Fast Entropy Coding in Pytorch

We implemented an entropy coding module as a C++ extension for PyTorch, because no existing fast Python entropy
 coding module was available. You'll need to build it if you plan to use the `--write_to_file` flag for `test.py`
 ([see Evaluation of Models](#evaluation-of-models)).

The implementation is based on [this blog post](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html),
meaning that we implement _arithmetic coding_.
It is **not optimized**, however, it's much faster than doing the equivalent thing in pure-Python (because of all the
 bit-shift etc.). Encoding an entire `512 x 512` image happens in 0.202s (see Appendix A in the paper).

A good starting point for optimizing the code would probably be the [`range_coder.cc`](https://github.com/tensorflow/compression/blob/master/tensorflow_compression/cc/kernels/range_coder.cc)
implementation of
[TFC](https://tensorflow.github.io/compression/).

The module can be built with or without CUDA. The only difference between the CUDA and non-CUDA versions is:
With CUDA, `_get_uint16_cdf` from `torchac.py` is done with a simple/non-optimized CUDA kernel (`torchac_kernel.cu`),
which has one benefit: we can directly write into shared memory! This saves an expensive copying step from GPU to CPU.

However, compiling with CUDA is probably a hassle. On our machines, it works with a GCC newer than version 5 but
older than 6 (tested with 5.5), in combination with nvcc 9. We did not test other configurations, but they may work.
Please comment if you have insights into which other configurations work (or don't.)

The main part (arithmetic coding), is always on CPU.

#### Compiling

Make sure a recent `gcc` is available in `$PATH` (do `gcc --version`, tested with 5.5).
For CUDA, make sure `nvcc -V` gives the desired version (tested with 9.0).

Then do:

```
conda activate l3c_env
cd src/torchac
COMPILE_CUDA=auto python setup.py
```

- `COMPILE_CUDA=auto`: Use CUDA if a `gcc` between 5 and 6, and `nvcc` 9 is avaiable
- `COMPILE_CUDA=force`: Use CUDA, don't check `gcc` or `nvcc`
- `COMPILE_CUDA=no`: Don't use CUDA

This installs a package called `torchac-backend-cpu` or `torchac-backend-gpu` in your `pip`. To test if it works,
you can do

```
conda activate l3c_env
cd src/torchac
python -c "import torchac"
```

It should not print anything.

## Evaluation of Models

To test an experiment, use `test.py`. For example, to test L3C and the baselines, run

```
python test.py /path/to/logdir 0524_0001,0524_0002,0524_0003 /some/imgdir,/some/other/imgdir \
    --names "L3C,RGB Shared,RGB" --recursive=auto
```

To use the entropy coder and get timings for encoding/decoding, use `--write_to_files` (this needs `torchac`):

```
python test.py /path/to/logdir 0524_0001 /some/imgdir --write_to_files
```

More flags available with `python test.py -h`.

## Using L3C to compress images

To encode/decode a single image, use `l3c.py`. This requires `torchac`:

```bash
# Encode to out.l3c
python l3c.py /path/to/logdir 0524_0001 enc /path/to/img out.l3c
# Decode from out.l3c, save to decoded.png
python l3c.py /path/to/logdir 0524_0001 dec out.l3c decoded.png
```

## Sampling

To sample from L3C, use `test.py` with `--sample`:

```bash
python test.py /path/to/logdir 0524_0001 /some/imgdir --sample=samples
```

This produces outputs in a directory `samples`. Per image, you'll get something like

```bash
# Ground Truth
0_IMGNAME_3.549_gt.png
# Sampling from RGB scale, resulting bitcost 1.013bpsp
0_IMGNAME_rgb_1.013.png
# Sampling from RGB scale and z1, resulting bitcost 0.342bpsp
0_IMGNAME_rgb+bn0_0.342.png
# Sampling from RGB scale and z1 and z2, resulting bitcost 0.121bpsp
0_IMGNAME_rgb+bn0+bn1_0.121.png
```

See Section 5.4. ("Sampling Representations") in the paper.

## Prepare Open Images for training

### Option 1: Easy and Slow

Use the `prep_openimages.sh` script. Run it in an environment with
Python 3,
skimage (`pip install scikit-image`, we run version 0.13.1), and 
[awscli](https://aws.amazon.com/cli/) (`pip install awscli`):
```bash
cd src
./prep_openimages.sh DATA_DIR
```
This will download all images to `DATA_DIR`. Make sure there is enough space there, with the resulting tars an 
everything probably around 300GB!

### Option 2: Involved but can be faster

1. Download [Open Images training sets and validation set](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations),
we used the parts 0, 1, 2, plus the validation set:
    ```
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_0.tar.gz train_0.tar.gz
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz train_1.tar.gz
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_2.tar.gz train_2.tar.gz
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz validation.tar.gz
    ```
1. Extract to a folder, let's say `data`. Now you should have `data/train_0`, `data/train_1`, `data/train_2`, as well
 as `data/validation`.
1. (Optional) to do the same preprocessing as in our paper, run the following. Note that it requires the `skimage`
package. This can be parallelize over some server, by implementing a `task_array.sh`, see `import_train_images.py`.
    ```
    python import_train_images.py data train_0 train_1 train_2 validation
    ```
1. Put all (preprocessed) images into a train and a validation folder, let's say `data/train_oi` and
`data/validation_oi`.
1. (Optional) If you are on a slow file system, it helps to cache the contents of `data/train_oi`. Run
    ```
    cd src
    export CACHE_P="data/cache.pkl"  # <--- Change this
    export PYTHONPATH=$(pwd)
    python dataloaders/images_loader.py update data/train_oi "$CACHE_P"  --min_size 128
    ```
    The `--min_size` makes sure to skip smaller images. *NOTE*: If you skip this step, make sure no files with
    dimensions smaller than 128 are in your training foler. If they are there, training might crash.
1. Put all this into a train config. You can adapt `configs/dl/oi.cf` and update it: Set `train_imgs_glob =
'data/train_oi'` (or whatever folder you used.) If you did the previous step, set `image_cache_pkl = 'data/cache
.pkl`, if you did not, set `image_cache_pkl = None`. Finally, update `val_glob = 'data/validation_oi'`.
1. (Optional) It helps to have one fixed validation image to monitor training. You may put any image at
`src/train/fixedimg.jpg` and it will be used for that (see `multiscale_trainer.py`).

## Future Work for Code

- Add support for `nn.DataParallel`.
- Incorporate TensorBoard support from PyTorch, instead of pip package.

## Paper Errata

- p.6: "On ImageNet32/64, we increase the batch size to 120 [...]."
&rarr; Batch size is actually also 30, like for the other experiments.
- p.13, Fig A2: There should not be any arrows between predictors `D(1)`, because we only train one predictor.

## Citation

If you use the work released here for your research, please cite this paper:
```
@inproceedings{mentzer2019practical,
    Author = {Mentzer, Fabian and Agustsson, Eirikur and Tschannen, Michael and Timofte, Radu and Van Gool, Luc},
    Booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Title = {Practical Full Resolution Learned Lossless Image Compression},
    Year = {2019}}
```
