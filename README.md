<div align="center">

<h1><a href="https://ieeexplore.ieee.org/document/9934924">Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset</a></h1>

**[Chenyang Liu](https://chen-yang-liu.github.io/), [Rui Zhao](https://ruizhaocv.github.io), [Hao Chen](http://chenhao.in/), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), and [Zhenwei Shi*✉](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

![visitors](https://visitor-badge.glitch.me/badge?page_id=Chen-Yang-Liu.RSICC.visitor)
![GitHub stars](https://badgen.net/github/stars/Chen-Yang-Liu/RSICC)

</div>

## Share us a :star: if this repo does help

### 🥳 New

🔥 Our survey "**Remote Sensing Temporal Vision-Language Models: A Comprehensive Survey": [Arxiv](https://arxiv.org/abs/2412.02573) || [Github](https://github.com/Chen-Yang-Liu/Awesome-RS-Temporal-VLM)** 🔥 

## LEVIR-CC Dataset 
**Download [Link](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)**

## RSICCfromer
Here, we provide the pytorch implementation of the paper: "Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset". 

For more information, please see our published paper in [[IEEE](https://ieeexplore.ieee.org/document/9934924) | [Lab Server](http://levir.buaa.edu.cn/publications/ChangeCaptioning.pdf)]  ***(Accepted by TGRS 2022)***

![RSICCformer_structure](Example/RSICCformer_structure.png)

### Installation and Dependencies
This fork updates the original code for modern Python and PyTorch and adds a Colab training notebook.

Training, evaluation and captioning have been run in the following environments:

| Environment | Python | PyTorch | torchvision | PyTorch CUDA build | GPU |
| --- | --- | --- | --- | --- | --- |
| Google Colab (November 2026) | 3.13.16 | 2.14.1+cu130 | 0.29.1+cu130 | 13.0 | NVIDIA A100, NVIDIA RTX PRO 6000 Blackwell |
| Local Windows virtual environment | 3.11.5 | 2.6.0+cu124 | 0.21.0+cu124 | 12.4 | NVIDIA RTX 3060 Laptop |

The Colab runs used the complete notebook workflow on LEVIR-CC. The local environment ran training, resume, evaluation and captioning on a small subset, and the unit tests. The CUDA version refers to the PyTorch build, not the installed NVIDIA driver. `requirements.txt` gives lower bounds; each run's `packages.txt` records the exact versions used.

With a fixed seed, the first epoch reproduced exactly between two runs on the same GPU model. Later epochs differ slightly unless `--deterministic` is set, and results differ between GPU models.

```bash
git clone https://github.com/yakupgulcan/RSICC.git
cd RSICC
conda create -n RSICCformer_env python=3.11
conda activate RSICCformer_env
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The commands above use the PyTorch/torchvision versions installed in the local training environment. METEOR evaluation also requires Java; check that `java -version` works before training or evaluation.

### Training in Google Colab
Open [RSICC_Colab_Training.ipynb](RSICC_Colab_Training.ipynb) in Colab, select a GPU runtime, edit the configuration cell and run all cells. The notebook only holds settings; the logic lives in this repository at the revision named by `REPO_REF`.

Each run creates `RSICC_Outputs/runs/<experiment>_seed<seed>_<time>/` in Google Drive:

| File | Content |
| --- | --- |
| `run.json` | Code commit, all notebook settings, dataset source and SHA-256, seeds, environment, best validation scores, test scores and the evaluated checkpoint's checksum |
| `packages.txt` | Installed Python packages |
| `data.log`, `train.log`, `eval.log` | Complete output of each step |
| `training_metrics.jsonl` | Validation scores per epoch |
| `models_checkpoint/` | Latest and best checkpoint |
| `eval_results/<time>/` | Test scores, generated captions and the example caption |

Notes:
- `SEED` controls training. `DATA_SEED` controls caption sampling when the training files are built; keep it fixed so that runs with different seeds use identical data.
- Checkpoints store weights and the model settings. Resuming checks that the settings match. Checkpoints written before this format can be evaluated but not resumed.
- To resume, set `RESUME_CHECKPOINT` to the Drive path of an earlier `checkpoint_*.pth.tar` and `TOTAL_EPOCHS` to the total target. Random-number states are not restored, so a resumed run is not identical to an uninterrupted one.
- New `train.py` options can be passed through `EXTRA_TRAIN_ARGS` without editing the notebook.
- `HARMONIC_ALPHA` (`--harmonic_alpha`) smooths the two CNN feature maps on one shared 8-neighbour graph before the feature encoder (`harmonic.py`): `(I + alpha*L) S = F`. `0` is the unchanged baseline. The graph has 14x14 nodes, so useful values are roughly 0.1 to 10. The value is stored in the checkpoint, so evaluation needs no extra option.

### Development workflow
- `main` always holds a working baseline. Results that are reported come from a tag or commit on GitHub, never from uncommitted code (`run.json` records `uncommitted_changes`).
- Pipeline changes go on `infra/<topic>` branches, architecture changes on `arch/<idea>` branches. To train a branch in Colab, push it and set `REPO_REF` to its name or commit.
- Where possible, make an architecture variant selectable by a `train.py` option, so that the baseline and the variant can run from the same commit.
- Run the unit tests before pushing: `python -m unittest discover -s tests`.

### Data preparation
Firstly, put the downloaded dataset in `./Levir_CC_dataset/`.
Then preprocess dataset as follows:
```python
python create_input_files.py
```
The preprocessing script sets `min_word_freq=5` in its `create_input_files` call; edit that argument in the script to change it.
After that, you can find some resulted files in `./data/`. 

Besides, the resulted files can also be downloaded from here: [[Google Drive](https://drive.google.com/drive/folders/1cEv-BXISfWjw1RTzL39uBojH7atjLdCG?usp=sharing)  &#124; [Baidu Pan](https://pan.baidu.com/s/1YrWcz090kdqOZ0lrbqXJJA) (code:nq9y)]. Extract it to `./data/`.

!NOTE: For a fair comparison, we suggest that future researchers ensure `min_word_freq <= 5` or use our preprocessed data above with `min_word_freq = 5`.

### Inference Demo
You can download our RSICCformer pretrained model——by [[Google Drive](https://drive.google.com/drive/folders/1cEv-BXISfWjw1RTzL39uBojH7atjLdCG?usp=sharing)  &#124; [Baidu Pan](https://pan.baidu.com/s/1SBGjVS0yd2KHdK9t4NuiyA) (code:2fbc)]

After downloaded the pretrained model (one ".pth.tar" file, please don't unzip it), you can put it in `./models_checkpoint/`.

Then, run a demo to get started as follows:
```python
python caption.py --img_A ./Example/A/train_000016.png --img_B ./Example/B/train_000016.png --path ./models_checkpoint/
```
After that, you can find the generated caption in `./eval_results/`


### Train
Make sure you performed the data preparation above. Then, start training as follows:
```python
python train.py  --data_folder ./data/ --savepath ./models_checkpoint/
```

### Evaluate
```python
python eval.py --data_folder ./data/ --path ./models_checkpoint/ --Split TEST
```
We recommend training 5 times to get an average score.

## Citation: 
```
@ARTICLE{9934924,
  author={Liu, Chenyang and Zhao, Rui and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset}, 
  year={2022},
  volume={60},
  number={},
  pages={1-20},
  doi={10.1109/TGRS.2022.3218921}}
```
## Reference:
Thanks to the following repository:
[a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git)




