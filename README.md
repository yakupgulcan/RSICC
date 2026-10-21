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

Training has been completed in the following environments:

| Environment | Python | PyTorch | torchvision | PyTorch CUDA build | GPU |
| --- | --- | --- | --- | --- | --- |
| Local Windows virtual environment | 3.11.5 | 2.6.0+cu124 | 0.21.0+cu124 | 12.4 | Not recorded |
| Google Colab | Not recorded | Not recorded | Not recorded | Not recorded | NVIDIA A100 |

The local versions were read from `.venv`; successful training in both environments was reported by the fork maintainer. The CUDA version above refers to the PyTorch build, not the installed NVIDIA driver. Other version combinations have not been verified. The lower bounds in `requirements.txt` are dependency constraints, not a tested compatibility matrix.

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
Open [RSICC_Colab_Training.ipynb](RSICC_Colab_Training.ipynb) in Colab and select a GPU runtime. Training has been run on an A100; the notebook does not require a specific Colab subscription.

1. Mount Google Drive and set `REPO_URL` and `REPO_REF` in the repository setup cell. Use the branch containing these updates, or a commit hash to reproduce a particular revision.
2. Install dependencies and run the environment report cell. Save its output with your experiment to record Python, PyTorch, torchvision, CUDA, GPU, and the Git revision.
3. Set the dataset download URL and Drive backup paths, then download or restore the data and preprocess it.
4. Run training, evaluation, and the example caption command. Checkpoints and evaluation results are copied or written to the configured Drive directories.

The example training command uses `--batch_size 32`, `--workers 4`, and `--epochs 20`. Adjust these for available memory and your experiment. Execution times depend on the runtime and data. Evaluation processes all files in the checkpoint directory, so use a directory containing only the checkpoint to evaluate when comparing individual models; result filenames are reused between checkpoints.

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




