<div align="center">
  <img src="assets/logo.png" width="100px" style="vertical-align: middle; display: inline;">
</div>

# <span style="font-variant: small-caps">Inst-IT</span>: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning

_[Wujian Peng<sup>1,2*</sup>](https://scholar.google.com/citations?user=GTuWk9YAAAAJ&hl=zh-CN), [Lingchen Meng<sup>1*</sup>](https://menglcool.github.io/), Yitong Chen<sup>1,2</sup>, Yiweng Xie<sup>1</sup>, Yang Liu<sup>1</sup>,_
_[Tao Gui<sup>1</sup>](https://guitaowufeng.github.io/), [Hang Xu<sup>3</sup>](https://xuhangcn.github.io/), [Xipeng Qiu<sup>1,2</sup>](https://xpqiu.github.io/en.html), [Zuxuan Wu<sup>1,2&dagger;</sup>](https://xpqiu.github.io/en.html), [Yu-Gang Jiang<sup>1</sup>](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)_

 <sup>1</sup>School of Computer Science, Fudan University; <sup>2</sup>Shanghai Innovation Institute; <sup>3</sup>Huawei Noah’s Ark Lab

 <sup>*</sup> Equal contributions; <sup>&dagger;</sup> Corresponding author.

[[🌐 Project Page](https://inst-it.github.io/)]
[[<img src="assets/paper.png" alt="📄" style="height: 1em;"> Paper](https://arxiv.org/abs/2412.03565)] 
[[<img src="assets/hf.png" alt="🤗" style="height: 1em;"> Inst-IT Bench](https://huggingface.co/datasets/Inst-IT/Inst-IT-Bench)] 
[<img src="assets/hf.png" alt="🤗" style="height: 1em;"> Inst-IT Dataset] 
[[<img src="assets/hf.png" alt="🤗" style="height: 1em;"> Checkpoints](https://huggingface.co/Inst-IT)] 
[<img src="assets/leaderboard.png" alt="🏆" style="height: 1em;"> Leaderboard] 

## 🔥 News
* `Dec. 11, 2024` :fire: Inst-IT Dataset is available at [here](https://huggingface.co/datasets/Inst-IT/Inst-IT-Dataset). Welcome to use our dataset!
* `Dec. 10, 2024` :fire: Inst-IT Bench is available at [here](https://huggingface.co/datasets/Inst-IT/Inst-IT-Bench), and the evaluation guidelines is coming.
* `Dec. 5, 2024` :fire: our checkpoints are available at [huggingface](https://huggingface.co/Inst-IT).
* `Dec. 5, 2024` :fire: our paper can be previewed on [arXiv](https://arxiv.org/abs/2412.03565).

## <img src="assets/leaderboard.png" alt="🏆" style="height: 1em;"> Inst-IT Bench: A Fine-grained Multimodal Benchmark for Evaluating LMMs at Instance-Level
### Step 1: Prepare environment and data
#### install
```shell
pip install openai
pip install -U "huggingface_hub[cli]"
```
#### prepare data
```shell
huggingface-cli download --repo-type dataset --resume-download Inst-IT/Inst-IT-Bench --local-dir Inst-IT-Bench
cd Inst-IT-Bench && unzip images.zip && unzip videos.zip
```

### Step 2: Inference your model on Inst-IT Bench
will be updated before `Dec. 12, 2024`, please stay tuned.

### Step 3: Evaluate model predictions
will be updated before `Dec. 12, 2024`, please stay tuned.

## <img src="assets/dataset.png" alt="🏆" style="height: 1em;"> Inst-IT Dataset: An Instruction Tuning Dataset with Multi-level Fine-Grained Annotations
Inst-IT Dataset can be downloaded [`here`](https://huggingface.co/datasets/Inst-IT/Inst-IT-Dataset). To the best of our knowledge, this is the first dataset that provides fine-grained annotations centric on specific instances. Inst-it Dataset contains **21k videos** and **51k images** (we treat images as static, single-frame videos). In total, Inst-it Dataset includes :
- **21k** videos
- **51k** images
- **21k** video-level descriptions
- **207k** frame-level descriptions (51k images, 156k video frames) (each frame-level description includes captions of 1)individual instances, 2)the entire image, and 3)the temporal changes.)
- **335k** open-ended QA pairs

We visualize the data structure in the figure below, and you can view a more detailed data sample [[`here`]](https://inst-it.github.io/#dataset).
<p align="center">
    <img src="https://inst-it.github.io/images/data.png" width="70%"> <br>
</p>

<details>
<summary>click here to see the annotation format of Inst-IT Bench</summary>
  
- video annotations in file [`inst_it_dataset_video_21k.json`](https://huggingface.co/datasets/Inst-IT/Inst-IT-Dataset/blob/main/inst_it_dataset_video_21k.json)

```
[
    {
        "video_id": int,
        "frame_level_caption": (annotation for each frames within this video)
          [
              {
                  "timestamp": int, (indicate the timestamp of this frame in the video, e.g. <1>)
                  "frame_name": string, (the image filename of this frame)
                  "instance_level": (caption for each instance within this frame)
                    {
                        "1": "caption for instance 1",
                        (more instance level captions ...)
                    },
                  "image_level": string, (caption for the entire frame)
                  "temporal_change": string (caption for the temporal changes relative to the previous frame)
              },
              (more frame level captions ...)
          ],
        "question_answer_pairs": (open ended question answer pairs)
          [
             {
                "question": "the question",
                "answer": "the corresponding answer"
              },
             (more question answer pairs ...)
          ],
        "video_level_caption": string, (a dense caption for the entire video, encompassing all frames)
        "video_path": string (the path to where this video is stored)
    },
    (more annotations for other videos ...)
]
```

- image annotations in file [`inst_it_dataset_image_51k.json`](https://huggingface.co/datasets/Inst-IT/Inst-IT-Dataset/blob/main/inst_it_dataset_image_51k.json)
```
[
    {
        "image_id": int,
        "instance_level_caption": (caption for each instance within this image)
          {
              "1": "caption for instance 1",
              (more instance level captions ...)
          },
        "image_level_caption": string, (caption for the entire image)
        "image_path": string (the path to where this image is stored)
    },
    (more annotations for other images ...)
]
```
</details>

Welcome to use our Inst-IT Dataset to train your LMMs! 

## <img src="assets/model.png" alt="🌐" style="height: 1em;"> Model weights
We trained two models base on LLaVA-Next using our [Inst-IT Dataset](https://huggingface.co/datasets/Inst-IT/Inst-IT-Dataset), which not only achieve outstanding performance on [Inst-IT Bench](https://huggingface.co/datasets/Inst-IT/Inst-IT-Bench) but also demonstrate significant improvements on other generic image and video understanding benchmarks. We provide the checkpoints here:
| Model | Checkpoints |
|:----------:|:----------:|
| LLaVA-Next-Inst-It-Vicuna-7B | [weights](https://huggingface.co/Inst-IT/LLaVA-Next-Inst-It-Qwen2-7B) | 
| LLaVA-Next-Inst-It-Qwen2-7B | [weights](https://huggingface.co/Inst-IT/LLaVA-Next-Inst-It-Vicuna-7B) |

## <img src="assets/todo.png" alt="📝" style="height: 1em;">  Todo
- [x] Release the Inst-IT Bench data and evaluation code.
- [x] Release the Inst-IT Dataset.
- [x] Release the checkpoint of our fine-tuned models.
- [ ] Release the meta-annotation of Inst-IT Dataset, such as instance sgementation masks, bounding boxes, and more ...
- [ ] Release the annotation file of Inst-IT Dataset, which follows the format in LLaVA codebase.
- [ ] Add a inference script of our finetuned model, i.e. a quick start snippet.
- [ ] Release the leaderboard of Inst-IT Bench.
- [ ] Release the training code.

##  <img src="assets/email.png" alt="📧" style="height: 1em;"> Contact Us
Feel free to contact us if you have any questions or suggestions 
- Email (Wujian Peng): wjpeng24@m.fudan.edu.cn
- Email (Lingchen Meng): lcmeng20@fudan.edu.cn
##  <img src="assets/cite.png" alt="📎" style="height: 1em;"> Citation
If you find our work helpful, please consider citing our paper :paperclip: and starring our repo :star2: :

```
 @article{peng2024boosting,
   title={Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning},
   author={Peng, Wujian and Meng, Lingchen and Chen, Yitong and Xie, Yiweng and Liu, Yang and Gui, Tao and Hang, Xu and Qiu, Xipeng and Wu, Zuxuan and Jiang, Yu-Gang},
   journal={arXiv preprint arXiv:2412.03565},
   year={2024}
 }
```