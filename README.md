<div align="center" style="font-family: charter;">
<img src="assets/logo.png" width="8%"/>
<h1><i>Inst-It</i>: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning</h1>
<a href="https://inst-it.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-Inst--It-ffc107?color=578FCA&logoColor=white" height="20" />
</a>
<a href="https://arxiv.org/abs/2412.03565" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Inst--It-red?logo=arxiv" height="20" />
</a>
<a href="https://huggingface.co/datasets/Inst-IT/Inst-It-Bench" target="_blank">
    <img alt="HF Dataset: Inst-It-Bench" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-Inst--It-ffc107?color=A9B5DF&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/datasets/Inst-IT/Inst-It-Dataset" target="_blank">
    <img alt="HF Dataset: Inst-It-Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-Inst--It-ffc107?color=B3D8A8&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/Inst-IT" target="_blank">
    <img alt="HF Model: Inst-It" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Inst--It-ffc107?color=FFCF50&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/spaces/Inst-IT/Leaderboard" target="_blank">
    <img alt="Leaderboard" src="https://img.shields.io/badge/🥇_Leaderboard-Inst--It-ffc107?color=E07A5F&logoColor=white" height="20" />
</a>

<div>
    <a href="https://scholar.google.com/citations?user=GTuWk9YAAAAJ&hl=zh-CN" target="_blank">Wujian Peng</a><sup>1,2*</sup>,</span>
    <a href="https://menglcool.github.io/" target="_blank">Lingchen Meng</a><sup>1*</sup>, </span>
    <a href="https://scholar.google.cz/citations?user=a40b6HQAAAAJ&hl=zh-CN" target="_blank">Yitong Chen</a><sup>1,2</sup>,</span>
    Yiweng Xie<sup>1</sup>,</span>
    Yang Liu<sup>1</sup>,</span>
    <a href="https://guitaowufeng.github.io/" target="_blank">Tao Gui</a><sup>1</sup>,</span>
    <a href="https://xuhangcn.github.io/" target="_blank">Hang Xu</a><sup>3</sup>,</span>
    <a href="https://xpqiu.github.io/en.html" target="_blank">Xipeng Qiu</a><sup>1,2</sup>,</span>
    <a href="https://zxwu.azurewebsites.net/" target="_blank">Zuxuan Wu</a><sup>1,2&dagger;</sup>,</span>
    <a href="https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en" target="_blank">Yu-Gang Jiang</a><sup>1</sup></span>
</div>

<div>
    <sup>1</sup>School of Computer Science, Fudan University&emsp;
    <sup>2</sup>Shanghai Innovation Institute&emsp;
    <sup>3</sup>Huawei Noah’s Ark Lab&emsp;
</div>

<div>
    <sup>*</sup> Equal contributions&emsp;
    <sup>&dagger;</sup> Corresponding author&emsp;
</div>

</div>

## 🔥 News
* `Feb. 19, 2025` Inst-It Bench [`Evaluation toolkit`](inst-it-bench/README.md) is released, you can evluate your model now!
* `Dec. 11, 2024` Inst-It Dataset is available at [`here`](https://huggingface.co/datasets/Inst-IT/Inst-It-Dataset). Welcome to use our dataset!
* `Dec. 5, 2024` our checkpoints are available at [`huggingface`](https://huggingface.co/Inst-IT).

## <img src="assets/leaderboard.png" alt="🏆" style="height: 1em;"> Inst-It Bench
Inst-It Bench is a fine-grained multimodal benchmark for evaluating LMMs at the instance-level.
* **Size:** ~1,000 image QAs and ~1,000 video QAs
* **Splits:** Image split and Video split
* **Evaluation Formats:** Open-Ended and Multiple-Choice
  
See this [`Evaluate.md`](inst-it-bench/README.md) to learn how to perform evaluation on Inst-It-Bench.

## <img src="assets/dataset.png" alt="🏆" style="height: 1em;"> Inst-It Dataset
Inst-It Dataset can be downloaded [`here`](https://huggingface.co/datasets/Inst-IT/Inst-IT-Dataset). To our knowledge, this is the first dataset that provides fine-grained annotations centric on specific instances. In total, Inst-it Dataset includes :
- **21k** videos
- **51k** images
- **21k** video-level descriptions
- **207k** frame-level descriptions (51k images, 156k video frames) (each frame-level description includes captions of 1)individual instances, 2)the entire image, and 3)the temporal changes.)
- **335k** open-ended QA pairs

We visualize the data structure in the figure below, and you can view a more detailed data sample [`here`](https://inst-it.github.io/#dataset).
<p align="center">
    <img src="https://inst-it.github.io/images/data.png" width="80%"> <br>
</p>

<details>
<summary>click here to see the annotation format of Inst-It-Bench</summary>
  
- video annotations in file [`inst_it_dataset_video_21k.json`](https://huggingface.co/datasets/Inst-IT/Inst-It-Dataset/blob/main/inst_it_dataset_video_21k.json)

```
[
    {
        "video_id": int,
        "frame_level_caption": (annotation for each frame within this video)
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

- image annotations in file [`inst_it_dataset_image_51k.json`](https://huggingface.co/datasets/Inst-IT/Inst-It-Dataset/blob/main/inst_it_dataset_image_51k.json)
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

## <img src="assets/model.png" alt="🌐" style="height: 1em;"> Model weights
We trained two models based on LLaVA-Next using our [`Inst-It-Dataset`](https://huggingface.co/datasets/Inst-IT/Inst-It-Dataset), which not only achieve outstanding performance on [`Inst-It-Bench`](https://huggingface.co/datasets/Inst-IT/Inst-It-Bench) but also demonstrate significant improvements on other generic image and video understanding benchmarks. We provide the checkpoints here:
| Model | Checkpoints |
|:----------:|:----------:|
| LLaVA-Next-Inst-It-Vicuna-7B | [`weights and docs`](https://huggingface.co/Inst-IT/LLaVA-Next-Inst-It-Qwen2-7B) | 
| LLaVA-Next-Inst-It-Qwen2-7B | [`weights and docs`](https://huggingface.co/Inst-IT/LLaVA-Next-Inst-It-Vicuna-7B) |

## <img src="assets/todo.png" alt="📝" style="height: 1em;">  Todo
- [x] Release the Inst-It Bench data and evaluation code.
- [x] Release the Inst-It Dataset.
- [x] Release the checkpoint of our fine-tuned models.
- [ ] Release the meta-annotation of Inst-It Dataset, such as instance segmentation masks, bounding boxes, and more ...
- [ ] Release the annotation file of Inst-It Dataset, which follows the format in the LLaVA codebase.
- [ ] Release the training code.

##  <img src="assets/email.png" alt="📧" style="height: 1em;"> Contact Us
Feel free to contact us if you have any questions or suggestions 
- Email (Wujian Peng): wjpeng24@m.fudan.edu.cn
- Email (Lingchen Meng): lcmeng20@fudan.edu.cn
##  <img src="assets/cite.png" alt="📎" style="height: 1em;"> Citation
If you find our work helpful, please consider citing our paper :paperclip: and starring our repo :star2: :

``` bibtex
@article{peng2024inst,
  title={Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning},
  author={Peng, Wujian and Meng, Lingchen and Chen, Yitong and Xie, Yiweng and Liu, Yang and Gui, Tao and Xu, Hang and Qiu, Xipeng and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2412.03565},
  year={2024}
}
```
