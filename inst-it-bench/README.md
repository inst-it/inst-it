# <img src="../assets/leaderboard.png" alt="🏆" style="height: 1em;"> Inst-It Bench
<a href="https://huggingface.co/datasets/Inst-IT/Inst-It-Bench" target="_blank">
    <img alt="HF Dataset: Inst-It-Bench" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-Inst--It-ffc107?color=A9B5DF&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/spaces/Inst-IT/Leaderboard" target="_blank">
    <img alt="Leaderboard" src="https://img.shields.io/badge/🥇_Leaderboard-Inst--It-ffc107?color=E07A5F&logoColor=white" height="20" />
</a>

Inst-It Bench is a fine-grained multimodal benchmark for evaluating LMMs at the instance-level.
* **Size:** ~1,000 image QAs and ~1,000 video QAs
* **Splits:** Image split and Video split
* **Evaluation Formats:** Open-Ended and Multiple-Choice

<p align="center">
    <img src="https://github.com/inst-it/inst-it.github.io/blob/main/images/web_bench_exp1.png?raw=true" width="90%"> <br>
</p>
<details>
<summary>Click here to unfold more data examples:</summary>
<p align="center">
    <img src="https://github.com/inst-it/inst-it.github.io/blob/main/images/web_bench_exp2.png?raw=true" width="90%"> <br>
</p>

<p align="center">
    <img src="https://github.com/inst-it/inst-it.github.io/blob/main/images/web_bench_exp3.png?raw=true" width="90%"> <br>
</p>
</details> 

## Step 1: Model predicting
The first step is to run inference on Inst-It Bench using your model and obtain the model’s predictions for the questions. Please ensure that your model responses are saved in the following formats, see [examples](output/).
```python
{
    "question_id": "xxx", # the id of each question
    "question": "xxx",  # the question
    "model_prediction": "xxx", # model responses
    "ground_truth": "xxx", # ground truth answer
    "split": "xxx" # image or video, this is required when employing GPT to evaluate open-ended QAs
}
```

We use our [LLaVA-NeXT-Inst-It-Vicuna-7B](https://huggingface.co/Inst-IT/LLaVA-Next-Inst-It-Vicuna-7B) as an example to demonstrate how to run inference on both image and video splits. You can easily modify them to run inference with your model.

* `--save_dir` is the path where the model’s responses will be saved.
* `--evaluation_formats` where `oe` stands for open-ended, and `mc` stands for multiple-choice.
* `--nproc_per_node` number of GPUs used for inference in paralell.
* `--cache_dir` when inference on video split, the script will automaitcally download videos from huggingface, you can specify where the videos will be saved, otherwise, the videos will be saved to the default huggingface cache dir.
``` shell
# install for LLaVA-NeXT-Inst-It
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# Image Split
python -m torch.distributed.launch --nproc_per_node=8 inference_image.py \
--save_dir output \
--evaluation_formats mc oe

# Video Split
python -m torch.distributed.launch --nproc_per_node=8 inference_video.py \
--save_dir output \
--evaluation_formats mc oe \
--cache_dir data
```

## Step 2: Evaluate model predictions
Once the model’s predictions are generated in [this format](output/), you can run the following scripts to compute the relevant metrics.
``` shell
pip install openai
pip install pydantic
```

* `--model_predictions`: The path where model predictions are saved, should be the same as `save_dir` in Step 1.
* `--log_path`: The path to save evaluation results.

### Multiple Choice
``` shell
# Image split
python evaluate.py \
--evaluation_formats mc \
--model_predictions output/image_mc_inference.json \
--log_path output/image_mc_evaluate.json

# Video split
python evaluate.py \
--evaluation_formats mc \
--model_predictions output/video_mc_inference.json \
--log_path output/video_mc_evaluate.json
```

### Open-Ended
We use `gpt-4o-2024-0806` to evaluate open-ended responses. 
* `--num-process` To speed up the evaluation, you can set the number of processes for parallel GPT API calls .
``` shell

export OPENAI_API_KEY="<YOUR_API_KEY>"
export OPENAI_BASE_URL="<YOUR_BASE_URL>" # set to None if use defualt base_url

# Image split
python evaluate.py \
--evaluation_formats oe \
--model_predictions output/image_oe_inference.json \
--log_path output/image_oe_evaluate.json \
--num_process 16

# Video split
python evaluate.py \
--evaluation_formats oe \
--model_predictions output/video_oe_inference.json \
--log_path output/video_oe_evaluate.json \
--num_process 16
```

## Step 3: Submit results to Leaderboard
* We host a leaderboard at [this website](https://huggingface.co/spaces/Inst-IT/Leaderboard).
* If you want to submit your results, please email wjpeng24@m.fudan.edu.cn following the instructions at [this doc]().
* Feel free to submit your model results and join the leaderboard : - )

