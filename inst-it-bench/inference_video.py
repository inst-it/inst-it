import os
import json
import argparse
import torch
import zipfile
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from prompts import prompt_template
import torch.distributed as dist
from dist import init_distributed_mode


def prepare_videos(cache_dir=None):
    """
    Download videos.zip from huggingface and unzip it
    cache_dir: the path to save downloaded videos
    """
    if cache_dir is None:
        hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
        hf_home = os.path.expanduser(hf_home)
        cache_dir = os.path.join(hf_home, "datasets/inst-it-bench")

    zip_path = hf_hub_download(
        repo_id="Inst-IT/Inst-It-Bench", 
        filename="videos_vpt.zip", 
        repo_type="dataset",
        local_dir=cache_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    
    print(f"Videos are downloaded and unzipped to: {cache_dir}")
    return cache_dir
    

DATASET = {
    "mc": load_dataset("Inst-IT/Inst-It-Bench", "video_multi_choice", split="test"),
    "oe": load_dataset("Inst-IT/Inst-It-Bench", "video_open_ended", split="test"),
}

PROMPT = {
    "mc": prompt_template["video_mc"],
    "oe": prompt_template["video_oe"]
}

def main(args):

    # distributed setup
    local_rank, world_size = args.rank, args.world_size
    device = torch.device('cuda:{}'.format(args.rank))

    # download videos from huggingface and unzip them
    video_root = prepare_videos(cache_dir=args.cache_dir)

    # =====================Modify here to load you model===========================
    # TODO: Load your model here
    from llava.model.builder import load_pretrained_model
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        tokenizer_image_token,
        process_images
    )
    from llava.conversation import SeparatorStyle, conv_templates

    overwrite_config = {}
    overwrite_config["mm_spatial_pool_stride"] = 2
    overwrite_config["mm_spatial_pool_mode"] = 'bilinear'
    overwrite_config["mm_pooling_position"] = 'after'
    overwrite_config["mm_newline_position"] = 'no_token'

    model_path = "Inst-IT/LLaVA-Next-Inst-It-Vicuna-7B"
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, max_length = load_pretrained_model(
                model_path=model_path, 
                model_base=None, 
                model_name=model_name,
                # device_map="auto", 
                device_map=device, 
                torch_dtype='bfloat16', 
                overwrite_config=overwrite_config,
                attn_implementation='sdpa')
    # =====================Modify here to load you model===========================


    for eval_format in args.evaluation_formats:
        
        # Load Dataset
        dataset = DATASET[eval_format]

        # Split Dataset for Each GPU
        dataset = [data for data in dataset]
        dataset = dataset[local_rank::world_size]

        # Prompt Templates
        prompt_template = PROMPT[eval_format]

        # Inference
        results = []
        for data in tqdm(dataset, desc=f"Processing {eval_format} Inference"):
            # prepare input text
            if eval_format == "mc":
                question = data["question"]
                options = f"A. {data['choice_a']}\nB. {data['choice_b']}\nC. {data['choice_c']}\nD. {data['choice_d']}"
                input_text = f"{prompt_template['pre_prompt']}\n{question}\n{options}\n{prompt_template['post_prompt']}"
            elif eval_format == "oe":
                question = data["question"]
                input_text = f"{prompt_template['pre_prompt']}\n{prompt_template['post_prompt']}\n{question}"
            
            # prepare input video
            video_path = os.path.join(video_root, data["video_path"])
            frame_names = os.listdir(video_path)
            frame_names.sort()
            video = []
            for frame in frame_names:
                image = Image.open(os.path.join(video_path, frame))
                video.append(image.convert("RGB"))

            # =====================Modify here to run model inference===========================
            # TODO: Run model inference here
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
            video = video.bfloat16()
            videos = [video]

            input_text = DEFAULT_IMAGE_TOKEN + "\n" + input_text

            conv_template = 'vicuna_v1'
            conv = conv_templates[conv_template].copy()
            conv.append_message(conv.roles[0], input_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    modalities="video",
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    max_new_tokens=128
                )

            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # =====================Modify here to run model inference===========================


            results.append(
                {
                    "question_id": data["question_id"],
                    "question": data["question"],
                    "model_prediction": pred,
                    "ground_truth": data["answer"],
                    "split": "video"
                }
            )
        
        # Gather results from each GPU
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, results)
        all_results = [item for sublist in gathered_results for item in sublist]
        all_results.sort(key=lambda x: x["question_id"])

        # Save Predictions
        if local_rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, f"video_{eval_format}_inference.json"), "w") as f:
                json.dump(all_results, f, indent=4)

        print(f"{eval_format} Inference Finished, the model responsing is saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", 
        type=str,
         help="path to save inference results"
         )
    parser.add_argument(
        "--evaluation_formats", 
        nargs="+", 
        choices=["mc", "oe"],
        help="Evaluation formats, support multiple-choice (mc) and open-ended (oe) formats."
        )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The path to save videos downloaded from huggingface datasets."
    )
    # distributed parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local-rank', default=0, type=int)

    args = parser.parse_args()

    # distributed 
    init_distributed_mode(args)
    print(args)
    
    main(args)
