import argparse
import json
import os
import re
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import prompt_template


PROMPTS = {
    "image": prompt_template["gpt_eval"]["image"],
    "video": prompt_template["gpt_eval"]["video"]
}


class SampleScore(BaseModel):
    sample_score: float


class GPTEvaluator:
    def __init__(self, api_key: str, base_url: str = None, timeout: int = 1000):
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        if base_url is None:
            print("You are using the defualt base url, you can set openai base url by `export OPENAI_BASE_URL=<YOUR_BASE_URL>`")
        else:
            self.client.base_url = base_url

    def eval(self, prompt: str, eval_input: str):
        # prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": eval_input}
                ]
            }
        ]
        
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=SampleScore)
        response = completion.choices[0].message.parsed

        return response


def extract_characters_regex(s):
    """
    Process prefix in multi-choice response
    This function is borrowed from VideoMME: https://video-mme.github.io/
    """
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer",
        "Answer:",
        "answer:",
        "answer",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]


def compute_metric_mc(model_predictions, log_path):
    
    total_num = 0
    valid_num = 0
    correct_num = 0
    
    for model_pred in tqdm(model_predictions, desc="Evaluate Multi-Choice Predictions:"):
        pred = model_pred["model_prediction"]
        extracted_pred = extract_characters_regex(pred)
        model_pred["extracted_model_prediction"] = extracted_pred
        gt = model_pred["ground_truth"]
        if extracted_pred != "":
            valid_num += 1
            if extracted_pred == gt:
                correct_num += 1
        total_num += 1

    accuracy = correct_num / valid_num * 100 if valid_num > 0 else "No valid predictions."

    result = {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "correct_num": correct_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_predictons": model_predictions
    }

    with open(log_path, "w") as f:
        json.dump(result, f, indent=4)


def evaluate_prediction(gpt_evaluator, model_pred):
    
    split = model_pred["split"]
    prompt = PROMPTS[split]

    eval_input = json.dumps({
        "question": model_pred["question"],
        "ground_truth_answer": model_pred["ground_truth"],
        "tester_response": model_pred["model_prediction"]
    })

    # Call GPT API
    try:
        response = gpt_evaluator.eval(prompt=prompt, eval_input=eval_input)
        score = response.sample_score
        model_pred["score"] = score
        model_pred["valid"] = True
        model_pred["error_in_gpt_eval"] = "No Error."
        return model_pred, score, True
    except Exception as e:
        print(f"Error details: {str(e)}")
        model_pred["score"] = 0
        model_pred["valid"] = False
        model_pred["error_in_gpt_eval"] = str(e)
        return model_pred, 0, False


def compute_metric_oe(model_predictions, log_path, num_process=1):
    """
    num_process: number of process to call GPT API in parallel
    """

    api_key = os.getenv('OPENAI_API_KEY', None)
    base_url = os.getenv('OPENAI_BASE_URL', None)
    gpt_evaluator = GPTEvaluator(api_key=api_key, base_url=base_url)
    
    total_num = 0
    valid_num = 0
    total_score = 0
    model_predictions_with_scores = []

    with ThreadPoolExecutor(max_workers=num_process) as executor:

        futures = [executor.submit(evaluate_prediction, gpt_evaluator, model_pred) for model_pred in model_predictions]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluate Open-Ended Predictions:"):
            model_pred, score, is_valid = future.result()
            model_predictions_with_scores.append(model_pred)
            total_score += score
            if is_valid:
                valid_num += 1
            total_num += 1

    accuracy = total_score / valid_num * 100 if valid_num > 0 else "No valid evaluations."

    result = {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpt_prompt": PROMPTS[model_pred["split"]],
        "model_predictons": model_predictions
    }

    with open(log_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_predictions", 
        type=str,
            help="path to saved model prediction results"
        )
    parser.add_argument(
        "--evaluation_formats", 
        choices=["mc", "oe"],
        help="Evaluation formats, support multiple-choice (mc) and open-ended (oe) formats."
        )
    parser.add_argument(
        "--log_path", 
        type=str,
            help="path to save evaluation results"
        )
    parser.add_argument(
        "--num_process", 
        type=int,
        default=1,
        help="number of process to call GPT in parallel"
        )

    args = parser.parse_args()

    # load model prediction
    model_predictions = json.load(open(args.model_predictions))

    # evaluate
    if args.evaluation_formats == "mc":
        compute_metric_mc(model_predictions, args.log_path)

    elif args.evaluation_formats == "oe":
        compute_metric_oe(model_predictions, args.log_path, args.num_process)

    print(f"The evaluation results and metrics are saved to: {args.log_path}")
