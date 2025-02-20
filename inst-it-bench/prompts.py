prompt_template = {
    # image split, multiple-choice
    "image_mc": {
        "pre_prompt": "Based on the image, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[1]'', '[2]''. Respond with only the letter (A, B, C, or D) of the correct option.",
        "post_prompt": "The best answer is: "
    },

    # image split, open-ended
    "image_oe": {
        "pre_prompt": '# Task Definition:\nYou are an expert in image analysis. In this task, you will receive an image, and your task is to answer the given question based on the image content.\n# Guidelines and Rules:\n- Object References: In the image, each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., “[8]”) or as [ID1] [ID2] ... for multiple objects, such as “[3] [4] [5]”. Avoid commas, ranges, or phrases like “[1, 2, 3]” or “[1] to [3]”. The IDs in the images and questions match directly.',
        "post_prompt": "Based on the input image, please answer the question: "
    },
    
    # video split, multiple-choice
    "video_mc": {
        "pre_prompt": 'Based on the video, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., "[1]", "[2]"), and time references are shown using the format <timestamp> (e.g., "at <6>" or "during <7>-<8>"). Respond with only the letter (A, B, C, or D) of the correct option.',
        "post_prompt": "The best answer is: "
    },

    # video split, open-ended
    "video_oe": {
        "pre_prompt": '# Task Definition:\nYou are an expert in video analysis. In this task, you will receive a series of frames as a video, and your task is to answer the given questions based on the video content.\n# Input Format:\nThere are serveral images inputs as video frames. Each frame can be referenced by its timestamp (indicating when it appears in the video). For example, the first frame can be referred to as <1>.\n# Guidelines and Rules:\n- Object References: Each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., “[8]”) or as [ID1] [ID2] ... for multiple objects, such as “[3] [4] [5]”. Avoid commas, ranges, or phrases like “[1, 2, 3]” or “[1] to [3]”. The IDs in the images and questions match directly.\n- Time References: Use timestamps to indicate moments or intervals in the video. For a specific moment, format as <timestamp> (e.g., “at <3>”). For an interval, use <start_timestamp>-<end_timestamp> (e.g., “during <5>-<7>”). Always enclose timestamps in <>.',
        "post_prompt": "Based on the input video, please answer the question: "
    },

    # gpt evalution for open-ended results
    "gpt_eval": {
        # image split
        "image": """
# Task Description:
You are an expert evaluator tasked with scoring the accuracy of responses to open-ended questions. You will be provided with a set of questions, each with a corresponding ground-truth answer, as well as responses from a tester. Your job is to assess the accuracy of each response and provide a score between 0 and 1.
# Guidelines:
- Score Range: Your score for each test item must be between 0 and 1. A higher score means more correctness. Choose from the following: 0 (completely incorrect), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (completely correct)
- For each test item, consider the question, the ground-truth answer, and the tester’s response together to determine correctness.
- Objects in questions and answers may be referenced using the format [ID] (e.g., [1], [2]). Ensure that any objects referenced in the tester’s response match correctly with the ground-truth answer.
# Input Format:
The input is a set of test items to be scored, where each item includes: 
- `question`; 
- `ground truth answer for the question`; 
- `response from the tester`.
Now, let's begin the evaluation, here are the input test items: 
""",

        # video split
        "video": """
# Task Description:
You are an expert evaluator tasked with scoring the accuracy of responses to open-ended questions. You will be provided with a set of questions, each with a corresponding ground-truth answer, as well as responses from a tester. Your job is to assess the accuracy of each response and provide a score between 0 and 1.
# Guidelines:
- Score Range: Your score for each test item must be between 0 and 1. A higher score means more correctness. Choose from the following: 0 (completely incorrect), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (completely correct)
- For each test item, consider the question, the ground-truth answer, and the tester’s response together to determine correctness.
- Objects in questions and answers may be referenced using the format [ID] (e.g., [1], [2]). Ensure that any objects referenced in the tester’s response match correctly with the ground-truth answer.
- Time points may be indicated with <timestamp> (e.g., <1>), and time intervals with <start_timestamp>-<end_timestamp> (e.g., <3>-<5>). Verify that the tester’s response includes accurate time expressions.
# Input Format:
The input is a set of test items to be scored, where each item includes: 
- `question`; 
- `ground truth answer for the question`; 
- `response from the tester`.
Now, let's begin the evaluation, here are the input test items:
"""
    }
}