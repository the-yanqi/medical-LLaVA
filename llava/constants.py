CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

QUESTION_LIST_NATURAL = ['Give a brief description of the image.',
 'Give a short and clear explanation of the subsequent image.',
 'Provide a brief description of the given image.',
 'Share a concise interpretation of the image provided.',
 "Present a compact description of the photo's key features.",
 'Write a terse but informative summary of the picture.',
 'Render a clear and concise summary of the photo.',
 'Summarize the visual content of the image.',
 'Describe the image concisely.'] 

QUESTION_LIST_BINARY = ["Are there detectable masses/nodule/cyst on the mammogram? Answer with 'yes' or 'no' only.",
                "Does this image indicate the presence of any breast nodules or masses? Answer with 'yes' or 'no' only.",
                "Are masses visible on the mammogram? Answer with 'yes' or 'no' only.",
                "Is there any evidence of abnormal growths in this mammogram? Answer with 'yes' or 'no' only.",
                "Are nodular features or mass present in this mammographic image? Answer with 'yes' or 'no' only."]

QUESTION_LIST = ['What observations can you make from the mammogram of the xxx breast?',
'Can you highlight any notable features in the mammogram of the xxx breast?',
'Are there discernible abnomalies in the mammogram of the xxx breast?',
'What stands out to you in this mammogram of the xxx breast?',
'Based on the mammogram of the xxx breast, what are your primary observations?',
'Please provide a brief overview of the mammogram results of the xxx breast.',
'Can you pinpoint any areas of concern in the mammogram of the xxx breast?']

