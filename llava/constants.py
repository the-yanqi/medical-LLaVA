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

QUESTION_LIST_BINARY = ["Are there detectable masses/nodule/cyst on the mammogram? Answer with 'yes' or 'no' only.",
                "Does this image indicate the presence of any breast nodules or masses? Answer with 'yes' or 'no' only.",
                "Are masses visible on the mammogram? Answer with 'yes' or 'no' only.",
                "Is there any evidence of abnormal growths in this mammogram? Answer with 'yes' or 'no' only.",
                "Are nodular features or mass present in this mammographic image?"]

QUESTION_LIST = ['What observations can you make from the mammogram?',
'Can you highlight any notable features in the mammogram?',
'Are there discernible anomalies in the mammogram?',
'Can you walk me through your interpretation of the mammogram?',
'What stands out to you in this mammogram?',
'Based on the mammogram, what are your primary observations?',
'Please provide a brief overview of the mammogram results.',
'What details can you extract from this mammogram?',
'Can you pinpoint any areas of concern in the mammogram?']