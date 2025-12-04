import torch

DEPTH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768
IMAGE_ENCODER_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
MAX_LENGTH = 64
MAX_SEQ_LENGTH = 1024
MODEL_PATH = "model/model.pt"
NO_REPEAT_NGRAM_SIZE = 3
NUM_CAPTIONS = 5
NUM_HEADS = 16
REPETITION_PENALTY = 1.3
TEMPERATURE = 0.7
TEXT_ENCODER_ID = "intfloat/e5-base-v2"
TOP_K = 45
TOP_P = 0.9
TRANSFORM_IMAGE_SIZE = (384, 384)
