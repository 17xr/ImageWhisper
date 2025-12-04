import torch
from app.config import (
    DEPTH,
    DEVICE,
    EMBED_DIM,
    MAX_LENGTH,
    MAX_SEQ_LENGTH,
    MODEL_PATH,
    NO_REPEAT_NGRAM_SIZE,
    NUM_CAPTIONS,
    NUM_HEADS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TEXT_ENCODER_ID,
    TOP_K,
    TOP_P,
)

from architecture.transformer import ImageCaption
from transformers import AutoModel, AutoTokenizer
from utils.utils import nucleus_sampling_generate, format_caption

tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID)
word_embeddings = AutoModel.from_pretrained(
    TEXT_ENCODER_ID, trust_remote_code=True
).embeddings.word_embeddings

model = ImageCaption(
    word_embeddings,
    tokenizer.vocab_size,
    EMBED_DIM,
    NUM_HEADS,
    MAX_SEQ_LENGTH,
    DEPTH,
)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def generate_caption(tensor):
    captions = nucleus_sampling_generate(
        model,
        tensor,
        tokenizer,
        DEVICE,
        num_captions=NUM_CAPTIONS,
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        repetition_penalty=REPETITION_PENALTY,
    )

    return [format_caption(caption) for caption in captions]
