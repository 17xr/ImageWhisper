import torch
from app.config import (
    BEAM_SIZE,
    DEPTH,
    DEVICE,
    DIVERSITY_PENALTY,
    DROPOUT,
    EMBED_DIM,
    INIT_TEMPERATURE,
    INIT_TOP_K,
    MAX_LEN,
    MAX_SEQ_LEN,
    MODEL_PATH,
    NO_REPEAT_NGRAM_SIZE,
    NUM_HEADS,
    STOCHASTIC_INIT_STEPS,
    TEMPERATURE,
    TEXT_ENCODER_ID,
    TOP_K
)

from architecture.transformer import ImageCaption
from transformers import AutoModel, AutoTokenizer
from utils.utils import beam_search_generate, format_caption

tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID)
word_embeddings = AutoModel.from_pretrained(
    TEXT_ENCODER_ID, trust_remote_code=True
).embeddings.word_embeddings

model = ImageCaption(
    DEVICE,
    word_embeddings,
    tokenizer,
    EMBED_DIM,
    NUM_HEADS,
    MAX_SEQ_LEN,
    DEPTH,
    DROPOUT,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def generate_caption(tensor):
    captions = beam_search_generate(
        model,
        tensor,
        tokenizer,
        device=DEVICE,
        max_length=MAX_LEN,
        beam_size=BEAM_SIZE,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        diversity_penalty=DIVERSITY_PENALTY,
        stochastic_init_steps=STOCHASTIC_INIT_STEPS,
        init_top_k=INIT_TOP_K,
        init_temperature=INIT_TEMPERATURE,
    )

    return [format_caption(caption) for caption in captions]
