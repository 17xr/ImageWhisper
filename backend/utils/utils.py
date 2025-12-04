import torch
import torch.nn.functional as F


def nucleus_sampling_generate(
    model,
    image,
    tokenizer,
    device,
    num_captions,
    max_length,
    temperature,
    top_k,
    top_p,
    no_repeat_ngram_size,
    repetition_penalty,
):
    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id

    image_features = image.to(device)
    if image_features.dim() == 2:
        image_features = image_features.unsqueeze(0)
    image_features = image_features.expand(num_captions, -1, -1, -1)

    curr_seq = torch.full(
        (num_captions, 1), start_token, device=device, dtype=torch.long
    )
    unfinished_sequences = torch.ones(num_captions, dtype=torch.long, device=device)

    for step in range(max_length):
        with torch.no_grad():
            logits = model(image_features, curr_seq)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if repetition_penalty > 1.0:
            score = torch.gather(next_token_logits, 1, curr_seq)
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            next_token_logits.scatter_(1, curr_seq, score)

        if no_repeat_ngram_size > 0 and curr_seq.size(1) >= no_repeat_ngram_size - 1:
            for idx in range(num_captions):
                if unfinished_sequences[idx] == 0:
                    continue
                current_tokens = curr_seq[idx].tolist()
                check_ngram = tuple(current_tokens[-(no_repeat_ngram_size - 1) :])
                for i in range(len(current_tokens) - no_repeat_ngram_size + 1):
                    if (
                        tuple(current_tokens[i : i + no_repeat_ngram_size - 1])
                        == check_ngram
                    ):
                        next_token_logits[
                            idx, current_tokens[i + no_repeat_ngram_size - 1]
                        ] = -float("inf")

        if top_k > 0:
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < top_k_values[..., -1, None]] = -float(
                "inf"
            )

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = -float("inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + end_token * (
            1 - unfinished_sequences.unsqueeze(1)
        )
        curr_seq = torch.cat([curr_seq, next_tokens], dim=1)
        unfinished_sequences = unfinished_sequences.mul(
            (next_tokens.squeeze() != end_token).long()
        )

        if unfinished_sequences.max() == 0:
            break

    generated_captions = [
        tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in curr_seq
    ]
    return generated_captions


def format_caption(caption):
    caption = caption.strip().capitalize()
    return caption if caption.endswith(".") else caption + "."
