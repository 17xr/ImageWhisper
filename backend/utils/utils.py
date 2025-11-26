import torch
import torch.nn.functional as F


def nucleus_sampling_generate(
    model,
    image,
    tokenizer,
    device,
    num_captions=5,
    max_length=64,
    temperature=0.7,
    top_k=30,
    top_p=0.9,
    no_repeat_ngram_size=0,
    repetition_penalty=1.2,
):
    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id

    with torch.no_grad():
        image_features = image.to(device)

        target_shape = list(image_features.shape)
        target_shape[0] = num_captions
        image_features = image_features.expand(*target_shape)

    curr_seq = torch.full(
        (num_captions, 1), start_token, device=device, dtype=torch.long
    )
    curr_pm = torch.ones(num_captions, 1, device=device, dtype=torch.bool)
    unfinished_sequences = torch.ones(num_captions, dtype=torch.long, device=device)

    for step in range(max_length):
        L = curr_seq.size(1)
        attn_mask = torch.tril(torch.ones((num_captions, L, L), device=device))

        with torch.no_grad():
            logits = model(image_features, curr_seq, attn_mask, curr_pm)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if repetition_penalty > 1.0:
            score = torch.gather(next_token_logits, 1, curr_seq)
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            next_token_logits.scatter_(1, curr_seq, score)

        if no_repeat_ngram_size > 0 and L >= no_repeat_ngram_size - 1:
            for idx in range(num_captions):
                current_tokens = curr_seq[idx].tolist()
                check_ngram = tuple(current_tokens[-(no_repeat_ngram_size - 1) :])

                for i in range(len(current_tokens) - no_repeat_ngram_size + 1):
                    if (
                        tuple(current_tokens[i : i + no_repeat_ngram_size - 1])
                        == check_ngram
                    ):
                        banned_token = current_tokens[i + no_repeat_ngram_size - 1]
                        next_token_logits[idx, banned_token] = -float("inf")

        if top_k > 0:
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            min_values = top_k_values[..., -1, None]
            top_k_mask = next_token_logits < min_values
            next_token_logits[top_k_mask] = -float("inf")

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

            top_p_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            top_p_mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[top_p_mask] = -float("inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + end_token * (
            1 - unfinished_sequences.unsqueeze(1)
        )

        curr_seq = torch.cat([curr_seq, next_tokens], dim=1)
        curr_pm = torch.cat(
            [curr_pm, torch.ones(num_captions, 1, dtype=torch.bool, device=device)],
            dim=1,
        )

        unfinished_sequences = unfinished_sequences.mul(
            (next_tokens.squeeze() != end_token).long()
        )

        if unfinished_sequences.max() == 0:
            break

    generated_captions = []
    for seq in curr_seq:
        caption = tokenizer.decode(seq, skip_special_tokens=True).strip()
        generated_captions.append(caption)

    return generated_captions


def format_caption(caption):
    caption = caption.strip().capitalize()
    return caption if caption.endswith(".") else caption + "."
