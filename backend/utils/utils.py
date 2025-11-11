import torch
import torch.nn.functional as F


def beam_search_generate(
    model,
    image,
    tokenizer,
    device,
    max_length=64,
    beam_size=4,
    no_repeat_ngram_size=2,
    temperature=1.5,
    top_k=100,
    diversity_penalty=1.0,
    stochastic_init_steps=3,
    init_top_k=30,
    init_temperature=1.4,
):
    model.eval()

    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id

    with torch.no_grad():
        image_features = image.to(device)

    init_seq = torch.tensor([[start_token]], device=device)
    init_kpm = torch.ones(1, 1, dtype=torch.bool, device=device)

    beams = [{"seq": init_seq, "score": 0.0, "kpm": init_kpm, "ngrams": set()}]

    for _ in range(stochastic_init_steps):
        new_beams = []
        for b in beams:
            with torch.no_grad():
                logits = model(image_features, b["seq"], b["kpm"], b["kpm"])
                log_probs = F.log_softmax(
                    logits[0, -1] / max(init_temperature, 1e-8), dim=-1
                )
            top_vals, top_idx = torch.topk(log_probs, init_top_k)
            probs = top_vals.exp()
            probs /= probs.sum()
            sampled = torch.multinomial(probs, beam_size, replacement=False)

            for s_idx in sampled.tolist():
                token_id = int(top_idx[s_idx].item())
                token_logprob = float(top_vals[s_idx].item())
                new_seq = torch.cat(
                    [b["seq"], torch.tensor([[token_id]], device=device)], dim=1
                )
                new_kpm = torch.cat(
                    [b["kpm"], torch.ones(1, 1, dtype=torch.bool, device=device)], dim=1
                )
                new_ngrams = set(b["ngrams"])
                if (
                    no_repeat_ngram_size > 0
                    and len(b["seq"][0]) >= no_repeat_ngram_size - 1
                ):
                    new_ngrams.add(
                        tuple(
                            b["seq"][0, -no_repeat_ngram_size + 1 :].tolist()
                            + [token_id]
                        )
                    )
                new_beams.append(
                    {
                        "seq": new_seq,
                        "score": b["score"] + token_logprob,
                        "kpm": new_kpm,
                        "ngrams": new_ngrams,
                    }
                )

        new_beams.sort(key=lambda x: x["score"] / x["seq"].size(1), reverse=True)
        beams = new_beams[:beam_size]

    completed = []
    for step in range(stochastic_init_steps + 1, max_length):
        if not beams:
            break

        all_candidates = []
        proposal_counts = {}
        beam_proposals = []

        for b in beams:
            last_token = b["seq"][0, -1].item()
            if last_token == end_token:
                completed.append(b)
                continue

            L = b["seq"].size(1)
            attn_mask = torch.tril(torch.ones((1, L, L), device=device))

            with torch.no_grad():
                logits = model(image_features, b["seq"], attn_mask, b["kpm"])
                log_probs = F.log_softmax(
                    logits[0, -1] / max(temperature, 1e-8), dim=-1
                )

            top_k_val = min(top_k, log_probs.size(-1))
            top_vals, top_idx = torch.topk(log_probs, top_k_val)
            beam_proposals.append((b, top_vals, top_idx))

            for tok in top_idx.tolist():
                proposal_counts[tok] = proposal_counts.get(tok, 0) + 1

        if not beam_proposals:
            break

        for b, top_vals, top_idx in beam_proposals:
            current_tokens = b["seq"][0].tolist()
            penalty = torch.tensor(
                [proposal_counts[tok.item()] for tok in top_idx],
                device=device,
                dtype=top_vals.dtype,
            )
            adjusted_logits = top_vals - diversity_penalty * penalty
            probs = F.softmax(adjusted_logits, dim=-1)

            num_samples = min(beam_size, probs.size(0))
            sampled_indices = torch.multinomial(probs, num_samples, replacement=False)

            for s_idx in sampled_indices.tolist():
                token_id = int(top_idx[s_idx].item())
                token_logprob = float(top_vals[s_idx].item())

                block = False
                if (
                    no_repeat_ngram_size > 0
                    and len(current_tokens) >= no_repeat_ngram_size - 1
                ):
                    prev = tuple(
                        current_tokens[-(no_repeat_ngram_size - 1) :] + [token_id]
                    )
                    if prev in b["ngrams"]:
                        block = True
                if block:
                    continue

                new_seq = torch.cat(
                    [b["seq"], torch.tensor([[token_id]], device=device)], dim=1
                )
                new_kpm = torch.cat(
                    [b["kpm"], torch.ones(1, 1, dtype=torch.bool, device=device)], dim=1
                )
                new_score = b["score"] + token_logprob

                new_ngrams = set(b["ngrams"])
                if (
                    no_repeat_ngram_size > 0
                    and len(current_tokens) >= no_repeat_ngram_size - 1
                ):
                    new_ngrams.add(
                        tuple(
                            current_tokens[-(no_repeat_ngram_size - 1) :] + [token_id]
                        )
                    )

                all_candidates.append(
                    {
                        "seq": new_seq,
                        "score": new_score,
                        "kpm": new_kpm,
                        "ngrams": new_ngrams,
                    }
                )

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x["score"] / x["seq"].size(1), reverse=True)
        beams = all_candidates[:beam_size]

    completed.extend(beams)
    completed.sort(key=lambda x: x["score"] / x["seq"].size(1), reverse=True)

    unique_captions = []
    seen = set()
    for b in completed:
        tokens = b["seq"][0].cpu().numpy()
        caption = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not caption:
            continue
        low = caption.lower()
        if low in seen:
            continue
        seen.add(low)
        unique_captions.append(caption)
        if len(unique_captions) >= beam_size:
            break

    return unique_captions[:beam_size]


def format_caption(caption):
    caption = caption.strip().capitalize()
    return caption if caption.endswith(".") else caption + "."
