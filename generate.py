"""
generate.py -- Text generation inference script

Loads a trained Mamba checkpoint and generates new text token-by-token.
By default, uses the checkpoint created by train.py.

Usage:
    python generate.py --prompt "To be, or not" --max_new 100
"""

import argparse
import torch
import torch.nn.functional as F

from mamba.model import Mamba, MambaConfig


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate text using the model.
    Since this implementation doesn't currently cache the SSM state h_k
    between steps (for simplicity), it performs a full forward pass at
    each step. In a production Mamba inference engine, you would only
    update the state from the last step (O(1) decoding).
    """
    model.eval()
    idx = prompt_ids

    for _ in range(max_new_tokens):
        # Forward pass on the current sequence
        logits, _ = model(idx)

        # Get the logits for the very last token
        logits = logits[:, -1, :] / temperature

        # Optional top-k cropping
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample the next token
        next_idx = torch.multinomial(probs, num_samples=1)

        # Append to the sequence and continue
        idx = torch.cat((idx, next_idx), dim=1)

    return idx


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {args.ckpt}...")

    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    ch2id  = ckpt["ch2id"]
    id2ch  = ckpt["id2ch"]

    model = Mamba(config).to(device)
    model.load_state_dict(ckpt["model"])
    print("Model loaded successfully.\n")

    # Encode prompt
    prompt = args.prompt
    if not prompt:
        prompt = "\n"  # Start with newline if no prompt

    prompt_ids = torch.tensor([[ch2id.get(c, 0) for c in prompt]], dtype=torch.long).to(device)

    # Generate
    out_ids = generate(model, prompt_ids, args.max_new,
                       temperature=args.temperature, top_k=args.top_k)

    # Decode
    out_text = "".join([id2ch[i.item()] for i in out_ids[0]])

    print("─── GENERATED TEXT ──────────────────────────────────────────────")
    print(out_text)
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",      type=str,   default="To be,")
    p.add_argument("--max_new",     type=int,   default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=40)
    p.add_argument("--ckpt",        type=str,   default="mamba_ckpt.pt")
    args = p.parse_args()
    main(args)
