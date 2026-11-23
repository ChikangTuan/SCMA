import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from scma.data import SynDeepLesion
from scma.flow import training_loss
from scma.model import VelocityUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument("--mask-dir", type=Path)
    masks.add_argument("--use-simulation-masks", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sigma-max", type=float, default=128.0)
    parser.add_argument("--sigma-min", type=float, default=16.0)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.save_every) < 1 or args.workers < 0 or not (0 < args.lr < float("inf")):
        parser.error("updates, batch size, save interval and lr must be positive; workers must be nonnegative")
    if not (0 <= args.alpha < float("inf") and 0 <= args.beta < float("inf")
            and 0 < args.sigma_min < args.sigma_max < float("inf")):
        parser.error("expected finite nonnegative weighting amplitudes and sigma_max > sigma_min > 0")
    weighting = dict(alpha=args.alpha, beta=args.beta, sigma_max=args.sigma_max, sigma_min=args.sigma_min)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = SynDeepLesion(args.data, split="train", mask_dir=args.mask_dir)
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.batch_size * args.updates,
                            generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    model = VelocityUNet().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Training on {len(dataset)} pairs, {args.updates} updates, {device}", flush=True)
    for step, batch in enumerate(loader, start=1):
        clean, condition, mask, distance = (batch[k].to(device) for k in ("clean", "condition", "mask", "distance"))
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, clean, condition, mask, distance, **weighting)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.updates:
            print(f"update {step}/{args.updates}, weighted velocity MSE {loss.item():.7g}", flush=True)
        if step % args.save_every == 0 or step == args.updates:
            target = args.output / f"scma_{step:06d}.pt"
            temp = target.with_suffix(".partial")
            settings = dict(weighting=weighting, lr=args.lr, batch_size=args.batch_size,
                            seed=args.seed, simulation_masks=args.use_simulation_masks)
            torch.save({"format": "scma-paper-v1", "step": step, "settings": settings,
                        "model": model.state_dict()}, temp)
            os.replace(temp, target)


if __name__ == "__main__":
    main()
