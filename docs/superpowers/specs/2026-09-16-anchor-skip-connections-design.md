# Round 8: residual skip connections that inject the anchor into every decoder layer

**Date:** 2026-09-16
**Status:** agreed in conversation, being implemented

## Why

Round 7's anchored RSSM predicts `anchor + head(decoder([h, s]))`. The anchor
(the latest window day of the same drive/layup type) enters only at the
output, so the decoder's correction cannot depend on it: the same belief
state produces the same correction whether the anchor says 43 C or 13 C.
Injecting the anchor into every hidden layer lets the correction be
conditioned on the value it corrects.

Chosen in conversation over two alternatives (real-data MAE latents; belief
state re-injected per layer). Sim-only; no real data involved.

## Model (`experiment/rssm.py`)

New options, RSSM only (gru_vae stayed below the baseline with anchoring in
round 7 and is not the selected model):

- `decoder_layers` (default 1, the round 5-7 decoder).
- `anchor_skip` (default False; requires `anchored=True`).

Decoder with `L = decoder_layers` hidden layers:

    x_0 = [h_t, s_t]
    x_l = ELU(W_l x_{l-1} + P_l anchor_t)          l = 1..L   (P_l only if anchor_skip)
    obs_t = anchor_t + head(x_L)                    head zero-initialised

`P_l` are linear projections of the anchor (n_obs -> hidden). With the head
zero-initialised, an untrained model still reproduces the anchor exactly.
Filtering reconstruction uses the in-window anchors the same way, so
filtering and imagination decode through identical paths.

With `anchor_skip=False` and `decoder_layers=1` the decoder is parameter-for-
parameter the round 7 one, and loading a round 7 checkpoint works unchanged.

## Comparison (same protocol as round 7)

Dev split: train -10/10 C, select on 25 C, 42 C untouched.

| Variant | decoder_layers | anchor_skip |
|---|---|---|
| round 7 (reference) | 1 | no |
| depth control | 2 | no |
| **anchor skips** | 2 | yes |

The depth control separates "the skips help" from "a deeper decoder helps".
If the skip variant wins on dev factual and what-if tolerance accuracy, it is
trained on -10/10/25 C and scored on 42 C once. This is the second use of
the 42 C test set (round 7 was the first); both numbers are reported.
One seed per variant, stated as a limitation.
