# Round 10: an attention-picked anchor

**Date:** 2026-09-16
**Status:** agreed in conversation, being implemented

## Why

Round 7 predicts `anchor + correction`, where the anchor is a fixed rule: the
latest window day whose drive/layup type matches the day being predicted.
That rule is doing most of the work (it alone scores 83.3% on the unseen
42 C test), and the learned part only adjusts it.

Round 8 showed what breaks: injecting the anchor's RAW values into hidden
layers let the correction key on absolute values outside the training range,
and unseen-climate accuracy collapsed from 87.5% to 38.4% on dev.

Attention avoids that failure by construction. If the anchor is a convex
combination of days the model actually observed in this window, every anchor
value lies inside the range of that battery's own recent history, whatever
the climate. The model chooses WHICH days to copy; it never invents a value.

## Model (`experiment/attention.py`, RSSM flag `--attn-anchor`)

For predicted day `t`:

    query_i  = W_q [action_t, h_t]                    (day t's schedule and belief)
    key_i    = W_k [action_i]                         (window day i's schedule)
    score_i  = query . key / sqrt(d)
             + beta  * 1[type_i == type_t]            type-match bonus
             + gamma * i / (L - 1)                    recency bonus
    a        = softmax(score)                         (masked; see below)
    anchor_t = sum_i a_i * observable_i               convex combination
    pred_t   = anchor_t + head(decoder([h_t, s_t]))   head zero-initialised

`beta` (init 20) and `gamma` (init 60) are learnable scalars, and `W_q`,
`W_k` are zero-initialised. So an untrained model puts almost all weight on
the latest same-type day -- round 7's anchor -- and, with the zero head,
reproduces round 7's starting point. Training can only refine which days get
copied.

Masking: rolling out day `t` may attend over the whole observed window.
Reconstructing window day `t` during filtering may attend only over days
`< t` (causal), matching `window_anchors`' rule, so filtering never peeks.

Counterfactual schedules change `action_t`, so the anchor follows the
counterfactual day types exactly as the fixed rule did.

## Protocol

Dev split only (train -10/10 C, score 25 C), compared against round 7
(87.5% / 65.8%), the depth control (86.6% / 67.4%) and the baseline
(83.5% / 60.5%). 42 C is run only if this wins on dev, and would be its
third use -- reported as such.

Noted risk, carried from round 8's verdict: the 25 C dev split has now
selected among twelve variants. If the attention anchor wins narrowly, the
result should be confirmed with leave-one-climate-out rather than trusted
from this split alone.
