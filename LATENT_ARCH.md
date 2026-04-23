## How LatentFormer Differs from Standard Mask2Former

LatentFormer keeps the same high-level backbone-to-segmentation structure as Mask2Former, but changes what decoder queries represent and how final object predictions are formed.

In standard Mask2Former, each decoder query is directly treated as an object hypothesis: it predicts a class distribution and a binary mask, and training uses Hungarian matching between predicted queries and ground-truth instances.

LatentFormer instead treats decoder queries as latent components. Each query predicts:
- class logits
- mask embeddings
- a latent signature
- a seed score

These query-level outputs are not used as final object predictions directly. Instead, LatentFormer groups and aggregates queries into object-level prototypes using signature similarity.

### Key architectural differences

- **Latent signature space**
  LatentFormer adds a learned signature head for each query, so queries live in a latent embedding space rather than acting only as direct mask predictors.

- **Seed-based object formation**
  At inference time, LatentFormer selects high-confidence seed queries, merges duplicate seeds in signature space, and uses the surviving seeds as anchors for final predictions. Conceptually, this seed merging plays a role similar to non-maximum suppression (NMS), but in latent signature space rather than directly in box or mask space.

- **Prototype aggregation**
  Final class and mask predictions are produced by aggregating multiple query outputs into seed-conditioned prototypes, rather than reading predictions from individual queries one-to-one.

- **Native interpretability**
  Because final predictions are formed through explicit aggregation in latent space, LatentFormer is naturally more interpretable. For a given predicted region, we can inspect which queries contribute most strongly to it, including both positive and negative contributions at the pixel level. This also makes query specialization easier to study: some queries appear to focus on borders, others on class-specific evidence, others on region-level grouping, and others on harder cases such as areas where multiple objects overlap.

- **Ground-truth encoder**
  During training, LatentFormer encodes each GT instance into a latent signature using class, box, and mask-context information. This gives the model an explicit target representation in latent space.

- **Latent-space matching**
  Instead of matching queries to GT mainly with class and mask costs, LatentFormer matches query signatures to GT signatures, optionally incorporating seed confidence. In that sense, the model tries to perform both instance matching during training and NMS-like duplicate suppression during inference in the same signature space.

- **Additional training objectives**
  Beyond classification and mask losses, LatentFormer introduces:
  - seed prediction loss
  - signature alignment loss
  - GT signature separation loss

- **Competitive mask assignment**
  LatentFormer treats prototype masks more like a partition over latent slots, using soft competition across prototypes, instead of independent per-query binary masks.

- **Soft attention biasing**
  LatentFormer keeps the Mask2Former-style idea that mask predictions guide later cross-attention, but it changes the mechanism. Standard Mask2Former converts predicted masks into a detached boolean attention mask, so masked-out pixels are disallowed and gradients do not flow through that masking decision. LatentFormer supervises masks with softmax competition across latent slots rather than independent sigmoid binary masks, so an individual query prediction is not naturally a standalone hard foreground mask. LatentFormer therefore normalizes predicted mask logits and passes them as a continuous attention bias. This keeps attention guidance differentiable and lets latent components softly prefer or avoid spatial evidence, which fits the cooperative query aggregation design better than a hard per-query binary gate.

### Intuition

A useful way to think about the difference is:

- **Mask2Former:** one query tries to become one object.
- **LatentFormer:** many queries can cooperate to represent one object, and the model learns how to group them through latent signatures and seed selection.

This makes LatentFormer less of a direct query-to-instance predictor and more of a latent grouping architecture built on top of the Mask2Former framework.
