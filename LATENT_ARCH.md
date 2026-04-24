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
  LatentFormer adds a learned signature head for each query, so queries live in a latent embedding space rather than acting only as direct mask predictors. Each signature represents a unique object in the image; different objects have different signatures, and queries predicting the same object have similar signatures.

- **Ground-truth encoder**
  During training, LatentFormer encodes each GT instance into a latent signature using class, box, and mask-context information. This gives the model an explicit target representation for the decoder to learn. 
  
  A new GT signature separation loss is used to ensure object representations don't collapse.

- **Prototype aggregation**
  Final class and mask predictions are produced by aggregating multiple query outputs into object signatures, so each query can contribute to multiple objects, and each object can be made up of different queries. During training the GT signatures are used as object signatures, while during inference certain queries are selected ("seed queries") to represent objects.
  
- **Seed-based object formation**
  At inference time, LatentFormer selects high-confidence "seed queries", merges duplicate seeds in signature space, and uses the surviving seeds as object signatures for final predictions. Conceptually, this seed merging plays a role similar to non-maximum suppression (NMS), but in latent signature space rather than directly in box or mask space.

  These seed queries are learned with two new losses: a hungarian-based loss for a new "seed head", and a signature alignment loss that encourages signatures of "seed queries" to become aligned with those of the GT signatures.

- **Latent-space matching**
  Instead of matching queries to GT mainly in prediction space -- with class and mask costs -- LatentFormer matches query signatures to GT signatures in latent space. This also speeds up training due to avoiding the computation of the pairwise mask costs between each prediction and each GT.

- **Native interpretability**
  Because final predictions are formed through explicit aggregation in latent space, LatentFormer is naturally more interpretable. For a given pixel, we can inspect which queries contribute most strongly to its prediction, including both positive and negative contributions. This also incentivizes query specialization: some queries appear to focus on borders, others on class-specific evidence, others on region-level grouping, and others on harder cases such as areas where multiple objects overlap.

- **Competitive mask assignment**
  While Mask2Former trains binary masks for each object independently, the slot-based nature of LatentFormer allows to generate masks with softmax in a competitive manner. Each object competes to explain each pixel in the image, which accelerates training.

  This is only possible due to the nature of LatentFormer. If you tried to use softmax for mask prediction in Mask2Former, you would find that hungarian assignment cannot be computed: the prediction of each query depends on which other queries you select, making it an optimization problem that cannot be efficiently solved. As LatentFormer completely separates mask losses (what object each pixel belongs to) from seed losses (which queries most faithfully represent an object), this is not a problem.

- **Soft attention biasing**
  LatentFormer keeps the Mask2Former-style idea that mask predictions guide later cross-attention, but it changes the mechanism. Standard Mask2Former converts predicted masks into a detached boolean attention mask, so masked-out pixels are disallowed and gradients do not flow through that masking decision. LatentFormer supervises masks with softmax competition rather than independent sigmoid binary masks, so an individual query prediction is not naturally a hard foreground mask. LatentFormer therefore normalizes predicted mask logits and passes them as a continuous attention bias. This keeps attention guidance differentiable and lets latent components softly prefer or avoid spatial evidence, which fits the cooperative query aggregation design better than a hard per-query binary gate.

### Intuition

A useful way to think about the difference is:

- **Mask2Former:** one query tries to become one object.
- **LatentFormer:** many queries can cooperate to represent one object, and the model learns how to group them through latent signatures and seed selection.

This makes LatentFormer less of a direct query-to-instance predictor and more of a latent grouping architecture built on top of the Mask2Former framework.
