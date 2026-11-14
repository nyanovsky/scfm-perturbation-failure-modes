"""The protein-language-model control.

Two questions the main sweep left open, and the code that answers them:

``audit``       where each model's static gene-identity table came from — was it
                updated during pretraining, or is it still the vectors it was
                initialised with?
``uniprot``     gene -> canonical human protein sequence, by a recorded route.
``esm_embed``   those sequences through a public ESM2 checkpoint, giving a gene
                table with no single-cell model anywhere in its path.
``identity``    how close each model's table is to the raw protein table.

The raw table then goes through the same translator, splits, scoring and nulls
as every other arm, which is the only way to say whether the provenance matters
for this task.
"""
