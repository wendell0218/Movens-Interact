# Third-Party Notices

## Movens-Zero

The body expert, Qwen runtime, root40 representation, normalization, and progress
encoding are adapted from [Movens-Zero](https://github.com/wendell0218/Movens-Zero),
commit `5306ce203eb18db4628850e323a980d9ba6422d0`. This release packages the
runtime with an independent hand expert and manipulation training and simulation
entry points.

The Movens-Zero XL checkpoint is downloaded from its
[original model repository](https://huggingface.co/wendell0218/Movens-Zero-XL).
It is not included in the Git repository or the adapter release files.

## Qwen2.5-Omni-3B

The frozen multimodal backbone uses
[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B).
Its weights, tokenizer, processor, and configuration remain subject to the
[Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE).
The download helper retrieves processor files from the upstream repository.
Movens-Interact adapter files contain the adapted body expert and hand expert,
not the Qwen weights.

## Robot and Scene Assets

The simulation uses G1/Dex3 robot geometry and apple, plate, and table assets
referenced by [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
commit `b5040e20dea6871929d94f2e9d6fe0df298024ba`.

Binary meshes and textures are fetched from upstream rather than redistributed
in Git. File origins, revisions, sizes, and checksums are listed in
[`assets/manifest.json`](assets/manifest.json). See
[`assets/NOTICE.md`](assets/NOTICE.md) and the upstream
[license](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b5040e20dea6871929d94f2e9d6fe0df298024ba/LICENSE)
and
[third-party notices](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b5040e20dea6871929d94f2e9d6fe0df298024ba/legal).
The source-code license of a containing repository does not establish a separate
license for every referenced mesh or texture.

## Apple Demonstrations

Data preparation downloads the selected episode and metadata from
[GR00T-N1.7-AppleToPlate](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate),
revision `d89c126a713c6632432a607c12661546ff4d6ea9`.
Dataset files remain subject to the upstream dataset terms and are not
redistributed in this repository.
