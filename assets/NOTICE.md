# Simulation assets

The scene is adapted from the G1 and RoboCasa models in [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b5040e20dea6871929d94f2e9d6fe0df298024ba), copyright NVIDIA Corporation and affiliates. Adaptations include fixed-base position control, task-object placement and an approximate torso camera.

The upstream repository declares [Apache License 2.0 for source code](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b5040e20dea6871929d94f2e9d6fe0df298024ba/LICENSE). Its [third-party notices](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b5040e20dea6871929d94f2e9d6fe0df298024ba/legal/THIRD-PARTY%20SOFTWARE%20NOTICES%20AND%20ASSET%20LICENSES%20-%20GEAR-SONIC.txt) identify software components, but do not provide a separate per-file license mapping for these G1 and Omniverse meshes and textures. Consult the upstream terms for those assets.

Binary mesh and texture files are excluded from this repository. `manifest.json` lists the 63 required files, pinned public download URLs, sizes and SHA256 digests. `simulation/prepare_assets.py` downloads them directly from upstream or reuses a verified local cache. The vendor directory contains local runtime assets only.
