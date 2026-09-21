# Use these models with gem-x.cpp

These custom GGUF models require the gem-x.cpp runtime. They do not run in
llama.cpp. Build the runtime and browser demo using your source checkout's
README instructions, then download the models from its repository root:

```sh
hf download LocalAI-io/GEM-X-GGUF gem-x-contact-f32.gguf vitpose-f32.gguf yolox-f32.gguf --local-dir generated/reference
./demo/gemx-demo --threads 8
```

Open http://localhost:8098 and choose **Live webcam**, then **Start live**.
Keep the camera still and your full body in view.

Offline video additionally requires the SAM3D submodule's Body worker and
three model files. Build the worker as described in the runtime's demo guide,
then download its models to the default locations:

```sh
hf download LocalAI-io/sam-3d-body-dinov3-GGUF body-dinov3-f32.gguf body-pose-branch-f32.gguf --local-dir sam3d.cpp/generated/models/sam-3d-body-dinov3
hf download LocalAI-io/sam-3d-body-dinov3-GGUF mhr-lod1-f32.gguf --local-dir sam3d.cpp/generated/models/mhr-public
```

Choose **Offline video**, select a clip and press **Build motion**. Completed
motion can be downloaded as an animated skeleton GLB.

`hf` is supplied by the optional `huggingface_hub` Python package. Python is
only needed to download or convert models, not for the native runtime. All
components retain the licenses documented in their model repositories.
