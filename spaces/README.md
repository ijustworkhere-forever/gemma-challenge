# Spaces

Exploratory Hugging Face Spaces used for free or low-cost smoke testing.

## `zerogpu_smoke`

Use Gradio when creating this Space. Hugging Face ZeroGPU is currently tied to
Gradio Spaces, not Docker or Static Spaces.

This Space checks whether a GPU-decorated function can execute and reports the
visible device. It is not a contest benchmark and should not be compared with
official `hf_a10g` results.
