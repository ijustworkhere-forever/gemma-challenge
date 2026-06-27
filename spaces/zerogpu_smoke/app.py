import time

import gradio as gr

try:
    import spaces
except ImportError:
    spaces = None


def gpu_decorator(fn):
    if spaces is None:
        return fn
    return spaces.GPU(duration=30)(fn)


@gpu_decorator
def run_smoke_test():
    start = time.time()

    try:
        import torch
    except ImportError:
        return "torch is not installed in this Space."

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    x = torch.randn((512, 512), device=device)
    y = x @ x

    if cuda_available:
        torch.cuda.synchronize()
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        device_name = "CPU fallback"
        memory_gb = 0.0

    elapsed = time.time() - start
    checksum = float(y[0, 0].detach().cpu())

    return (
        f"device={device}\n"
        f"name={device_name}\n"
        f"memory_gb={memory_gb:.2f}\n"
        f"elapsed_sec={elapsed:.4f}\n"
        f"checksum={checksum:.6f}"
    )


app = gr.Interface(
    fn=run_smoke_test,
    inputs=None,
    outputs=gr.Textbox(label="Result", lines=8),
    title="Gemma Challenge ZeroGPU Smoke",
)


if __name__ == "__main__":
    app.launch()
