"""Pick the compute device: Apple GPU (MPS), NVIDIA GPU (CUDA) or CPU."""
import torch


def resolve_device(name: str = "auto") -> torch.device:
    """name: 'auto' (MPS > CUDA > CPU), 'mps', 'cuda' or 'cpu'.

    An explicitly requested GPU that is not available falls back to the CPU with a warning,
    so a typo or a non-Apple-Silicon Python never crashes a long run.
    """
    name = name.lower()
    have_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    have_cuda = torch.cuda.is_available()
    if name == "auto":
        return torch.device("mps" if have_mps else "cuda" if have_cuda else "cpu")
    if name == "mps" and not have_mps:
        print("  WARNING: MPS (Apple GPU) is not available in this Python; using CPU. "
              "(Is Python running natively on Apple Silicon, i.e. platform.machine() == 'arm64'?)")
        return torch.device("cpu")
    if name == "cuda" and not have_cuda:
        print("  WARNING: CUDA is not available; using CPU.")
        return torch.device("cpu")
    if name not in ("cpu", "mps", "cuda"):
        raise ValueError(f"unknown device {name!r}; choose auto, cpu, mps or cuda")
    return torch.device(name)


def describe(dev: torch.device) -> str:
    if dev.type == "mps":
        return "mps (Apple GPU via Metal)"
    if dev.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(dev)})"
    return f"cpu ({torch.get_num_threads()} threads)"


def synchronize(dev: torch.device):
    """Wait for queued GPU work (needed for honest timings)."""
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()
