def register():
    """Register the HPU platform."""

    return "vllm_hpu.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    pass
    #    import vllm_hpu.ops  # noqa: F401
