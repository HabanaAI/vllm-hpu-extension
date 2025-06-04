def register():
    """Register the HPU platform."""

    return "vllm_hpu.platform.HpuPlatform"

def register_ops():
    """Register custom ops for the HPU platform."""

    from vllm_hpu.ops.hpu_fused_moe import HPUUnquantizedFusedMoEMethod # noqa: F401