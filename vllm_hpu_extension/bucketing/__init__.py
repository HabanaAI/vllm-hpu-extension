
def get_bucketing_context():
    use_exponential_bucketing = os.environ.get(
        'VLLM_EXPONENTIAL_BUCKETING', 'false').lower() == 'true'
    if use_exponential_bucketing:
        from vllm_hpu_extension.bucketing.exponential import (
            HPUExponentialBucketingContext as HPUBucketingContext)
    else:
        from vllm_hpu_extension.bucketing.linear import HPUBucketingContext

    return HPUBucketingContext.get_instance()