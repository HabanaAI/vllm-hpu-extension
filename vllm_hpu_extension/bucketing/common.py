
import os
from typing import Dict
import inspect

class Singleton(type):
    """
    A metaclass that creates a Singleton instance. This ensures that only one instance of the class exists.

    Attributes:
        _instances (Dict[type, object]): A dictionary to store the single instance of each class.
        _instances_argspec (Dict[type, object]): A dictionary to store the argument specifications of each instance.

    Methods:
        __call__(cls, *args, **kwargs):
            Creates or returns the single instance of the class. If the instance already exists, it checks that the 
            arguments used to create the instance are the same as the ones provided. Raises an assertion error if 
            the arguments differ.
    """
    _instances: Dict[type, object] = {}
    _instances_argspec: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        argspec = inspect.getcallargs(super().__call__, args, kwargs)
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
            cls._instances_argspec[cls] = argspec
        assert cls._instances_argspec[cls] == argspec, "Singleton instance already initialized with different arguments"
        return cls._instances[cls]

def get_bucketing_context():
    use_exponential_bucketing = os.environ.get(
        'VLLM_EXPONENTIAL_BUCKETING', 'false').lower() == 'true'
    if use_exponential_bucketing:
        from vllm_hpu_extension.bucketing.exponential import (
            HPUExponentialBucketingContext as HPUBucketingContext)
    else:
        from vllm_hpu_extension.bucketing.linear import HPUBucketingContext

    return HPUBucketingContext