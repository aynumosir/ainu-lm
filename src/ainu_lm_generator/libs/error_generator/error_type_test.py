import random

from .error_type import ErrorType


def test_random_sample() -> None:
    random.seed(42)
    error_type = ErrorType.random()
    assert error_type == ErrorType.REPLACE
