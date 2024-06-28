from .lcs import lcs


def test_lcs() -> None:
    str1 = "kitten"
    str2 = "sitting"

    assert lcs(str1, str2) == "itt"
