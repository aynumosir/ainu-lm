from .leven import leven


def test_compute_levenshtein_distance() -> None:
    distance = leven("kitten", "sitting")
    assert distance == 3
