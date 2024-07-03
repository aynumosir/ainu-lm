from .normalizer_ainu import normalize


def test_strip_accents() -> None:
    assert normalize("áéíóú") == "aeiou"
    assert normalize("k=eyáykopuntek") == "k=eyaykopuntek"


def test_remove_linking() -> None:
    assert normalize("a_b_c") == "abc"


def test_remove_redundant_whitespaces() -> None:
    assert normalize("a  b  c") == "a b c"


def test_remove_sakehe_symbol() -> None:
    assert normalize("V foo bar") == "foo bar"
    assert normalize("V1 foo bar") == "foo bar"
    assert normalize("V2 foo bar") == "foo bar"
    assert normalize("V1 V2 foo bar") == "foo bar"

    assert normalize("Ｖ foo bar") == "foo bar"
    assert normalize("Ｖ1 foo bar") == "foo bar"
    assert normalize("Ｖ2 foo bar") == "foo bar"


def test_remove_biratori_annotation() -> None:
    assert normalize("[abc] foo bar") == "foo bar"


def test_remove_aa_ken_annotation() -> None:
    assert normalize("foo bar[1]") == "foo bar"


def test_remove_koshobungei_annotation() -> None:
    assert normalize("*foo bar") == "foo bar"


def test_remove_speaker_annotation() -> None:
    assert normalize("（川上）foo bar") == "foo bar"


def test_remove_glottal_stop_before_prefix() -> None:
    assert normalize("arikiki'=an") == "arikiki=an"
    assert normalize("inkar=’an") == "inkar=an"


def test_ainugo_archive_annotation() -> None:
    assert normalize("foo bar［注］") == "foo bar"


def test_does_not_affect_normal_text() -> None:
    assert (
        normalize("onkami=an kane soyenpa=an hine") == "onkami=an kane soyenpa=an hine"
    )
