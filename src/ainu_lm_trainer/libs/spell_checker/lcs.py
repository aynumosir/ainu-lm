from difflib import SequenceMatcher


# https://stackoverflow.com/a/39404777
def lcs(a: str, b: str) -> str:
    matcher = SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return a[match.a : match.a + match.size]
