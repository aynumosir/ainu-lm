class TaggedWord:
    word: str
    pos: str
    has_leading_whitespace: bool

    def __init__(self, word: str, pos: str) -> None:
        if word.startswith(" "):
            self.has_leading_whitespace = True
            word = word[1:]
        else:
            self.has_leading_whitespace = False

        self.word = word
        self.pos = pos

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaggedWord):
            return NotImplemented
        return self.word == other.word and self.pos == other.pos

    def __str__(self) -> str:
        if self.has_leading_whitespace:
            return f" {self.word}"
        else:
            return self.word

    def __hash__(self) -> int:
        return hash((self.word, self.pos))
