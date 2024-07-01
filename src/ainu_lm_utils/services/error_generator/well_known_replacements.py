from .confusion_set import confusion_set
from .tagged_word import TaggedWord


class WellKnownReplacements:  # consider a better name
    mapper = confusion_set

    def get(self, tagged_word: TaggedWord) -> str | None:
        for tagged_word_, replacement in self.mapper:
            if tagged_word == tagged_word_:
                return replacement
        return None
