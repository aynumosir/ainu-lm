# fmt: off
# cspell: disable
from .tagged_word import TaggedWord

irregular_plural_intransitive_verb = {
    (TaggedWord("an", "VI"), "oka"),
    (TaggedWord("oka", "VI"), "an"),

    (TaggedWord("as", "VI"), "roski"),
    (TaggedWord("roski", "VI"), "as"),

    (TaggedWord("a", "VI"), "rok"),
    (TaggedWord("rok", "VI"), "a"),

    (TaggedWord("arpa", "VI"), "paye"),
    (TaggedWord("paye", "VI"), "arpa"),

    (TaggedWord("ek", "VI"), "arki"),
    (TaggedWord("arki", "VI"), "ek"),

    (TaggedWord("omanan", "VI"), "payeoka"),
    (TaggedWord("payeoka", "VI"), "omanan"),

    (TaggedWord("rikin", "VI"), "rikip"),
    (TaggedWord("rikip", "VI"), "rikin"),

    (TaggedWord("ran", "VI"), "rap"),
    (TaggedWord("rap", "VI"), "ran"),

    (TaggedWord("san", "VI"), "sap"),
    (TaggedWord("sap", "VI"), "san"),

    (TaggedWord("yan", "VI"), "yap"),
    (TaggedWord("yap", "VI"), "yan"),

    (TaggedWord("ahun", "VI"), "ahup"),
    (TaggedWord("ahup", "VI"), "ahun"),
}

irregular_plural_transitive_verb = {
    (TaggedWord("are", "VT"), "rokte"),
    (TaggedWord("rokte", "VT"), "are"),

    (TaggedWord("arpare", "VT"), "payere"),
    (TaggedWord("payere", "VT"), "arpare"),

    (TaggedWord("ekte", "VT"), "arkire"),
    (TaggedWord("arkire", "VT"), "ekte"),

    (TaggedWord("asi", "VT"), "roski"),
    # (TaggedWord("roski", "VT"), "asi"),

    (TaggedWord("rayke", "VT"), "ronnu"),
    (TaggedWord("ronnu", "VT"), "rayke"),

    (TaggedWord("rikinka", "VT"), "rikipte"),
    (TaggedWord("rikipte", "VT"), "rikinka"),

    (TaggedWord("ranke", "VT"), "rapte"),
    (TaggedWord("rapte", "VT"), "ranke"),

    (TaggedWord("sanke", "VT"), "sapte"),
    (TaggedWord("sapte", "VT"), "sanke"),

    (TaggedWord("yanke", "VT"), "yapte"),
    (TaggedWord("yapte", "VT"), "yanke"),

    (TaggedWord("ahunke", "VT"), "ahupte"),
    (TaggedWord("ahupte", "VT"), "ahunke"),
}

confusion_set = irregular_plural_intransitive_verb | irregular_plural_transitive_verb
