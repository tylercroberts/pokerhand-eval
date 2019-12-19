import re

class InvalidCardError(ValueError):
    pass

class Card:
    """Card represents a card in a standard pack of 52 playing cards."""
    SUIT_TO_STRING = {1: "s", 2: "h", 3: "d", 4: "c"}

    RANK_TO_STRING = {
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "T",
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
    }

    STRING_TO_SUIT = dict([(v, k) for k, v in SUIT_TO_STRING.items()])
    STRING_TO_RANK = dict([(v, k) for k, v in RANK_TO_STRING.items()])

    REPR_RE = re.compile(r"\((.*?)\)")

    def __init__(self, rank, suit):
        """Create a card.

        Arguments:
        First argument: An int or string representing the rank.
        If int, the ranks are 2-14, with 10-14 representing ten through ace.
        If string, "T", "J", "Q", "K", "A" are ten through ace.

        Second argument: An int or string representing the suit.
        If int, 1-4 represent spades, hearts, diamonds, clubs respectively.
        If string, the suits are "s", "h", "d", "c".
        """
        if isinstance(rank, int):
            if rank < 2 or rank > 14:
                raise InvalidCardError("%s is not a valid rank" % rank)
            self.rank = rank
        elif isinstance(rank, "".__class__):
            try:
                self.rank = self.STRING_TO_RANK[rank.upper()]
            except KeyError:
                raise InvalidCardError("'%s' is not a valid rank" % rank)
        else:
            raise TypeError("rank must be int or string")

        if isinstance(suit, int):
            if suit < 1 or suit > 4:
                raise InvalidCardError("%s is not a valid suit" % suit)
            self.suit = suit
        elif isinstance(suit, "".__class__):
            try:
                self.suit = self.STRING_TO_SUIT[suit.lower()]
            except KeyError:
                raise InvalidCardError("'%s' is not a valid suit" % suit)
        else:
            raise TypeError("suit must be int or string")

    def __repr__(self):
        return "<Card(%s%s)>" % (
            self.RANK_TO_STRING[self.rank],
            self.SUIT_TO_STRING[self.suit],
        )

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.rank == other.rank
            and self.suit == other.suit
        )

    def __hash__(self):
        return hash((self.rank, self.suit))


    @classmethod
    def from_repr(self, repr):
        """Return a card instance from repr.
        This is really dirty--it just matches between the parens.
        It's meant for debugging.
        """
        between_parens = re.search(self.REPR_RE, repr).group(1)
        rank = self.STRING_TO_RANK[between_parens[0].upper()]
        suit = self.STRING_TO_SUIT[between_parens[1].lower()]
        return Card(rank, suit)
