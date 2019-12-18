from .lookup_tables import LookupTables
from .popcount import PopCount
from itertools import combinations
from operator import mul, __or__, __and__, __xor__
from functools import reduce
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Iterable
from pokereval.card import Card


class HandLengthException(BaseException):
    pass


class BaseHandEvaluator(ABC):

    def __init__(self):
        self.pc = PopCount()

    @abstractmethod
    def evaluate(self, hand: List[Card]):
        raise NotImplementedError

    @staticmethod
    def _validate_inputs(hand, board):
        """ Checks that hand lengths and board size are valid"""
        hand_lengths = [2, 5, 6, 7]
        if len(hand) not in hand_lengths:
            raise HandLengthException("Only %s hole cards are supported" % ", ".join(map(str, hand_lengths)))
        cards = list(hand) + list(board)
        if 2 > len(cards) > 7:
            # wrong number of cards
            raise HandLengthException("Only 2, 5, 6, 7 cards total are supported by evaluate_hand")

        return cards

    def _evaluate_possible_hands(self, possible_hands: List[Iterable], board: List[Card], benchmark_rank: int):
        hands_beaten = 0
        for h in possible_hands:
            possible_opponent_rank = self.evaluate(list(h) + board)
            if benchmark_rank < possible_opponent_rank:
                hands_beaten += 1
            # Note that this treats ties as exactly in the middle of wins/losses. May want to tune depending on target.
            elif benchmark_rank == possible_opponent_rank:
                hands_beaten += 0.5

        return hands_beaten

    def evaluate_hand(self, hand: List[Card], board: List[Card] = ()):
        """
        Return the percentile of the best 5 card hand made from these
        cards, against an equivalent number of cards..

        Uses the `.evaluate(cards)` method implemented in subclasses to identify rank.

        # TODO: Could pass 'oppoonent range' instead of all possible cards
          to paramatarize the possible opponent hands in call to `_evaluate_possible_hands()`

        Args:
            hand (list): Contains cards in the hand of a given player
            board (list): Contains cards on the board

        Returns:
            (float) hands_beaten / possible_opponent_hands
        """
        # Default values in case we screw up
        rank = 7463
        percentile = 0.0

        cards = self._validate_inputs(hand, board)
        rank = self.evaluate(cards)

        # Here just iterates over all possible opponent hands to figure out how many are beaten.
        possible_opponent_hands = list(combinations(LookupTables.deck - set(cards), len(hand)))
        hands_beaten = self._evaluate_possible_hands(possible_opponent_hands, board, rank)

        return float(hands_beaten) / len(list(possible_opponent_hands))


class TwoHandEvaluator(BaseHandEvaluator):

    def __init__(self):
        super(TwoHandEvaluator, self).__init__()

    def evaluate(self, hand: List[Card]):
        """
        Using lookup table, return percentile of your hand with two cards
        """
        if len(hand) != 2:
            raise HandLengthException("Only 2-card hands are supported by the Two evaluator")

        if hand[0].suit == hand[1].suit:
            if hand[0].rank < hand[1].rank:
                return LookupTables.Two.suited_ranks_to_percentile[hand[0].rank][hand[1].rank]
            else:
                return LookupTables.Two.suited_ranks_to_percentile[hand[1].rank][hand[0].rank]
        else:
            return LookupTables.Two.unsuited_ranks_to_percentile[hand[0].rank][hand[1].rank]


class FiveHandEvaluator(BaseHandEvaluator):

    def __init__(self):
        super(FiveHandEvaluator, self).__init__()

    def card_to_binary(self, card: Card):
        """
        Convert the lookup_tables.Card representation to a binary
        representation for use in 5-card hand evaluation
        """
        # This is Cactus Kev's algorithm, reimplemented in Python since we can't
        # use C libraries

        # First we need to generate the following representation
        # Bits marked x are not used.
        # xxxbbbbb bbbbbbbb cdhsrrrr xxpppppp

        # b is one bit flipped for A-2
        # c, d, h, s are flipped if you have a club, diamond, heart, spade
        # r is just the numerical rank in binary, with deuce = 0
        # p is the prime from LookupTable.primes corresponding to the rank,
        # in binary
        # Then shift appropriately to fit the template above

        b_mask = 1 << (14 + card.rank)
        cdhs_mask = 1 << (card.suit + 11)
        r_mask = (card.rank - 2) << 8
        p_mask = LookupTables.primes[card.rank - 2]
        # OR them together to get the final result
        return b_mask | r_mask | p_mask | cdhs_mask

    def card_to_binary_lookup(self, card: Card):
        return LookupTables.Five.card_to_binary[card.rank][card.suit]

    # TODO: Return a class of hand too? Would be useful to see if we can make
    # a draw or something.
    def evaluate(self, hand: List[Card]):
        """
        Return the rank of this 5-card hand amongst all 5-card hands.
        """
        if len(hand) != 5:
            raise HandLengthException("Only 5-card hands are supported by the Five evaluator")

        # This implementation uses the binary representation from
        # card_to_binary
        card_to_binary = self.card_to_binary_lookup
        # bh stands for binary hand
        bh = list(map(card_to_binary, hand))

        has_flush = reduce(__and__, bh, 0xF000)
        # This is a unique number based on the ranks if your cards,
        # assuming your cards are all different
        q = reduce(__or__, bh) >> 16
        if has_flush:
            # Look up the rank of this flush
            return LookupTables.Five.flushes[q]

        else:
            # The q still works as a key if you have 5 unique cards.
            # First we look for straights, then duplicate value. (pair, trip, etc.)
            possible_rank = LookupTables.Five.unique5[q]
            if possible_rank != 0:
                return possible_rank
            else:
                # Compute the unique product of primes, because we have a pair
                # or trips, etc. Use the product to look up the rank.
                q = reduce(mul, [card & 0xFF for card in bh])
                # Here, use dict instead of sparse array
                return LookupTables.Five.pairs.get(q)


class SixHandEvaluator(BaseHandEvaluator):

    def __init__(self):
        super(SixHandEvaluator, self).__init__()

    @staticmethod
    def card_to_binary(card: Card):
        """ Convert the Card to a binary representation for use in 6-card hand evaluation"""
        # This a variant on Cactus Kev's algorithm. We need to replace
        # the 4-bit representation of suit with a prime number representation
        # so we can look up whether something is a flush by prime product

        # First we need to generate the following representation
        # Bits marked x are not used.
        # xxxbbbbb bbbbbbbb qqqqrrrr xxpppppp

        # b is one bit flipped for A-2
        # q is 2, 3, 5, or 7 for spades, hearts, clubs, diamonds
        # r is just the numerical rank in binary, with deuce = 0
        # p is the prime from LookupTable.primes corresponding to the rank,
        # in binary
        # Then shift appropriately to fit the template above
        b_mask = 1 << (14 + card.rank)
        q_mask = LookupTables.primes[card.suit - 1] << 12
        r_mask = (card.rank - 2) << 8
        p_mask = LookupTables.primes[card.rank - 2]
        # OR them together to get the final result
        return b_mask | q_mask | r_mask | p_mask

    @staticmethod
    def card_to_binary_lookup(card: Card):
        return LookupTables.Six.card_to_binary[card.rank][card.suit]

    def _evaluate_0x(self, bh, even_xor):
        even_popcount = self.pc.popcount32_table16(even_xor)
        if even_popcount == 2:  # 0-2
            prime_product = reduce(mul, [card & 0xFF for card in bh])
            return LookupTables.Six.prime_products_to_rank[prime_product]
        else:  # 0-3
            return LookupTables.Six.even_xors_to_rank[even_xor]

    def _evaluate_x0(self, bh, odd_xor):
        odd_popcount = self.pc.popcount32_table16(odd_xor)
        if odd_popcount == 4:  # 4-0
            prime_product = reduce(mul, [card & 0xFF for card in bh])
            return LookupTables.Six.prime_products_to_rank[prime_product]
        else:  # 6-0, 2-0
            return LookupTables.Six.odd_xors_to_rank[odd_xor]

    def _evaluate_xx(self, bh, odd_xor, even_xor):
        """Odd popcount either 4 or 2, even popcount either 2 or 1."""
        odd_popcount = self.pc.popcount32_table16(odd_xor)
        if odd_popcount == 4:  # 4-1
            return LookupTables.Six.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
        else:  # 2-x
            even_popcount = self.pc.popcount32_table16(even_xor)
            if even_popcount == 2:  # 2-2
                return LookupTables.Six.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
            else:  # 2-1
                prime_product = reduce(mul, [card & 0xFF for card in bh])
                return LookupTables.Six.prime_products_to_rank[prime_product]

    def evaluate(self, hand: List[Card]):
        """
        Return the rank amongst all possible 5-card hands of any kind
        using the best 5-card hand from the given 6-card hand.
        """
        if len(hand) != 6:
            raise HandLengthException("Only 6-card hands are supported by the Six evaluator")

        card_to_binary = self.card_to_binary_lookup
        bh = list(map(card_to_binary, hand))
        odd_xor = reduce(__xor__, bh) >> 16
        even_xor = (reduce(__or__, bh) >> 16) ^ odd_xor

        # Basically use prime number trick but map to bool instead of rank
        # Once you have a flush, there is no other higher hand you can make
        # except straight flush, so just need to determine the highest flush
        flush_prime = reduce(mul, [(card >> 12) & 0xF for card in bh])
        flush_suit = False
        if flush_prime in LookupTables.Six.prime_products_to_flush:
            flush_suit = LookupTables.Six.prime_products_to_flush[flush_prime]

        # Now use ranks to determine hand via lookup

        # If you have a flush, use odd_xor to find the rank
        # That value will have either 4 or 5 bits
        if flush_suit:
            if even_xor == 0:
                # There might be 0 or 1 cards in the wrong suit, so filter
                # TODO: There might be a faster way?
                bits = reduce(__or__, [(card >> 16) for card in [card for card in bh if (card >> 12) & 0xF == flush_suit]])
                return LookupTables.Six.flush_rank_bits_to_rank[bits]
            else:
                # you have a pair, one card in the flush suit,
                # so just use the ranks you have by or'ing the two
                return LookupTables.Six.flush_rank_bits_to_rank[odd_xor | even_xor]

        # Otherwise, get ready for a wild ride:

        # Can determine this by using 2 XORs to reduce the size of the
        # lookup. You have an even number of cards,
        # any odd_xor with an odd number of bits set is not possible.
        # Possibilities are `odd_xor`-`even_xor`:
        # 6-0 => High card or straight (1,1,1,1,1,1)
        #   Look up by odd_xor
        # 4-1 => Pair (1,1,1,1,2)
        #   Look up by even_xor (which pair) then odd_xor (which set of kickers)
        # 4-0 => Trips (1,1,1,3)
        #   Don't know which one is the triple, use prime product of ranks
        # 2-2 => Two pair (1,1,2,2)
        #   Look up by odd_xor then even_xor (or vice-versa)
        # 2-1 => Four of a kind (1,1,4) or full house (1,3,2)
        #   Look up by prime product
        # 2-0 => Full house using 2 trips (3,3)
        #   Look up by odd_xor
        # 0-3 => Three pairs (2,2,2)
        #   Look up by even_xor
        # 0-2 => Four of a kind with pair (2,4)
        #   Look up by prime product

        # Any time you can't disambiguate 2/4 or 1/3, use primes.
        # We also assume you can count bits or determine a power of two.
        # (see PopCount class.)
        if even_xor == 0:  # x-0
            return self._evaluate_x0(bh, odd_xor)
        elif odd_xor == 0:  # 0-x
            return self._evaluate_0x(bh, even_xor)
        else:  # odd_popcount is 4 or 2
            return self._evaluate_xx(bh, odd_xor, even_xor)


class SevenHandEvaluator(BaseHandEvaluator):
    def __init__(self):
        super(SevenHandEvaluator, self).__init__()

    def card_to_binary(self, card: Card):
        """
        Convert the lookup_tables.Card representation to a binary
        representation for use in 7-card hand evaluation
        """
        # Same as for 6 cards
        b_mask = 1 << (14 + card.rank)
        q_mask = LookupTables.primes[card.suit - 1] << 12
        r_mask = (card.rank - 2) << 8
        p_mask = LookupTables.primes[card.rank - 2]
        return b_mask | q_mask | r_mask | p_mask

    def card_to_binary_lookup(self, card: Card):
        return LookupTables.Seven.card_to_binary[card.rank][card.suit]

    def evaluate(self, hand: List[Card]):
        """
        Return the rank amongst all possible 5-card hands of any kind
        using the best 5-card hand from the given 6-card hand.
        """
        if len(hand) != 7:
            raise HandLengthException("Only 7-card hands are supported by the Seven evaluator")

        # bh stands for binary hand, map to that representation
        card_to_binary = self.card_to_binary_lookup
        bh = list(map(card_to_binary, hand))

        # Use a lookup table to determine if it's a flush as with 6 cards
        flush_prime = reduce(mul, [(card >> 12) & 0xF for card in bh])
        flush_suit = False
        if flush_prime in LookupTables.Seven.prime_products_to_flush:
            flush_suit = LookupTables.Seven.prime_products_to_flush[flush_prime]

        # Now use ranks to determine hand via lookup
        odd_xor = reduce(__xor__, bh) >> 16
        even_xor = (reduce(__or__, bh) >> 16) ^ odd_xor

        if flush_suit:
            # There will be 0-2 cards not in the right suit
            even_popcount = self.pc.popcount32_table16(even_xor)
            if even_xor == 0:
                # TODO: There might be a faster way?
                bits = reduce(__or__,
                              [(card >> 16) for card in [card for card in bh if (card >> 12) & 0xF == flush_suit]])
                return LookupTables.Seven.flush_rank_bits_to_rank[bits]
            else:
                if even_popcount == 2:
                    return LookupTables.Seven.flush_rank_bits_to_rank[odd_xor | even_xor]
                else:
                    bits = reduce(__or__,
                                  [(card >> 16) for card in [card for card in bh if (card >> 12) & 0xF == flush_suit]])
                    return LookupTables.Seven.flush_rank_bits_to_rank[bits]

        # Odd-even XOR again, see Six.evaluate_rank for details
        # 7 is odd, so you have to have an odd number of bits in odd_xor
        # 7-0 => (1,1,1,1,1,1,1) - High card
        # 5-1 => (1,1,1,1,1,2) - Pair
        # 5-0 => (1,1,1,1,3) - Trips
        # 3-2 => (1,1,1,2,2) - Two pair
        # 3-1 => (1,1,1,4) or (1,1,3,2) - Quads or full house
        # 3-0 => (1,3,3) - Full house
        # 1-3 => (1,2,2,2) - Two pair
        # 1-2 => (1,2,4) or (3,2,2) - Quads or full house
        # 1-1 => (3,4) - Quads
        if even_xor == 0:  # x-0
            odd_popcount = self.pc.popcount32_table16(odd_xor)
            if odd_popcount == 7:  # 7-0
                return LookupTables.Seven.odd_xors_to_rank[odd_xor]
            else:  # 5-0, 3-0
                prime_product = reduce(mul, [card & 0xFF for card in bh])
                return LookupTables.Seven.prime_products_to_rank[prime_product]
        else:
            odd_popcount = self.pc.popcount32_table16(odd_xor)
            if odd_popcount == 5:  # 5-1
                return LookupTables.Seven.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
            elif odd_popcount == 3:
                even_popcount = self.pc.popcount32_table16(even_xor)
                if even_popcount == 2:  # 3-2
                    return LookupTables.Seven.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
                else:  # 3-1
                    prime_product = reduce(mul, [card & 0xFF for card in bh])
                    return LookupTables.Seven.prime_products_to_rank[prime_product]
            else:
                even_popcount = self.pc.popcount32_table16(even_xor)
                if even_popcount == 3:  # 1-3
                    return LookupTables.Seven.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
                elif even_popcount == 2:  # 1-2
                    prime_product = reduce(mul, [card & 0xFF for card in bh])
                    return LookupTables.Seven.prime_products_to_rank[prime_product]
                else:  # 1-1
                    return LookupTables.Seven.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
