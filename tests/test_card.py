import pytest
from pokereval.card import Card, InvalidCardError
from pokereval.hand_evaluator import TwoHandEvaluator, FiveHandEvaluator, SixHandEvaluator, \
    SevenHandEvaluator, HandLengthException
import numpy as np


@pytest.fixture
def pair_cards():
    five = [[Card(2, 1), Card(4, 2)], [Card(2, 3), Card(5, 4), Card(9, 1)]]
    six = [[Card(2, 1), Card(4, 2)], [Card(2, 3), Card(5, 4), Card(9, 1), Card(8, 2)]]
    seven = [[Card(2, 1), Card(4, 2)], [Card(2, 3), Card(3, 4), Card(9, 1), Card(8, 2), Card(7, 2)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def twopair_cards():
    five = [[Card(2, 1), Card(6, 2)], [Card(2, 3), Card(6, 4), Card(3, 1)]]
    six = [[Card(2, 1), Card(6, 2)], [Card(2, 3), Card(6, 4), Card(3, 1), Card(8, 2)]]
    seven = [[Card(2, 1), Card(6, 2)], [Card(2, 3), Card(6, 4), Card(3, 1), Card(8, 2), Card(7, 2)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def trips_cards():
    five = [[Card(10, 1), Card(10, 2)], [Card(10, 3), Card(3, 3), Card(4, 3)]]
    six = [[Card(10, 1), Card(10, 2)], [Card(10, 3), Card(3, 3), Card(4, 3), Card(8, 2)]]
    seven = [[Card(10, 1), Card(10, 2)], [Card(10, 3), Card(3, 3), Card(4, 3), Card(8, 2), Card(7, 2)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def flush_cards():
    five = [[Card(2, 1), Card(6, 1)], [Card(8, 1), Card(4, 1), Card(3, 1)]]
    six = [[Card(2, 1), Card(6, 1)], [Card(8, 1), Card(4, 1), Card(3, 1), Card(2, 2)]]
    seven = [[Card(2, 1), Card(6, 1)], [Card(8, 1), Card(4, 1), Card(3, 1), Card(2, 2), Card(3, 2)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def straight_flush_cards():
    five = [[Card(2, 1), Card(3, 1)], [Card(4, 1), Card(5, 1), Card(6, 1)]]
    six = [[Card(2, 1), Card(3, 1)], [Card(4, 1), Card(5, 1), Card(6, 1), Card(2, 2)]]
    seven = [[Card(2, 1), Card(3, 1)], [Card(4, 1), Card(5, 1), Card(6, 1), Card(2, 2), Card(3, 2)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def straight_cards():
    five = [[Card(2, 1), Card(3, 2)], [Card(4, 3), Card(5, 1), Card(6, 1)]]
    six = [[Card(2, 1), Card(3, 2)], [Card(4, 3), Card(5, 1), Card(6, 1), Card(2, 2)]]
    seven = [[Card(2, 1), Card(3, 2)], [Card(4, 3), Card(5, 1), Card(6, 1), Card(2, 2), Card(3, 2)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def fullhouse_cards():
    five = [[Card(2, 1), Card(2, 2)], [Card(2, 3), Card(3, 1), Card(3, 2)]]
    six = [[Card(2, 1), Card(2, 2)], [Card(2, 3), Card(3, 1), Card(3, 2), Card(5, 2)]]
    seven = [[Card(2, 1), Card(2, 2)], [Card(2, 3), Card(3, 1), Card(3, 2), Card(5, 2), Card(6, 4)]]
    return {'five': five, 'six': six, 'seven': seven}


@pytest.fixture
def fullhouse_twotrips_cards():
    six = [[Card(2, 1), Card(2, 2)], [Card(2, 3), Card(3, 1), Card(3, 2), Card(3, 4)]]
    seven = [[Card(2, 1), Card(2, 2)], [Card(2, 3), Card(3, 1), Card(3, 2), Card(3, 4), Card(6, 4)]]
    return {'six': six, 'seven': seven}


class TestCards(object):

    def test_attrs(self):
        a = Card(2, 1)
        a.__repr__()

    def test_rank(self):
        for i in range(2, 15):
            Card(i, 1)
        for i in range(2, 10):
            Card(str(i), 1)
        for i in ("T", "J", "Q", "K", "A"):
            Card(i, 1)
            Card(i.lower(), 1)

        with pytest.raises(TypeError):
            Card({}, 1)
        with pytest.raises(InvalidCardError):
            Card(1, 1)
        with pytest.raises(InvalidCardError):
            Card(15, 1)
        with pytest.raises(InvalidCardError):
            Card("F", 3)
        with pytest.raises(InvalidCardError):
            Card(3, "F")

    def test_suit(self):
        for i in range(1, 5):
            Card(2, i)
        for i in ("s", "h", "d", "c"):
            Card(2, i)
            Card(2, i.upper())

        with pytest.raises(TypeError):
            Card(2, [])
        with pytest.raises(InvalidCardError):
            Card(2, 0)
        with pytest.raises(InvalidCardError):
            Card(2, 5)


class TestEvaluator(object):
    # TODO: Need to evaluate each type of hand at least once for each Evaluator subclass.
    def test_evaluator(self):

        with pytest.raises(HandLengthException):
            board = []
            hole = [Card(2, 1)]
            fwe = TwoHandEvaluator()
            score = fwe.evaluate_hand(hole, board)
        with pytest.raises(HandLengthException):
            hole = [Card(2, 1), Card(3, 1)]
            board = [Card(4, 1), Card(5, 1), Card(6, 1), Card(7, 1), Card(8, 1), Card(2,2), Card(2, 3), Card(2,4)]
            fwe = TwoHandEvaluator()
            score = fwe.evaluate_hand(hole, board)

    def test_two_cards(self):
        hole = [Card(2, 1), Card(2, 2)]
        board = []
        twe = TwoHandEvaluator()
        score = twe.evaluate_hand(hole, board)
        assert np.isclose(score, 0.503265306122449)

    def test_five_cards(self, pair_cards, twopair_cards, straight_cards,
                        flush_cards, fullhouse_cards, straight_flush_cards):
        hole = [Card(10, 1), Card(10, 2)]
        board = [Card(10, 3), Card(3, 3), Card(4, 3)]
        fwe = FiveHandEvaluator()
        trips_score = fwe.evaluate_hand(hole, board)
        assert np.isclose(trips_score, 0.9583718778908418)

        # Test all hands:
        pair_score = fwe.evaluate_hand(*pair_cards['five'])
        twopair_score = fwe.evaluate_hand(*twopair_cards['five'])
        flush_score = fwe.evaluate_hand(*flush_cards['five'])
        straight_score = fwe.evaluate_hand(*straight_cards['five'])
        fullhouse_score = fwe.evaluate_hand(*fullhouse_cards['five'])
        straight_flush_score = fwe.evaluate_hand(*straight_flush_cards['five'])

        # This may not be guaranteed because the possible hands changes given the board, need to make better fixtures.
        # assert twopair_score > pair_score
        # assert trips_score > twopair_score
        # assert straight_score > trips_score
        # assert flush_score > straight_score
        # assert fullhouse_score > flush_score
        # assert straight_flush_score > flush_score

    def test_six_cards(self, pair_cards, twopair_cards, straight_cards,
                       flush_cards, fullhouse_cards, fullhouse_twotrips_cards, straight_flush_cards):
        hole = [Card(2, 1), Card(2, 2)]
        board = [Card(2, 3), Card(3, 3), Card(4, 3), Card(5, 3)]
        swe = SixHandEvaluator()
        trips_score = swe.evaluate_hand(hole, board)
        assert np.isclose(trips_score, 0.4405797101449275)

        pair_score = swe.evaluate_hand(*pair_cards['six'])
        twopair_score = swe.evaluate_hand(*twopair_cards['six'])
        flush_score = swe.evaluate_hand(*flush_cards['six'])
        straight_score = swe.evaluate_hand(*straight_cards['six'])
        fullhouse_score = swe.evaluate_hand(*fullhouse_cards['six'])
        fullhouse_twotrips_score = swe.evaluate_hand(*fullhouse_twotrips_cards['six'])
        straight_flush_score = swe.evaluate_hand(*straight_flush_cards['six'])


    def test_seven_cards(self, pair_cards, twopair_cards, straight_cards,
                         flush_cards, fullhouse_cards, fullhouse_twotrips_cards, straight_flush_cards):
        hole = [Card(2, 1), Card(2, 2)]
        board = [Card(2, 3), Card(3, 3), Card(4, 3), Card(5, 3), Card(8, 4)]
        swe = SevenHandEvaluator()
        trips_score = swe.evaluate_hand(hole, board)
        assert np.isclose(trips_score, 0.4292929292929293)

        pair_score = swe.evaluate_hand(*pair_cards['seven'])
        twopair_score = swe.evaluate_hand(*twopair_cards['seven'])
        flush_score = swe.evaluate_hand(*flush_cards['seven'])
        straight_score = swe.evaluate_hand(*straight_cards['seven'])
        fullhouse_score = swe.evaluate_hand(*fullhouse_cards['seven'])
        fullhouse_twotrips_score = swe.evaluate_hand(*fullhouse_twotrips_cards['seven'])
        straight_flush_score = swe.evaluate_hand(*straight_flush_cards['seven'])