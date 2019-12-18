import pytest
from pokereval.card import Card, InvalidCardError
from pokereval.hand_evaluator import TwoHandEvaluator, FiveHandEvaluator, SixHandEvaluator, \
    SevenHandEvaluator, HandLengthException
import numpy as np


class TestCards():

    def test_attrs(self):
        a = Card(2, 1)
        a.__repr__
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

class TestEvaluator():

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

    def test_five_cards(self):
        hole = [Card(2, 1), Card(2, 2)]
        board = [Card(2, 3), Card(3, 3), Card(4, 3)]
        fwe = FiveHandEvaluator()
        score = fwe.evaluate_hand(hole, board)
        assert np.isclose(score, 0.9250693802035153)

    def test_six_cards(self):
        hole = [Card(2, 1), Card(2, 2)]
        board = [Card(2, 3), Card(3, 3), Card(4, 3), Card(5, 3)]
        swe = SixHandEvaluator()
        score = swe.evaluate_hand(hole, board)
        assert np.isclose(score, 0.4405797101449275)

    def test_seven_cards(self):
        hole = [Card(2, 1), Card(2, 2)]
        board = [Card(2, 3), Card(3, 3), Card(4, 3), Card(5, 3), Card(5, 4)]
        swe = SevenHandEvaluator()
        score = swe.evaluate_hand(hole, board)
        assert np.isclose(score, 0.8909090909090909)