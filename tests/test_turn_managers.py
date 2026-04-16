# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2025 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr
import pytest
from typing import Iterable

from .dummy import DummyActor
from syndisco import TurnManager, RandomWeighted, RoundRobin, Actor


@pytest.fixture
def actors():
    return [DummyActor("A"), DummyActor("B"), DummyActor("C")]


def assert_no_consecutive_repetition(sequence):
    for i in range(1, len(sequence)):
        assert sequence[i] != sequence[i - 1]


class TestSuperclass:
    def test_turn_manager_requires_names(self):
        class DummyTM(TurnManager):
            def __init__(self, actors: Iterable[Actor] | None = None):
                super().__init__(actors)

            def _next_impl(self):  # type: ignore
                return None

        tm = DummyTM()

        with pytest.raises(ValueError):
            tm.next()

    def test_set_actors_overwrites_previous(self, actors):
        class DummyTM(TurnManager):
            def __init__(self):
                super().__init__()
                self.idx = 0

            def _next_impl(self):
                val = self._actors[self.idx % len(self._actors)]
                self.idx += 1
                return val

        tm = DummyTM()

        # First assignment
        tm.set_actors(actors)
        _ = [tm.next() for _ in range(len(actors))]

        # New actors
        new_actors = [DummyActor("X"), DummyActor("Y")]
        tm.set_actors(new_actors)

        # Ensure names updated
        assert tm._actors == new_actors

        # Ensure next() now uses new actors only
        for _ in range(10):
            assert tm.next() in new_actors

    def test_set_actors_initializes_names(self, actors):
        class DummyTM(TurnManager):
            def _next_impl(self):
                return self._actors[0]

        tm = DummyTM()
        tm.set_actors(actors)

        assert tm._actors == actors

    def test_iteration_protocol(self, actors):
        class DummyTM(TurnManager):
            def __init__(self, actors):
                super().__init__(actors)
                self.i = 0

            def _next_impl(self):
                val = self._actors[self.i % len(self._actors)]
                self.i += 1
                return val

        tm = DummyTM(actors)
        tm.set_actors(actors)

        iterator = iter(tm)
        assert iterator is tm

        first = next(tm)
        second = next(tm)

        assert first in actors
        assert second in actors


class TestRounRobin:
    def test_round_robin_cycles(self, actors):
        rr = RoundRobin(actors)
        rr.set_actors(actors)

        seen = [rr.next() for _ in range(len(actors) * 2)]

        # Expect repetition of same order
        assert seen[: len(actors)] == seen[len(actors):]

    def test_round_robin_returns_valid_actor(self, actors):
        rr = RoundRobin(actors)
        rr.set_actors(actors)

        for _ in range(10):
            assert rr.next() in actors

    def test_empty_actor_list(self):
        rr = RoundRobin([])
        rr.set_actors([])

        with pytest.raises(ValueError):
            rr.next()

    def test_single_actor(self, actors):
        single = [actors[0]]

        rr = RoundRobin(single)
        rr.set_actors(single)

        for _ in range(5):
            assert rr.next() == single[0]

    def test_no_consecutive_repetition(self, actors):
        tm = RoundRobin(actors)
        tm.set_actors(actors)

        sequence = [tm.next() for _ in range(20)]

        assert_no_consecutive_repetition(sequence)


class TestRandomWeighted:
    def test_random_weighted_invalid_probability(self, actors):
        with pytest.raises(AssertionError):
            RandomWeighted(actors, p_respond=-0.1)

        with pytest.raises(AssertionError):
            RandomWeighted(actors, p_respond=1.1)

    def test_random_weighted_valid_probability(self, actors):
        rw = RandomWeighted(actors, p_respond=0.5)
        assert 0 <= rw.chance_to_respond <= 1

    def test_random_weighted_returns_valid_actor(self, actors):
        rw = RandomWeighted(actors, p_respond=0.5)
        rw.set_actors(actors)

        for _ in range(20):
            actor = rw.next()
            assert actor in actors

    def test_random_weighted_initial_state(self, actors):
        rw = RandomWeighted(actors, p_respond=0.5)

        assert rw._last_speaker is None
        assert rw._second_to_last_speaker is None

    def test_no_repetition_when_p_zero(self, actors):
        tm = RandomWeighted(actors, p_respond=0.0)
        tm.set_actors(actors)

        sequence = [tm.next() for _ in range(50)]

        assert_no_consecutive_repetition(sequence)
