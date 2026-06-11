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
from collections import Counter

from .dummy import DummyActor
from syndisco import (
    TurnManager,
    RespondTurnManager,
    QueueTurnManager,
    Actor,
    DiscussionExperiment,
)


@pytest.fixture
def actors():
    return [DummyActor(f"User{i}") for i in range(5)]


@pytest.fixture
def minimal_experiment(actors):
    return DiscussionExperiment(
        users=actors,
        num_active_users=3,
        num_discussions=5,
        num_turns=2,
    )


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
        rr = QueueTurnManager(actors)
        rr.set_actors(actors)

        seen = [rr.next() for _ in range(len(actors) * 2)]

        # Expect repetition of same order
        assert seen[: len(actors)] == seen[len(actors) :]

    def test_round_robin_returns_valid_actor(self, actors):
        rr = QueueTurnManager(actors)
        rr.set_actors(actors)

        for _ in range(10):
            assert rr.next() in actors

    def test_empty_actor_list(self):
        rr = QueueTurnManager([])
        rr.set_actors([])

        with pytest.raises(ValueError):
            rr.next()

    def test_single_actor(self, actors):
        single = [actors[0]]

        rr = QueueTurnManager(single)
        rr.set_actors(single)

        for _ in range(5):
            assert rr.next() == single[0]

    def test_no_consecutive_repetition(self, actors):
        tm = QueueTurnManager(actors)
        tm.set_actors(actors)

        sequence = [tm.next() for _ in range(20)]

        assert_no_consecutive_repetition(sequence)


class TestRandomWeighted:
    def test_random_weighted_invalid_probability(self, actors):
        with pytest.raises(AssertionError):
            RespondTurnManager(actors, p_respond=-0.1)

        with pytest.raises(AssertionError):
            RespondTurnManager(actors, p_respond=1.1)

    def test_random_weighted_valid_probability(self, actors):
        rw = RespondTurnManager(actors, p_respond=0.5)
        assert 0 <= rw.chance_to_respond <= 1

    def test_random_weighted_returns_valid_actor(self, actors):
        rw = RespondTurnManager(actors, p_respond=0.5)
        rw.set_actors(actors)

        for _ in range(20):
            actor = rw.next()
            assert actor in actors

    def test_random_weighted_initial_state(self, actors):
        rw = RespondTurnManager(actors, p_respond=0.5)

        assert rw._last_speaker is None
        assert rw._second_to_last_speaker is None

    def test_no_repetition_when_p_zero(self, actors):
        tm = RespondTurnManager(actors, p_respond=0.0)
        tm.set_actors(actors)

        sequence = [tm.next() for _ in range(50)]

        assert_no_consecutive_repetition(sequence)


class TestNoDuplicateActors:
    """
    Regression tests for the random.choices -> random.sample bug.
    When random.choices was used, the same Actor object could appear
    multiple times in a discussion's participant list, causing the
    TurnManager to cycle through duplicate references.
    """

    def test_active_users_are_unique_objects(self, minimal_experiment):
        """
        Each Discussion created by _create_synthetic_discussion must
        contain no duplicate Actor instances (by identity).
        """
        for _ in range(20):
            discussion = minimal_experiment._create_synthetic_discussion()
            user_ids = [id(u) for u in discussion._users]
            assert len(user_ids) == len(set(user_ids)), (
                "Duplicate Actor objects found in discussion._users. "
                "Likely caused by random.choices instead of random.sample."
            )

    def test_turn_manager_has_no_duplicate_actors(self, minimal_experiment):
        """
        The TurnManager's internal actor list must not contain the same
        Actor object more than once after _create_synthetic_discussion.
        """
        for _ in range(20):
            discussion = minimal_experiment._create_synthetic_discussion()
            tm_actor_ids = [
                id(a) for a in discussion._next_turn_manager._actors
            ]
            assert len(tm_actor_ids) == len(
                set(tm_actor_ids)
            ), "Duplicate Actor objects found in TurnManager._actors."

    def test_active_users_are_unique_by_name(self, minimal_experiment):
        """
        Since each fixture actor has a unique name, no name should repeat
        within a single discussion's participant list.
        """
        for _ in range(20):
            discussion = minimal_experiment._create_synthetic_discussion()
            names = [u.get_actor_name() for u in discussion._users]
            counts = Counter(names)
            duplicates = {n: c for n, c in counts.items() if c > 1}
            assert (
                not duplicates
            ), f"Duplicate actor names in discussion: {duplicates}"

    def test_num_active_users_exactly_respected(self, actors):
        """
        The participant count must equal num_active_users exactly —
        no more (from duplicates being counted as distinct) and no fewer.
        """
        num_active = 3
        exp = DiscussionExperiment(
            users=actors,
            num_active_users=num_active,
            num_discussions=10,
            num_turns=2,
        )
        for _ in range(10):
            discussion = exp._create_synthetic_discussion()
            assert len(discussion._users) == num_active

    def test_discussion_draws_from_provided_pool(self, actors):
        """
        Every participant in a generated discussion must come from the
        original users pool — no phantom actors introduced.
        """
        exp = DiscussionExperiment(
            users=actors,
            num_active_users=3,
            num_discussions=10,
            num_turns=2,
        )
        actor_ids = {id(a) for a in actors}
        for _ in range(10):
            discussion = exp._create_synthetic_discussion()
            for user in discussion._users:
                assert (
                    id(user) in actor_ids
                ), f"Actor '{user.get_actor_name()}' not in original pool."


class TestExperimentValidation:
    """Guard-rail tests ensuring the constructor rejects impossible configs."""

    def test_raises_when_pool_smaller_than_active_users(self):
        small_pool = [DummyActor("A"), DummyActor("B")]
        with pytest.raises(ValueError, match="inadequ"):
            DiscussionExperiment(
                users=small_pool,
                num_active_users=3,
                num_discussions=1,
                num_turns=2,
            )

    def test_exact_pool_size_accepted(self):
        """Pool size == num_active_users is a valid edge case."""
        exact_pool = [DummyActor("A"), DummyActor("B"), DummyActor("C")]
        exp = DiscussionExperiment(
            users=exact_pool,
            num_active_users=3,
            num_discussions=1,
            num_turns=2,
        )
        discussion = exp._create_synthetic_discussion()
        user_ids = [id(u) for u in discussion._users]
        assert len(user_ids) == len(set(user_ids))

    def test_raises_on_zero_discussions(self, actors):
        with pytest.raises(ValueError):
            DiscussionExperiment(
                users=actors,
                num_active_users=2,
                num_discussions=0,
                num_turns=2,
            )

    def test_raises_on_single_turn(self, actors):
        with pytest.raises(ValueError):
            DiscussionExperiment(
                users=actors,
                num_active_users=2,
                num_discussions=1,
                num_turns=1,
            )

    def test_raises_on_single_active_user(self, actors):
        with pytest.raises(ValueError):
            DiscussionExperiment(
                users=actors,
                num_active_users=1,
                num_discussions=1,
                num_turns=2,
            )
