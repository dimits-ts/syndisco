"""
Module automating and managing batches of discussion/annotation tasks defined
in the syndisco.jobs module.
"""

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

import random
import typing
import datetime
import logging as pylog
from pathlib import Path

from tqdm.auto import tqdm

from . import actors
from . import turn_manager as tmanager
from . import jobs


logger = pylog.getLogger(Path(__file__).name)


class DiscussionExperiment:
    """
    An experiment that creates, manages, and executes multiple synthetic
    discussions using LLM-based agents.
    """

    def __init__(
        self,
        users: typing.Sequence[actors.Actor],
        seed_opinions: typing.Sequence[typing.Sequence[str]] | None = None,
        turn_manager_factory: typing.Callable[
            [], tmanager.TurnManager
        ] = tmanager.RoundRobin,
        history_ctx_len: int = 3,
        num_turns: int = 10,
        num_active_users: int = 2,
        num_discussions: int = 5,
    ):
        """
        Initialize a synthetic discussion experiment.

        :param users: List of all possible participants (LLM agents).
        :type users: list[Actor]
        :param seed_opinions: Hardcoded seed discussion
            segments to initiate synthetic discussions.
            Each segment is a sequence of comments (strings).
            One segment will be selected randomly for each new synthetic
            discussion and will be uttered by random synthetic participants.
            None if no seed opinions are to be provided.
        :type seed_opinions: Sequence[Sequence[str]], optional
        :param turn_manager_factory:
            The class representing the strategy for selecting the next speaker.
            Defaults to :class:RoundRobin.
        :type turn_manager_factory: Callable[[], TurnManager]
        :param history_ctx_len: Number of past comments visible as context.
        :type history_ctx_len: int
        :param num_turns: Number of turns per discussion.
        :type num_turns: int
        :param num_active_users: Number of active participants per discussion.
        :type num_active_users: int
        :param num_discussions: Total number of synthetic discussions to run.
        :type num_discussions: int
        """
        self.seed_opinions = (
            seed_opinions if seed_opinions is not None else [[]]
        )
        self.users = users

        if len(self.users) < num_active_users:
            raise ValueError(
                f"Number of given users ({len(self.users)}) "
                "is inadequeate for number of requested users per discussion"
                f"({num_active_users})."
            )

        if num_discussions < 1:
            raise ValueError("num_discussions must be at least 1.")
        if num_turns < 2:
            raise ValueError("num_discussions must be at least 2.")
        if num_active_users < 2:
            raise ValueError("num_active_users must be at least 2.")

        self.turn_manager_factory = turn_manager_factory
        self._history_ctx_len = history_ctx_len
        self._num_active_users = num_active_users
        self._num_discussions = num_discussions
        self._num_turns = num_turns

    def begin(
        self,
        discussions_output_dir: Path,
        verbose: bool = True,
    ) -> None:
        """
        Generate and run all configured discussions.
        The method serializes each discussion immediately upon completion.
        Thus, limited data is lost upon even fatal errors during execution.

        :param discussions_output_dir:
            Directory to place the serialized :class:Logs for each discussion.
        :type discussions_output_dir: Path
        :param verbose: Whether to print intermediate progress and outputs.
        :type verbose: bool
        """
        logger.info("Starting synthetic discussion generation.")
        discussions = self._generate_discussions()
        self._run_all_discussions(discussions, discussions_output_dir, verbose)
        logger.info("Finished synthetic discussion generation.")

    def _generate_discussions(self) -> list[jobs.Discussion]:
        """
        Internal helper to generate Discussion objects from configuration.

        :return: A list of configured Discussion objects.
        :rtype: list[Discussion]
        """
        experiments = []
        for _ in range(self._num_discussions):
            experiments.append(self._create_synthetic_discussion())
        return experiments

    def _create_synthetic_discussion(self):
        """
        Create and return a single randomized Discussion instance.

        :return: A synthetic Discussion object.
        :rtype: Discussion
        """
        rand_topic = random.choice(self.seed_opinions)
        rand_users = list(random.choices(self.users, k=self._num_active_users))
        rand_seeds_users = (
            [actor.get_actor_name() for actor in rand_users[: len(rand_topic)]]
            if rand_topic is not None
            else None
        )
        tm = self.turn_manager_factory()
        tm.set_actors(rand_users)

        return jobs.Discussion(
            users=rand_users,
            history_context_len=self._history_ctx_len,
            conv_len=self._num_turns,
            seed_opinions=rand_topic,
            seed_opinion_usernames=rand_seeds_users,
            next_turn_manager=tm,
        )

    def _run_all_discussions(
        self,
        discussions: typing.Sequence[jobs.Discussion],
        output_dir: Path,
        verbose: bool,
    ) -> None:
        """
        Execute all generated discussions and write their outputs to disk.

        :param discussions: List of Discussion instances to run.
        :type discussions: Sequence[Discussion]
        :param output_dir: Directory to save output JSON files.
        :type output_dir: Path
        :param verbose: Whether to print discussion progress.
        :type verbose: bool
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, discussion in tqdm(list(enumerate(discussions))):
            pylog.info(f"Running experiment {i + 1}/{len(discussions) + 1}...")
            self._run_single_discussion(
                discussion=discussion, output_dir=output_dir, verbose=verbose
            )

    def _run_single_discussion(
        self, discussion: jobs.Discussion, output_dir: Path, verbose: bool
    ) -> None:
        """
        Run a single Discussion and store its results.

        :param discussion: The Discussion object to execute.
        :type discussion: jobs.Discussion
        :param output_dir: Directory to write the result file.
        :type output_dir: Path
        :param verbose: Whether to show detailed logging output.
        :type verbose: bool
        """
        try:
            logger.debug(f"Experiment parameters: {str(discussion)}")

            discussion.begin(verbose=verbose)
            output_path = _generate_datetime_filename(output_dir=output_dir)
            logs = discussion.get_logs()
            logs.export(output_path)
        except Exception as e:
            logger.exception(f"Experiment aborted due to error: {e}")


class AnnotationExperiment:
    """
    An experiment that uses LLM annotators to label synthetic discussion logs.
    """

    def __init__(
        self,
        annotators: typing.Sequence[actors.Actor],
        discussion_logs: jobs.Logs,
        history_ctx_len: int = 3,
    ):
        """
        Initialize an annotation experiment using LLM-based annotators.

        :param annotators: List of annotator agents.
        :type annotators: Sequence[Actor]
        :param discussion_logs: The discussions to be annotated.
        :type discussion_logs: Sequence[Actor]
        :param history_ctx_len: Number of previous comments visible to the
            annotator.
        :type history_ctx_len: int
        """
        self.annotators = annotators
        self.history_ctx_len = history_ctx_len
        self.discussion_logs = discussion_logs

    def begin(self, output_dir: Path, verbose: bool = True) -> None:
        """
        Start the annotation process.
        The method serializes each discussion immediately upon completion.
        Thus, limited data is lost upon even fatal errors during execution.

        :param output_dir: Directory to write annotation outputs.
        :type output_dir: Path
        :param verbose: Whether to display annotation progress.
        :type verbose: bool, defaults to True
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        annotation_tasks = self._generate_annotation_tasks()
        self._run_all_annotations(annotation_tasks, output_dir, verbose)

    def _generate_annotation_tasks(self) -> typing.Sequence[jobs.Annotation]:
        """
        Create annotation tasks by pairing each annotator with each discussion.

        :return: List of Annotation tasks.
        :rtype: Sequence[Annotation]
        """
        annotation_tasks = []
        for annotator in self.annotators:
            annotation_task = self._create_annotation_task(annotator)
            annotation_tasks.append(annotation_task)
        return annotation_tasks

    def _create_annotation_task(
        self, annotator: actors.Actor
    ) -> jobs.Annotation:
        """
        Construct a single Annotation task.

        :param annotator: The LLM-based annotator.
        :type annotator: Actor
        :param conv_logs_path: Path to the discussion log file.
        :type conv_logs_path: Path
        :return: Configured Annotation task.
        :rtype: Annotation
        """
        return jobs.Annotation(
            annotator=annotator,
            discussion_logs=self.discussion_logs,
            history_ctx_len=self.history_ctx_len,
        )

    def _run_all_annotations(
        self,
        annotation_tasks: typing.Sequence[jobs.Annotation],
        output_dir: Path,
        verbose: bool = True,
    ) -> None:
        """
        Execute and store all annotation tasks.

        :param annotation_tasks: List of Annotation objects.
        :type annotation_tasks: Sequence[Annotation]
        :param output_dir: Directory to save results.
        :type output_dir: Path
        :param verbose: Whether to log intermediate steps.
        :type verbose: bool, defaults to true
        """
        for annotation_task in tqdm(list(annotation_tasks)):
            self._run_single_annotation(annotation_task, output_dir, verbose)

        logger.info("Finished annotation generation.")

    def _run_single_annotation(
        self, annotation_task: jobs.Annotation, output_dir: Path, verbose: bool
    ) -> None:
        """
        Execute one annotation task and write its output.

        :param annotation_task: Single Annotation object to run.
        :type annotation_task: Annotation
        :param output_dir: Directory for output file.
        :type output_dir: Path
        :param verbose: Whether to show debug output.
        :type verbose: bool
        """
        try:
            logger.debug(f"Experiment parameters: {str(annotation_task)}")
            annotation_task.begin(verbose=verbose)
            output_path = _generate_datetime_filename(output_dir=output_dir)
            annotation_logs = annotation_task.get_logs()
            annotation_logs.export(output_path)
        except Exception:
            logger.exception("Annotation experiment aborted due to error.")


def _generate_datetime_filename(
    output_dir: Path, timestamp_format: str = "%y-%m-%d-%H-%M-%S"
) -> Path:
    """
    Generate a filename based on the current date and time.

    :param output_dir: The path to the generated file.
    :type output_dir: Path
    :param timestamp_format: strftime format, defaults to "%y-%m-%d-%H-%M-%S"
    :type timestamp_format: str, optional
    :return: the full path for the generated file
    :rtype: str
    """
    datetime_name = (
        datetime.datetime.now().strftime(timestamp_format) + ".json"
    )
    path = Path(output_dir / datetime_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path
