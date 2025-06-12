"""
Module automating and managing batches of discussion/annotation tasks defined
in the syndisco.jobs module.
"""

"""
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
"""

import time
import random
import logging
from pathlib import Path

from tqdm.auto import tqdm

from . import actors, turn_manager
from . import logging_util, _file_util
from . import jobs


logger = logging.getLogger(Path(__file__).name)


class DiscussionExperiment:
    """
    An experiment that creates, manages, and executes multiple synthetic
    discussions using LLM-based agents.
    """

    def __init__(
        self,
        users: list[actors.Actor],
        seed_opinions: list[str] = [],
        moderator: actors.Actor | None = None,
        next_turn_manager: turn_manager.TurnManager | None = None,
        history_ctx_len: int = 3,
        num_turns: int = 10,
        num_active_users: int = 2,
        num_discussions: int = 5,
    ):
        """
        Initialize a synthetic discussion experiment.

        :param users: List of all possible participants (LLM agents).
        :type users: list[actors.Actor]
        :param seed_opinions: Optional list of hardcoded seed comments to
            initiate discussions. One will be selected randomly for each.
        :type seed_opinions: list[str]
        :param moderator: Optional moderator agent, or None to omit moderation.
        :type moderator: actors.Actor or None
        :param next_turn_manager: Strategy for selecting the next speaker.
            Defaults to round-robin if None.
        :type next_turn_manager: turn_manager.TurnManager or None
        :param history_ctx_len: Number of past comments visible as context.
        :type history_ctx_len: int
        :param num_turns: Number of user (non-moderator) turns per discussion.
        :type num_turns: int
        :param num_active_users: Number of active participants per discussion.
        :type num_active_users: int
        :param num_discussions: Total number of synthetic discussions to run.
        :type num_discussions: int
        """
        self.seed_opinions = seed_opinions
        self.users = users
        self.moderator = moderator

        if next_turn_manager is None:
            logger.warning(
                "No TurnManager selected: Defaulting to round robin strategy."
            )
            self.next_turn_manager = turn_manager.RoundRobin()
        else:
            self.next_turn_manager = next_turn_manager

        self.history_ctx_len = history_ctx_len
        self.num_active_users = num_active_users
        self.num_discussions = num_discussions
        self.num_turns = num_turns

    def begin(
        self,
        discussions_output_dir: Path = Path("./output"),
        verbose: bool = True,
    ) -> None:
        """
        Generate and run all configured discussions.

        :param discussions_output_dir: Directory to write output JSON files.
        :type discussions_output_dir: Path
        :param verbose: Whether to print intermediate progress and outputs.
        :type verbose: bool
        """
        discussions = self._generate_discussions()
        self._run_all_discussions(discussions, discussions_output_dir, verbose)

    def _generate_discussions(self) -> list[jobs.Discussion]:
        """
        Internal helper to generate Discussion objects from configuration.

        :return: A list of configured Discussion objects.
        :rtype: list[jobs.Discussion]
        """
        experiments = []
        for _ in range(self.num_discussions):
            experiments.append(self._create_synthetic_discussion())
        return experiments

    def _create_synthetic_discussion(self):
        """
        Create and return a single randomized Discussion instance.

        :return: A synthetic Discussion object.
        :rtype: jobs.Discussion
        """
        rand_topic = random.choice(self.seed_opinions)
        rand_users = list(random.sample(self.users, k=self.num_active_users))

        return jobs.Discussion(
            users=rand_users,
            moderator=self.moderator,
            history_context_len=self.history_ctx_len,
            conv_len=self.num_turns,
            seed_opinion=rand_topic,
            seed_opinion_username=random.choice(rand_users).get_name(),
            next_turn_manager=self.next_turn_manager,
        )

    @logging_util.timing
    def _run_all_discussions(
        self,
        discussions: list[jobs.Discussion],
        output_dir: Path,
        verbose: bool,
    ) -> None:
        """
        Execute all generated discussions and write their outputs to disk.

        :param discussions: List of Discussion instances to run.
        :type discussions: list[jobs.Discussion]
        :param output_dir: Directory to save output JSON files.
        :type output_dir: Path
        :param verbose: Whether to print discussion progress.
        :type verbose: bool
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, discussion in tqdm(enumerate(discussions)):
            logging.info(
                f"Running experiment {i + 1}/{len(discussions) + 1}..."
            )
            self._run_single_discussion(
                discussion=discussion, output_dir=output_dir, verbose=verbose
            )

        logger.info("Finished synthetic discussion generation.")

    @logging_util.timing
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
            logger.info("Beginning conversation...")
            logger.debug(f"Experiment parameters: {str(discussion)}")

            start_time = time.time()
            discussion.begin(verbose=verbose)
            output_path = _file_util.generate_datetime_filename(
                output_dir=output_dir, file_ending=".json"
            )
            logging.debug(
                f"Finished discussion in {(time.time() - start_time)} seconds."
            )

            discussion.to_json_file(output_path)
            logger.info(f"Conversation saved to {output_path}")
        except Exception:
            logger.exception("Experiment aborted due to error.")


class AnnotationExperiment:
    """
    An experiment that uses LLM annotators to label synthetic discussion logs.
    """

    def __init__(
        self,
        annotators: list[actors.Actor],
        history_ctx_len: int = 3,
        include_mod_comments: bool = True,
    ):
        """
        Initialize an annotation experiment using LLM-based annotators.

        :param annotators: List of annotator agents.
        :type annotators: list[actors.Actor]
        :param history_ctx_len: Number of previous comments visible to the
            annotator.
        :type history_ctx_len: int
        :param include_mod_comments: Whether to include moderator comments
            during annotation.
        :type include_mod_comments: bool
        """
        self.annotators = annotators
        self.history_ctx_len = history_ctx_len
        self.include_mod_comments = include_mod_comments

    def begin(
        self, discussions_dir: Path, output_dir: Path, verbose: bool
    ) -> None:
        """
        Start the annotation process over all discussion logs in a directory.

        :param discussions_dir: Directory containing discussion logs.
        :type discussions_dir: Path
        :param output_dir: Directory to write annotation outputs.
        :type output_dir: Path
        :param verbose: Whether to display annotation progress.
        :type verbose: bool
        """
        if not discussions_dir.is_dir():
            raise OSError(
                f"Discussions directory ({discussions_dir}) is not a directory"
            ) from None

        output_dir.mkdir(parents=True, exist_ok=True)

        annotation_tasks = self._generate_annotation_tasks(discussions_dir)
        self._run_all_annotations(annotation_tasks, output_dir, verbose)

    def _generate_annotation_tasks(
        self, discussions_dir: Path
    ) -> list[jobs.Annotation]:
        """
        Create annotation tasks by pairing each annotator with each discussion.

        :param discussions_dir: Path to discussion log files.
        :type discussions_dir: Path
        :return: List of Annotation tasks.
        :rtype: list[jobs.Annotation]
        """
        annotation_tasks = []
        for annotator in self.annotators:
            for discussion_path in discussions_dir.iterdir():
                annotation_task = self._create_annotation_task(
                    annotator, discussion_path
                )
                annotation_tasks.append(annotation_task)
        return annotation_tasks

    def _create_annotation_task(
        self, annotator: actors.Actor, conv_logs_path: Path
    ) -> jobs.Annotation:
        """
        Construct a single Annotation task.

        :param annotator: The LLM-based annotator.
        :type annotator: actors.Actor
        :param conv_logs_path: Path to the discussion log file.
        :type conv_logs_path: Path
        :return: Configured Annotation task.
        :rtype: jobs.Annotation
        """
        return jobs.Annotation(
            annotator=annotator,
            conv_logs_path=conv_logs_path,
            history_ctx_len=self.history_ctx_len,
            include_moderator_comments=self.include_mod_comments,
        )

    @logging_util.timing
    def _run_all_annotations(
        self,
        annotation_tasks: list[jobs.Annotation],
        output_dir: Path,
        verbose: bool,
    ) -> None:
        """
        Execute and store all annotation tasks.

        :param annotation_tasks: List of Annotation objects.
        :type annotation_tasks: list[jobs.Annotation]
        :param output_dir: Directory to save results.
        :type output_dir: Path
        :param verbose: Whether to log intermediate steps.
        :type verbose: bool
        """
        for i, annotation_task in tqdm(enumerate(annotation_tasks)):
            logger.info(
                f"Running annotation {i + 1}/{len(annotation_tasks)}..."
            )
            self._run_single_annotation(annotation_task, output_dir, verbose)

        logger.info("Finished annotation generation.")

    @logging_util.timing
    def _run_single_annotation(
        self, annotation_task: jobs.Annotation, output_dir: Path, verbose: bool
    ) -> None:
        """
        Execute one annotation task and write its output.

        :param annotation_task: Single Annotation object to run.
        :type annotation_task: jobs.Annotation
        :param output_dir: Directory for output file.
        :type output_dir: Path
        :param verbose: Whether to show debug output.
        :type verbose: bool
        """
        try:
            logger.info("Beginning annotation...")
            logger.debug(f"Experiment parameters: {str(annotation_task)}")
            annotation_task.begin(verbose=verbose)
            output_path = _file_util.generate_datetime_filename(
                output_dir=output_dir, file_ending=".json"
            )
            annotation_task.to_json_file(output_path)
            logger.info(f"Annotation saved to {output_path}")
        except Exception:
            logger.exception("Annotation experiment aborted due to error.")
