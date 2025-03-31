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

import backend
import util
import tasks


logger = logging.getLogger(Path(__file__).name)


class DiscussionExperiment:
    """
    An Experiment where multiple, randomized synthetic discussions will take place.
    """

    def __init__(
        self,
        topics: list[str],
        users: list[backend.actors.LLMActor],
        moderator: backend.actors.LLMActor | None = None,
        next_turn_manager: backend.turn_manager.TurnManager | None = None,
        history_ctx_len: int = 3,
        num_turns: int = 10,
        num_active_users: int = 2,
        num_discussions: int = 5,
    ):
        self.topics = topics
        self.users = users
        self.moderator = moderator

        if next_turn_manager is None:
            logger.warning("No TurnManager selected: Defaulting to round-robin strategy.")
        self.next_turn_manager = next_turn_manager

        self.history_ctx_len = history_ctx_len
        self.num_active_users = num_active_users
        self.num_discussions = num_discussions
        self.num_turns = num_turns

    def begin(self, discussions_output_dir: Path = Path("./output")) -> None:
        """
        Begin the experiment by generating and executing a set of discussions.
        The results will be written as JSON files at the specified output directory
        """
        discussions = self._generate_discussion_experiments()
        self._run_all_discussions(discussions, discussions_output_dir)

    def _generate_discussion_experiments(self) -> list[tasks.Discussion]:
        """Generate experiments from the basic configurations and wrap them into
        Conversation objects.

        :param yaml_data: the serialized experiment configurations
        :type yaml_data: dict
        :param llm: the wrapped LLM
        :type llm: model.Model
        :return: a list of Conversation objects containing the experiments
        :rtype: _type_
        """
        experiments = []
        for _ in range(self.num_discussions):
            experiments.append(self._create_synthetic_discussion())
        return experiments

    def _create_synthetic_discussion(self):
        rand_topic = random.choice(self.topics)
        rand_users = random.sample(self.users, k=self.num_active_users)

        if self.next_turn_manager is None:
            next_turn_manager = backend.turn_manager.RoundRobbin()
        else:
            next_turn_manager = self.next_turn_manager
        next_turn_manager.initialize_names([user.name for user in rand_users])

        return tasks.Discussion(
            users=rand_users,
            moderator=self.moderator,
            history_context_len=self.history_ctx_len,
            conv_len=self.num_turns,
            seed_opinion=rand_topic,
            seed_opinion_user=random.choice(rand_users).name,
            next_turn_manager=next_turn_manager,
        )

    @util.output_util.timing
    def _run_all_discussions(
        self, discussions: list[tasks.Discussion], output_dir: Path
    ) -> None:
        """Creates experiments by combining the given input data, then runs each one sequentially.

        :param llm: The wrapped LLM
        :type llm: model.Model
        :param yaml_data: the serialized experiment configurations
        :type yaml_data: dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, discussion in enumerate(discussions):
            logging.info(f"Running experiment {i+1}/{len(discussions)+1}...")
            self._run_single_discussion(discussion=discussion, output_dir=output_dir)

        logger.info("Finished synthetic discussion generation.")

    @util.output_util.timing
    def _run_single_discussion(
        self, discussion: tasks.Discussion, output_dir: Path
    ) -> None:
        """Run a single discussion, then save its output to a auto-generated file.

        :param discussion: A Conversation object.
        :type discussion: generation.Conversation
        """
        try:
            logger.info("Beginning conversation...")
            logger.debug(f"Experiment parameters: {str(discussion)}")

            start_time = time.time()
            discussion.begin(verbose=True)
            output_path = util.file_util.generate_datetime_filename(
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
    An Experiment where multiple synthetic discussions are annotated by LLM-based annotators.
    """

    def __init__(
        self,
        annotators: list[backend.actors.LLMActor],
        history_ctx_len: int,
        include_mod_comments: bool,
    ):
        self.annotators = annotators
        self.history_ctx_len = history_ctx_len
        self.include_mod_comments = include_mod_comments

    def begin(self, discussions_dir: Path, output_dir: Path) -> None:
        """
        Begin the annotation experiment by generating and executing annotation tasks.
        The results will be written as JSON files in the specified output directory.
        """
        if not discussions_dir.is_dir():
            raise OSError(
                f"Discussions directory ({discussions_dir}) is not a directory"
            ) from None

        output_dir.mkdir(parents=True, exist_ok=True)

        annotation_tasks = self._generate_annotation_tasks(discussions_dir)
        self._run_all_annotations(annotation_tasks, output_dir)

    def _generate_annotation_tasks(
        self, discussions_dir: Path
    ) -> list[tasks.Annotation]:
        """Generate annotation experiments for each discussion and each annotator persona."""
        annotation_tasks = []
        for annotator in self.annotators:
            for discussion_path in discussions_dir.iterdir():
                annotation_task = self._create_annotation_task(
                    annotator, discussion_path
                )
                annotation_tasks.append(annotation_task)
        return annotation_tasks

    def _create_annotation_task(
        self, annotator: backend.actors.LLMActor, conv_logs_path: Path
    ) -> tasks.Annotation:
        return tasks.Annotation(
            annotator=annotator,
            conv_logs_path=conv_logs_path,
            history_ctx_len=self.history_ctx_len,
            include_moderator_comments=self.include_mod_comments,
        )

    @util.output_util.timing
    def _run_all_annotations(
        self, annotation_tasks: list[tasks.Annotation], output_dir: Path
    ) -> None:
        """Runs all annotation tasks sequentially and saves results."""
        for i, annotation_task in enumerate(annotation_tasks):
            logger.info(f"Running annotation {i+1}/{len(annotation_tasks)}...")
            self._run_single_annotation(annotation_task, output_dir)

        logger.info("Finished annotation generation.")

    @util.output_util.timing
    def _run_single_annotation(
        self, annotation_task: tasks.Annotation, output_dir: Path
    ) -> None:
        """Executes a single annotation experiment and saves its output."""
        try:
            logger.info("Beginning annotation...")
            logger.debug(f"Experiment parameters: {str(annotation_task)}")
            annotation_task.begin_annotation(verbose=True)
            output_path = util.file_util.generate_datetime_filename(
                output_dir=output_dir, file_ending=".json"
            )
            annotation_task.to_json_file(output_path)
            logger.info(f"Annotation saved to {output_path}")
        except Exception:
            logger.exception("Annotation experiment aborted due to error.")
