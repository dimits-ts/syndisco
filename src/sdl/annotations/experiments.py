import logging
from pathlib import Path

from sdl.backend import actors
from ..util import file_util, output_util
from . import generation

logger = logging.getLogger(Path(__file__).name)


class AnnotationExperiment:
    """
    An Experiment where multiple synthetic discussions are annotated by LLM-based annotators.
    """

    def __init__(
        self,
        annotators: list[actors.LLMActor],
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
    ) -> list[generation.AnnotationConv]:
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
        self, annotator: actors.LLMActor, conv_logs_path: Path
    ) -> generation.AnnotationConv:
        return generation.AnnotationConv(
            annotator=annotator,
            conv_logs_path=conv_logs_path,
            history_ctx_len=self.history_ctx_len,
            include_moderator_comments=self.include_mod_comments,
        )

    @output_util.timing
    def _run_all_annotations(
        self, annotation_tasks: list[generation.AnnotationConv], output_dir: Path
    ) -> None:
        """Runs all annotation tasks sequentially and saves results."""
        for i, annotation_task in enumerate(annotation_tasks):
            logger.info(f"Running annotation {i+1}/{len(annotation_tasks)}...")
            self._run_single_annotation(annotation_task, output_dir)

        logger.info("Finished annotation generation.")

    @output_util.timing
    def _run_single_annotation(
        self, annotation_task: generation.AnnotationConv, output_dir: Path
    ) -> None:
        """Executes a single annotation experiment and saves its output."""
        try:
            logger.info("Beginning annotation...")
            logger.debug(f"Experiment parameters: {str(annotation_task)}")
            annotation_task.begin_annotation(verbose=True)
            output_path = file_util.generate_datetime_filename(
                output_dir=output_dir, file_ending=".json"
            )
            annotation_task.to_json_file(output_path)
            logger.info(f"Annotation saved to {output_path}")
        except Exception:
            logger.exception("Annotation experiment aborted due to error.")
