"""
Generates experiments by combining the sample input configs. 
Each experiment is packaged into a Conversation object (@see generation.py).
Then runs each experiment sequentially, and saves the output to disk as an auto-generated file.
"""

import os
import random
import logging
import time
from pathlib import Path

from . import generation
from ..util import file_util, output_util
from ..backend import turn_manager
from ..backend import persona
from ..backend import actors
from ..backend import model


logger = logging.getLogger(Path(__file__).name)


@output_util.timing
def run_discussion_experiments(llm: model.BaseModel, yaml_data: dict) -> None:
    """Creates experiments by combining the given input data, then runs each one sequentially.

    :param llm: The wrapped LLM
    :type llm: model.Model
    :param yaml_data: the serialized experiment configurations
    :type yaml_data: dict
    """
    # Ensure output directory exists
    output_dir = yaml_data["discussions"]["files"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    experiments = _generate_discussion_experiments(llm=llm, yaml_data=yaml_data)
    for i, experiment in enumerate(experiments):
        logging.info(f"Running experiment {i+1}/{len(experiments)+1}...")
        _run_single_experiment(experiment=experiment, output_dir=output_dir)

    logger.info("Finished synthetic discussion generation.")


@output_util.timing
def _run_single_experiment(
    experiment: generation.Conversation, output_dir: Path
) -> None:
    """Run a single experiment, then save its output to a auto-generated file.

    :param experiment: A Conversation object describing the experiment.
    :type experiment: generation.Conversation
    :param output_dir: The directory where the auto-generated file with the
    experiment's output will be saved
    :type output_dir: Path
    """
    try:
        logger.info("Beginning conversation...")
        logger.debug(f"Experiment parameters: {str(experiment)}")

        start_time = time.time()
        experiment.begin_conversation(verbose=True)
        output_path = file_util.generate_datetime_filename(
            output_dir=output_dir, file_ending=".json"
        )
        logging.debug(f"Finished discussion in {(time.time() - start_time)} seconds.")

        experiment.to_json_file(output_path)
        logger.info(f"Conversation saved to {output_path}")
    except Exception:
        logger.exception("Experiment aborted due to error.")


def _generate_discussion_experiments(
    yaml_data: dict, llm: model.BaseModel
) -> list[generation.Conversation]:
    """Generate experiments from the basic configurations and wrap them into
    Conversation objects.

    :param yaml_data: the serialized experiment configurations
    :type yaml_data: dict
    :param llm: the wrapped LLM
    :type llm: model.Model
    :return: a list of Conversation objects containing the experiments
    :rtype: _type_
    """
    # Extract yaml configs
    paths = yaml_data["discussions"]["files"]
    turn_taking_config = yaml_data["discussions"]["turn_taking"]
    experiment_variables = yaml_data["discussions"]["experiment_variables"]

    # Paths for various required files and directories
    topics_dir = paths["topics_dir"]
    persona_path = paths["user_persona_path"]
    user_instruction_path = paths["user_instructions_path"]
    mod_instruction_path = paths["mod_instructions_path"]

    # Experiment variables
    num_experiments = experiment_variables["num_experiments"]
    num_users = experiment_variables["num_users"]
    include_mod = experiment_variables["include_mod"]

    personas = persona.from_json_file(persona_path)

    topics = file_util.read_files_from_directory(topics_dir)
    user_instructions = file_util.read_file(user_instruction_path)
    mod_instructions = file_util.read_file(mod_instruction_path)

    turn_taking_dict = {
        "conv_len": turn_taking_config["num_turns"],
        "history_ctx_len": turn_taking_config["history_ctx_len"],
        "turn_manager_type": turn_taking_config["turn_manager_type"],
        "turn_manager_config": {
            "respond_probability": turn_taking_config["respond_probability"]
        },
    }

    ctx_prompt = experiment_variables["context_prompt"]
    mod_attributes = experiment_variables["moderator_attributes"]
    mod_attributes = mod_attributes if include_mod else None

    experiments = []
    for _ in range(num_experiments):
        experiments.append(
            _create_synthetic_discussion(
                llm=llm,
                topics=topics,
                context=ctx_prompt,
                all_personas=personas,
                mod_attributes=mod_attributes,
                user_instructions=user_instructions,
                mod_instructions=mod_instructions,
                turn_manager_type=turn_taking_dict["turn_manager_type"],
                turn_manager_config=turn_taking_dict["turn_manager_config"],
                num_users=num_users,
                conv_len=turn_taking_dict["conv_len"],
                history_ctx_len=turn_taking_dict["history_ctx_len"],
            )
        )
    return experiments


def _create_synthetic_discussion(
    llm: model.BaseModel,
    topics: list[str],
    context: str,
    all_personas: list[persona.LlmPersona],
    mod_attributes: list[str] | None,
    user_instructions: str,
    mod_instructions: str | None,
    turn_manager_type: str,
    turn_manager_config: dict,
    num_users: int,
    conv_len: int,
    history_ctx_len: int,
) -> generation.Conversation:
    """
    Generate a synthetic discussion with users, a moderator, and a turn manager.

    This function creates a conversation simulation using an LLM for a specified number of users
    and a moderator, with topics, personas, and turn management strategies. The output is a
    structured conversation object.

    :param llm: The language model used to generate user and moderator responses.
    :type llm: model.Model
    :param topics: A list of topics to seed the discussion.
    :type topics: list[str]
    :param context: Contextual information shared with users and the moderator.
    :type context: str
    :param all_personas: A list of all possible personas that define user attributes and behaviors.
    :type all_personas: list[persona.LlmPersona]
    :param mod_attributes: Attributes for the moderator persona, defaults to None.
    :type mod_attributes: list[str] | None
    :param user_instructions: Instructions provided to the users.
    :type user_instructions: str
    :param mod_instructions: Instructions provided to the moderator, defaults to None.
    :type mod_instructions: str | None
    :param turn_manager_type: The type of turn manager to control turn-taking.
    :type turn_manager_type: str
    :param turn_manager_config: Configuration dictionary for the turn manager.
    :type turn_manager_config: dict
    :param num_users: Number of users participating in the discussion.
    :type num_users: int
    :param conv_len: Length of the conversation in terms of the number of turns.
    :type conv_len: int
    :param history_ctx_len: Length of the context history shared during the conversation.
    :type history_ctx_len: int
    :return: A synthetic conversation object.
    :rtype: generation.Conversation

    :raises AssertionError: If the number of users exceeds the number of provided personas.
    """
    assert num_users <= len(
        all_personas
    ), "Number of users must be less or equal to the number of provided personas"

    rand_personas = random.sample(all_personas, k=num_users)
    rand_topic = random.choice(topics)
    rand_user_names = [persona.username for persona in rand_personas]
    rand_user_attributes = [persona.to_attribute_list() for persona in rand_personas]

    users = _create_users(
        llm=llm,
        usernames=rand_user_names,
        attributes=rand_user_attributes,
        context=context,
        instructions=user_instructions,
    )
    mod = _create_moderator(
        llm=llm,
        mod_attributes=mod_attributes,
        instructions=mod_instructions,
        context=context,
    )

    next_turn_manager = turn_manager.turn_manager_factory(
        turn_manager_type, rand_user_names, config=turn_manager_config
    )

    return generation.Conversation(
        users=users,
        moderator=mod,
        history_context_len=history_ctx_len,
        conv_len=conv_len,
        seed_opinion=rand_topic,
        seed_opinion_user=random.choice(rand_user_names),
        next_turn_manager=next_turn_manager,
    )


def _create_users(
    llm: model.BaseModel,
    usernames: list[str],
    attributes: list[list[str]],
    context: str,
    instructions: str,
) -> list[actors.LLMUser]:
    """Create runtime LLMUser objects with the specified information.

    :param llm: the wrapped LLM instance that the users will use to generate text
    :type llm: model.Model
    :param usernames: a list of usernames for each of the users
    :type usernames: list[str]
    :param attributes: a list containing a list of personality/mood attributes for each user
    :type attributes: list[list[str]]
    :param context: the global context of the experiment
    :type context: str
    :param instructions: the instructions given to all LLM users (not the moderator)
    :type instructions: str
    :return: a list of initialized runtime LLMUser objects
    corresponding to the given characteristics
    :rtype: list[actors.LLMUser]
    """
    user_list = []

    assert len(usernames) == len(
        attributes
    ), "Number of usernames and user personality attribute lists must be the same"

    for username, user_attributes in zip(usernames, attributes):
        user_list.append(
            actors.LLMUser(
                model=llm,
                name=username,
                attributes=user_attributes,
                context=context,
                instructions=instructions,
            )
        )
    return user_list


def _create_moderator(
    llm: model.BaseModel | None,
    mod_attributes: list[str] | None,
    instructions: str | None,
    context: str,
) -> actors.LLMUser | None:
    """Create a LLMUser instance, which will assume the role of moderator.
    Returns None if any of the input arguments are set to None.

    :param llm: the wrapped LLM instance that the moderator will use to generate text
    :type llm: model.Model | None
    :param mod_attributes: a list of personality/mood attributes for the moderator
    :type mod_attributes: list[str] | None
    :param instructions: the moderator-specific instructions
    :type instructions: str | None
    :param context: the global context of the experiment
    :type context: str
    :return: an initialized LLMUser instance which will assume the role of moderator,
    or None if any of the fields above are set to None
    :rtype: actors.LLMUser | None
    """
    if mod_attributes is not None and llm is not None and instructions is not None:
        moderator = actors.LLMUser(
            model=llm,
            name="moderator",
            attributes=mod_attributes,
            context=context,
            instructions=instructions,
        )
    else:
        logger.warning("Warning: Generating conversation without moderator")
        moderator = None
    return moderator