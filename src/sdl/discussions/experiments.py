import os
import random
import logging
from pathlib import Path

from . import generation
from ..util import file_util
from ..backend import turn_manager
from ..backend import persona
from ..backend import actors
from ..backend import model


logger = logging.getLogger(Path(__file__).name)


def run_experiments(llm: model.Model, yaml_data: dict) -> None:
    # Ensure output directory exists
    output_dir = yaml_data["discussions"]["files"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    experiments = _generate_experiments(llm=llm, yaml_data=yaml_data)
    for experiment in experiments:
        _run_single_experiment(experiment=experiment, output_dir=output_dir)

    logger.info("Finished synthetic discussion generation.")


def _run_single_experiment(
    experiment: generation.Conversation, output_dir: Path
) -> None:
    try:
        logger.info("Beginning conversation...")
        experiment.begin_conversation(verbose=True)
        output_path = file_util.generate_datetime_filename(
            output_dir=output_dir, file_ending=".json"
        )
        experiment.to_json_file(output_path)
        logger.info("Conversation saved to " + str(output_path))
    except Exception:
        logger.exception("Experiment aborted due to error.")


def _generate_experiments(yaml_data: dict, llm: model.Model):
    # Extract yaml configs
    paths = yaml_data["discussions"]["files"]
    turn_taking_config = yaml_data["discussions"]["turn_taking"]
    experiment_variables = yaml_data["discussions"]["experiment_variables"]

    # Paths for various required files and directories
    topics_dir = paths["topics_dir"]
    persona_dir = paths["user_persona_dir"]
    user_instruction_path = paths["user_instructions_path"]
    mod_instruction_path = paths["mod_instructions_path"]

    # Experiment variables
    num_experiments = experiment_variables["num_experiments"]
    num_users = experiment_variables["num_users"]
    include_mod = experiment_variables["include_mod"]

    persona_files = os.listdir(persona_dir)
    personas = [
        persona.LlmPersona.from_json_file(os.path.join(persona_dir, persona_file))
        for persona_file in persona_files
    ]

    topics = file_util.read_files_from_directory(topics_dir)
    user_instructions = file_util.read_file(user_instruction_path)
    mod_instructions = file_util.read_file(mod_instruction_path)

    turn_taking_dict = {
        "conv_len": turn_taking_config["num_turns"],
        "history_ctx_len": turn_taking_config["history_ctx_len"],
        "turn_manager_type": turn_taking_config["turn_manager_type"],
        "turn_manager_config": {
            "respond_probability": turn_taking_config[
                "rand_weighted_respond_probability"
            ]
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
    llm: model.Model,
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
        turn_manager=next_turn_manager,
    )


def _create_users(
    llm: model.Model,
    usernames: list[str],
    attributes: list[list[str]],
    context: str,
    instructions: str,
) -> list[actors.LLMUser]:
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
    llm: model.Model | None,
    mod_attributes: list[str] | None,
    instructions: str | None,
    context: str,
) -> actors.LLMUser | None:
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
