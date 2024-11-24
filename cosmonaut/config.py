from pathlib import Path

import yaml

from cosmonaut.data_models import AIServiceProvider, Config, KnownBaseURLs


def _update_known_properties(config: Config) -> Config:
    if config.ai_client.base_url is None:
        ai_provider = config.ai_client.ai_provider

        match ai_provider:
            case AIServiceProvider.ANTHROPIC:
                config.ai_client.base_url = KnownBaseURLs.ANTHROPIC.value
            case AIServiceProvider.OPENAI:
                config.ai_client.base_url = KnownBaseURLs.OPENAI.value
            case AIServiceProvider.GEMINI:
                config.ai_client.base_url = KnownBaseURLs.GEMINI.value
            case _:
                raise ValueError(f"Unsupported AI Provider type: {ai_provider}")
    return config


def _get_prediction_schema(
    require_reason: bool, ai_provider: AIServiceProvider
) -> dict:
    if require_reason:
        label_detail = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["label", "reason"],
        }

        if ai_provider != AIServiceProvider.GEMINI:
            label_detail["additionalProperties"] = False
    else:
        label_detail = {"type": "string"}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {"type": "string"},
                        "labels": {
                            "type": "array",
                            "items": label_detail,
                        },
                    },
                    "required": ["category", "labels"],
                },
            }
        },
        "required": ["predictions"],
    }

    if ai_provider == AIServiceProvider.GEMINI:
        del schema["additionalProperties"]
        del schema["properties"]["predictions"]["items"]["additionalProperties"]

    return schema


def _build_instructions(config: Config, dirpath: Path | None = None) -> str:
    """Builds the instructions for the classifier."""

    if config.classifier.instructions_filename is None:
        if config.classifier.instructions is None:
            raise ValueError(
                "Instructions must be provided either in the config or in a file."
            )
        instructions = config.classifier.instructions
    else:
        instructions_filepath = (
            dirpath / config.classifier.instructions_filename
            if dirpath is not None
            else config.classifier.instructions_filename
        )
        with open(instructions_filepath, "r", encoding="utf-8") as f:
            instructions = f.read()

    categories_text = (
        "You are expected to make predictions for the following categories:\n\n"
    )

    for category_config in config.classifier.categories:
        categories_text += f"## category name: {category_config.category}\n"
        categories_text += f"category description: {category_config.description}\n\n"
        categories_text += (
            f"You can only predict a maximmum of {category_config.max_predictions} "
            "label(s) for this particular category.\n"
        )
        categories_text += "The following labels are available for this category:\n"

        for label_config in category_config.labels:
            if config.classifier.label_descriptions_provided:
                categories_text += f"- label name: {label_config.label}"
                categories_text += f", label description: {label_config.description}\n"
            else:
                categories_text += f"- label: {label_config}\n"

        categories_text += "\n"

    if config.classifier.examples:
        examples_text = (
            "Here are some concrete examples of how to make the predictions:\n\n"
        )
    else:
        examples_text = ""

    for index, example in enumerate(config.classifier.examples):
        prediction_response = "["

        for prediction in example.predictions:
            prediction_response += f"{prediction.model_dump()},"

        prediction_response += "]"
        examples_text += f"## Example {index + 1}:\n{prediction_response}\n"

    # Todo: write a function to simplify schema
    schema = _get_prediction_schema(
        config.classifier.require_reason, config.ai_client.ai_provider
    )
    config.ai_client.prediction_schema = schema

    schema_text = f"\nThe following is the schema for the response:\n\n{schema}\n\n"
    schema_text += (
        "You need to return a JSON object that matches the schema, and nothing else. "
        "Do not include any additional information in your response. "
        "The response should be a valid JSON object. Do not include any further "
        "explanation or commentary beyond the JSON object itself. "
        "Do not include any code blocks or backticks in your response.\n"
    )

    final_text = f"{instructions}\n{categories_text}\n{examples_text}\n{schema_text}\n"
    return final_text


def load_config(config_or_config_path: dict | Path) -> Config:
    """Loads the config from the given path. If a path is given,
    it is assumed to be a yaml config. If a dict is given, it is
    assumed to be a previously saved config as json, and is not
    rebuilt."""

    if isinstance(config_or_config_path, Path):
        with open(config_or_config_path, "r") as f:
            data = yaml.safe_load(f)

        config = Config.model_validate(data)
        config = _update_known_properties(config)
        config.classifier.instructions = _build_instructions(
            config, config_or_config_path.parent
        )
        return config

    if isinstance(config_or_config_path, dict):
        """Loading from a previously saved config"""
        return Config.model_validate(config_or_config_path)

    raise ValueError(f"Invalid config type: {type(config_or_config_path)}")
