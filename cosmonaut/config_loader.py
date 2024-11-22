from pathlib import Path

import yaml

from cosmonaut.data_models import Config


def _get_prediction_schema(require_reason: bool) -> dict:
    if require_reason:
        label_detail = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["label", "reason"],
        }
    else:
        label_detail = {
            "type": "array",
            "items": {"type": "string"},
        }
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
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


def _build_instructions(config: Config, dirpath: Path | None = None) -> str:
    """Builds the instructions for the classifier."""

    if dirpath is None:
        instructions_filepath = config.classifier.instructions_filename
    else:
        instructions_filepath = dirpath / config.classifier.instructions_filename

    with open(instructions_filepath, "r") as _file:
        instructions = _file.read()

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

    examples_text = ""

    if config.classifier.examples:
        examples_text = (
            "Here are some concrete examples of how to make the predictions:\n\n"
        )

    for index, example in enumerate(config.classifier.examples):
        prediction_response = "["

        for prediction in example.predictions:
            prediction_response += f"{prediction.model_dump()},"

        prediction_response += "]"
        examples_text += f"## Example {index + 1}:\n{prediction_response}\n"

    # Todo: write a function to simplify schema
    schema = _get_prediction_schema(config.classifier.require_reason)

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
    """Loads the configuration from the specified file path or from
    a dictionary. If a dictionary is provided, the path to the instructions
    file needs to be a full path otherwise it will be assumed to be relative
    to the directory containing the configuration file.
    """

    if isinstance(config_or_config_path, (Path, str)):
        with open(config_or_config_path, "r") as _file:
            data = yaml.safe_load(_file)
        instructions_dirpath = config_or_config_path.parent
    elif isinstance(config_or_config_path, dict):
        data = config_or_config_path
        instructions_dirpath = None
    else:
        raise ValueError(f"Invalid config type: {type(config_or_config_path)}")

    config = Config.model_validate(data)
    config.classifier.instructions = _build_instructions(config, instructions_dirpath)
    return config
