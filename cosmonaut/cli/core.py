import questionary
import yaml

from cosmonaut.data_models import AIServiceProvider, OpenAIResponseFormat


def config():
    questionary.print("========== AI Provider Configuration ==========")

    ai_provider = questionary.select(
        "Select AI Provider:", choices=[option.value for option in AIServiceProvider]
    ).ask()
    model_name = questionary.text("Enter model name:").ask()
    api_key_var = questionary.text("Enter environment variable name for API key:").ask()
    base_url = (
        questionary.text("Enter base URL (or enter to use default):").ask() or None
    )
    timeout = questionary.text(
        "Enter timeout in seconds:", validate=lambda x: x.isdigit()
    ).ask()
    max_tokens = questionary.text(
        "Enter max tokens:", validate=lambda x: x.isdigit()
    ).ask()

    if ai_provider == AIServiceProvider.OPENAI.value:
        openai_response_format = questionary.select(
            "Select response format:",
            choices=[option.value for option in OpenAIResponseFormat],
        ).ask()
    else:
        openai_response_format = OpenAIResponseFormat.TEXT

    categories = []

    questionary.print("========== Classifier Configuration ==========")

    use_file = questionary.confirm("Use a file for instructions?").ask()

    instructions_filename = None
    instructions = None

    if use_file:
        instructions_filename = questionary.text(
            "Enter the name of the file containing instructions:"
        ).ask()
    else:
        instructions = questionary.text("Enter instructions:").ask()

    require_reason = questionary.confirm("Require reason for predictions?").ask()
    label_descriptions_provided = questionary.confirm(
        "Provide label descriptions?"
    ).ask()

    while questionary.confirm("Add a category?").ask():
        category_name = questionary.text("Enter category name or identifier:").ask()
        category_description = questionary.text("Enter category description:").ask()
        max_predictions = questionary.text(
            "Enter maximum number of predictions allowed for this category:",
            validate=lambda x: x.isdigit(),
        ).ask()

        labels = []

        while questionary.confirm("Add a label to this category?").ask():
            label_name = questionary.text("Enter label name:").ask()

            if label_descriptions_provided:
                label_description = questionary.text("Enter label description:").ask()
                labels.append({"name": label_name, "description": label_description})
            else:
                labels.append(label_name)

        categories.append(
            {
                "category": category_name,
                "description": category_description,
                "max_predictions": int(max_predictions),
                "labels": labels,
            }
        )

    config = {
        "classifier": {
            "instructions_filename": instructions_filename,
            "instructions": instructions,
            "require_reason": require_reason,
            "label_descriptions_provided": label_descriptions_provided,
            "categories": categories,
            "examples": [],
        },
        "ai_client": {
            "ai_provider": ai_provider,
            "name": model_name,
            "api_key_var": api_key_var,
            "base_url": base_url,
            "timeout": int(timeout),
            "max_tokens": int(max_tokens),
            "openai_response_format": openai_response_format,
        },
        "data": {"result_column": "result"},
    }

    output_path = questionary.text(
        "Enter output path for config.yml:", default="config.yml"
    ).ask()

    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    questionary.print(f"Configuration file generated at: {output_path}")
    questionary.print(
        "Please edit the configuration file to add examples and other details."
    )
