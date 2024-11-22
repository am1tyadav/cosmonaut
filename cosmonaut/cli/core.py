import questionary
import typer
import yaml

from cosmonaut.data_models import AIServiceProvider

app = typer.Typer()


@app.command()
def config():
    typer.echo("============== AI Provider Configuration ==============")

    ai_provider = questionary.select(
        "Select AI Provider:", choices=[option.value for option in AIServiceProvider]
    ).ask()

    model_name = questionary.text("Enter model name:").ask()
    api_key_var = questionary.text("Enter environment variable name for API key:").ask()
    base_url = questionary.text("Enter base URL:").ask()
    timeout = questionary.text(
        "Enter timeout in seconds:", validate=lambda x: x.isdigit()
    ).ask()
    max_tokens = questionary.text(
        "Enter max tokens:", validate=lambda x: x.isdigit()
    ).ask()

    categories = []

    typer.echo("============== Classifier Configuration ==============")

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
            "instructions_filename": "instructions.txt",
            "instructions": None,
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
        },
        "data": {"result_column": "result"},
    }

    output_path = questionary.text(
        "Enter output path for config.yml:", default="config.yml"
    ).ask()

    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    typer.echo(f"Configuration file generated at: {output_path}")
    typer.echo("Please edit the configuration file to add examples and other details.")


def main():
    app()
