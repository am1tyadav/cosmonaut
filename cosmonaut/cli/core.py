import questionary
import typer
import yaml

app = typer.Typer()


@app.command()
def generate_config():
    ai_provider = questionary.select(
        "Select AI Provider:", choices=["anthropic", "openai", "gemini"]
    ).ask()

    model_name = questionary.text("Enter model name:").ask()
    api_key_var = questionary.text("Enter environment variable name for API key:").ask()
    base_url = questionary.text("Enter base URL (press enter for default):").ask()

    categories = {}

    while questionary.confirm("Add a category?").ask():
        category_id = len(categories)

        typer.echo(f"Category ID: {category_id}")

        cat_name = questionary.text("Enter category name:").ask()
        cat_desc = questionary.text("Enter category description:").ask()
        max_pred = questionary.text(
            "Enter maximum number of predictions allowed for this category:",
            validate=lambda x: x.isdigit(),
        ).ask()

        labels = {}

        while questionary.confirm("Add a label to this category?").ask():
            label_id = len(labels)

            typer.echo(f"Label ID: {label_id} for category {category_id}")

            label_name = questionary.text("Enter label name:").ask()
            label_desc = questionary.text("Enter label description:").ask()

            labels[label_id] = {"name": label_name, "description": label_desc}

        categories[category_id] = {
            "name": cat_name,
            "description": cat_desc,
            "max_predictions": int(max_pred),
            "labels": labels,
        }

    config = {
        "classifier": {
            "instructions_filename": "instructions.txt",
            "instructions": None,
            "categories": categories,
            "examples": [],
        },
        "ai_client": {
            "ai_provider": ai_provider,
            "name": model_name,
            "api_key_var": api_key_var,
            "base_url": base_url if base_url else None,
            "timeout": 10,
            "max_tokens": 2048,
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
