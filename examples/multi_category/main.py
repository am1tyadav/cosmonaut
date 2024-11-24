from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cosmonaut import Cosmonaut

load_dotenv()


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["text"]
    return f"Please suggest the age and gender labels based on the book summary: {text}"


if __name__ == "__main__":
    inputs = pd.DataFrame(
        {
            "text": [
                (
                    "A fantastical tale of a king who discovers a dark world "
                    "beneath his kingdom. The king must make difficult and "
                    "cruel decisions to save his kingdom from the evil forces."
                )
            ]
        }
    )

    config_filepath = Path(__file__).parent / "config.yml"
    classifier = Cosmonaut(config_filepath, create_prompt)
    response = classifier.run(inputs)

    print(response[classifier.config.data.result_column].tolist())

    # Save config for future use or deployment
    # classifier.save_config(Path(__file__).parent / "config.json")
