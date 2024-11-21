from pathlib import Path
from pprint import pprint

import pandas as pd

from cosmonaut import Cosmonaut


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["text"]
    return f"Please classify the following user feedback: {text}"


if __name__ == "__main__":
    inputs = pd.DataFrame(
        {
            "text": [
                "I used to like it but not anymore.",
                "I think it's a really good product.",
            ]
        }
    )

    config_filepath = Path(__file__).parent / "config.yml"
    classifier = Cosmonaut(config_filepath, create_prompt)
    response = classifier.run(inputs)

    pprint(response[classifier.config.data.result_column].tolist())
