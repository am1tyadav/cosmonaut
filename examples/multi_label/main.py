from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cosmonaut import Cosmonaut

load_dotenv()


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["text"]
    return f"Please suggest one or two gifts for the following user: {text}"


if __name__ == "__main__":
    inputs = pd.DataFrame(
        {
            "text": [
                (
                    "I am a 20 year old who likes to play video games."
                    "I buy them online often with gift cards that I get on my birthday."
                ),
                "I dont really do online shopping but I do like cars",
            ]
        }
    )

    config_filepath = Path(__file__).parent / "config.yml"
    classifier = Cosmonaut(config_filepath, create_prompt)
    response = classifier.run(inputs)

    print(response[classifier.config.data.result_column].tolist())
