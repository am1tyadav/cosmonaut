# Step by Step Guide

[Example Source Code](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)

A multi-label, single category classification problem allows each input to have multiple labels within the same category. In our example, we will classify "Gift Suggestions" based on a user's text description of their interests. Given a text input describing a user's preferences, the classifier will generate one or more appropriate gift suggestions as labels within the "Gift Suggestions" category.

## Step 1: Create a configuration file

Cosmonaut expects the classification problem to be _described_ in a configuration file. Additionally, this config file is used to configure the AI provider.

Please take a look at one of the example configuration files to see what is required to describe the classification problem. [config.yml](https://github.com/am1tyadav/cosmonaut/blob/main/examples/multi_label/config.yml)

In order to create the configuration file, we can use the CLI:

```bash
cosmonaut-config
```

Answer the questions, and a configuration file will be created for you.

## Step 2: Create a system instructions file

Now that the classifier is described, we need to provide the AI provider with instructions on how to generate the predictions. We can do this by creating a system instructions file. In the configuration file, we can specify the system instructions file to use:

```yaml
classifier:
  instructions_filename: instructions.txt
```

Next, we will create this instructions file:

```txt
You are an expert gift recommender. You can recommend either one or two  gifts for a user given some information about their preferences.
```

There is no need to provide any examples in the system instructions file as these will be created automatically from the examples provided in the configuration file.

## Step 3: Create a prompt function

While the system prompt is populated automatically when we instantiate a `Cosmonaut` object, we need to create a prompt function that will be used to create the prompt for each input. This prompt function will be passed to the `Cosmonaut` object when it is instantiated. For this example, we will create the following prompt function:

```python
def create_prompt(inputs: pd.DataFrame) -> str:
    text = inputs["text"]
    return f"Please suggest one or two gifts for the following user: {text}"
```

## Step 4: Create and run a Cosmonaut classifier

Finally, we can create a Cosmonaut classifier, and run it on some data. A full example is provided below:

```python
from pathlib import Path

import pandas as pd

from cosmonaut import Cosmonaut


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["text"]
    return f"Please suggest one or two gifts for the following user: {text}"


if __name__ == "__main__":
    inputs = pd.DataFrame(
        {
            "text": [
                "I likes to play video games. I buy them online often with gift cards that I get on my birthday.",
                "I dont really do online shopping but I do like cars",
            ]
        }
    )

    config_filepath = Path(__file__).parent / "config.yml"
    response = Cosmonaut(config_filepath, create_prompt).run(inputs)

    print(response.head())
```

You can expect to see something like the following output for each of the inputs:

```json
{
  "success": true,
  "info": "success",
  "detail": null,
  "predictions": {
    "predictions": [
      {
        "category": "Gift Suggestion",
        "labels": [
          {
            "label": "Video Game",
            "reason": "The user explicitly states they enjoy playing video games and plays them frequently"
          },
          {
            "label": "Gift Card",
            "reason": "The user mentions they regularly use gift cards to purchase video games online"
          }
        ]
      }
    ]
  }
}
```
