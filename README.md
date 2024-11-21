# Cosmonaut

![Image](assets/cosmonaut_header.jpg)

Helping you find structure in the cosmos of data.

Cosmonaut is a tool for creating classifiers for unstructured data. Bring you own AI provider, provide minimal configuration, and get started in minutes.

> Cosmonaut is currently in active development, and may not be suitable for production use.

## Features

### Supports a Number of Classification Scenarios

| Scenario                          | Description                                        | Example                                                                         |
| --------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------- |
| Single label, individual category | Assigns one label from a single category           | Sentiment classification: Positive, Negative, or Neutral                        |
| Single label, multiple categories | Assigns one label for each of multiple categories  | Topic: Tech/Finance/Health AND Sentiment: Positive/Negative                     |
| Multi label, individual category  | Assigns multiple labels from a single category     | Topic tags: Can select any combination of Tech, Finance, and Health             |
| Multi label, multiple categories  | Assigns multiple labels across multiple categories | Topics: Tech + Finance AND Languages: English + Spanish (multiple per category) |

### Produces Structured Outputs

- Predictions are returned in structured outputs ✅
- Supports structured outputs feature provided by the AI Providers, when available ⏳

### Supports a Number of Unstructed Data Formats

- Text ✅
- Images ⏳
- Audio ⏳
- Video ⏳

### Supports a Number of AI Providers

- Anthropic
  - Via Official REST API ✅
- Gemini
  - Via Official REST API ⏳
  - Via OpenAI Compatible REST API ✅
- OpenAI
  - Via Official REST API ✅
- Support for other providers with OpenAI compatible REST API ✅

### Easy to Use

- Configuration file generation via CLI ✅
- Examples ✅

## Installation

Install Cosmonaut:

```bash
pip install git+https://github.com/am1tyadav/cosmonaut.git
```

## Quickstart

Provided you have a configuration file (eg. [config.yml](https://github.com/am1tyadav/cosmonaut/tree/main/examples/single_label/config.yml)) describing your classification problem, and have a Pandas DataFrame for the inputs, you can run Cosmonaut with:

```python
from pathlib import Path
import pandas as pd
from cosmonaut import Cosmonaut

config_filepath: Path = ...
inputs: pd.DataFrame = ...

# Define how to create the prompt for each input
def create_prompt(row):
    return f"Some context about the input: {row['text']}"

# Create and run a Cosmonaut classifier
classifier = Cosmonaut(config_filepath, create_prompt)
predictions = classifier.run(inputs)
```

## Example - Multi Label, Single Category Classification

[Example Source Code](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)

A multi-label, single category classification problem allows each input to have multiple labels within the same category. In our example, we will classify "Gift Suggestions" based on a user's text description of their interests. Given a text input describing a user's preferences, the classifier will generate one or more appropriate gift suggestions as labels within the "Gift Suggestions" category.

### Step 1: Create a configuration file

Cosmonaut expects the classification problem to be _described_ in a configuration file. Additionally, this config file is used to configure the AI provider. For this example, we will describe the classifier in the following way:

```yaml
classifier:
  categories:
    0: # The ID of the category. In this example, we have just the one category
      name: Gift Suggestion # The name of the category
      description: A gift suggestion based on user preferences.
      max_predictions: 2 # The maximum number of predictions to generate for this category. We will generate up to 2 predictions for this category.
      labels:
        0: # The ID of the label. In this example, we have 4 labels
          name: Automobile # The name of the label
          description: a car, or any vehicle with 4 wheels
        1:
          name: Video Game
          description: a game played on a computer, console, or handheld device
        2:
          name: Book
          description: a book, or any written work
        3:
          name: Gift Card
          description: a gift card, or a gift certificate
```

It is often a really good idea to provide a few examples to the AI provider. We can do this by adding a `examples` section to the configuration file:

```yaml
classifier:
    examples:
      - prediction: # First example
          0: # The ID of the category the following prediction belongs to
            name: Gift Suggestion # The name of the category
            labels: # Predicted labels for this category
              1: # The ID of the label
                name: Video Game # The name of the label
                reason: The user is interested in video games. # The reason for the prediction
              3:
                name: Gift Card
                reason: The user likes to use gift cards to buy video games online.
    - prediction:
        0:
          name: Gift Suggestion
          labels:
            2:
              name: Book
              reason: The user likes to read books but don't have other interests.
```

We have provided 2 examples - the first one has two predicted labels, and the second one has one label for the singular category.

We also need to configure the AI provider:

```yaml
ai_client:
  ai_provider: anthropic
  name: claude-3-5-sonnet-20241022
  api_key_var: ANTHROPIC_API_KEY # Please do not use the API key directly in your code. Instead, use an environment variable.
  base_url: https://api.anthropic.com/v1
  timeout: 10
  max_tokens: 2048
```

And finally, we specify a column name for the predictions to be stored in:

```yaml
data:
  result_column: result
```

> 💡 Tip: While you can manually create the configuration file, a CLI is provided to generate the configuration file for you. After installing Cosmonaut, you can run `cosmonaut-config` to generate the configuration file interactively. Running `cosmonaut-config` will prompt you to enter the details of the classification problem and the AI provider.

### Step 2: Create a system instructions file

Now that the classifier is described, we need to provide the AI provider with instructions on how to generate the predictions. We can do this by creating a system instructions file. In the configuration file, we can specify the system instructions file to use:

```yaml
classifier:
  instructions_filename: instructions.txt
  instructions: null
```

Note that the `instructions` field is set to `null` because we will populating it when the config file is loaded.

Next, we will create the instructions file:

```txt
You are an expert gift recommender. You can recommend either one or two  gifts for a user given some information about their preferences.
```

There is no need to provide any examples in the system instructions file as these will be created automatically from the examples provided in the configuration file.

### Step 3: Create a prompt function

While the system prompt is populated automatically when we instantiate a `Cosmonaut` object, we need to create a prompt function that will be used to create the prompt for each input. This prompt function will be passed to the `Cosmonaut` object when it is instantiated. For this example, we will create the following prompt function:

```python
def create_prompt(inputs: pd.Series) -> str:
    text = inputs["text"]
    return f"Please suggest one or two gifts for the following user: {text}"
```

### Step 4: Create and run a Cosmonaut classifier

Finally, we can create a Cosmonaut classifier, and run it on some data. A full example is provided below:

```python
from pathlib import Path
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv

from cosmonaut import Cosmonaut

# Load environment variables from .env file
# Needed for the API key
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
    response: pd.DataFrame = classifier.run(inputs)

    print(response.head())
```

## Scaling Cosmonaut

Cosmonaut uses Pandas to handle the input data - however, this can be a bottleneck when dealing with large datasets. Fortunately, we can use Dask, Ray or Spark to parallelize the data processing. Example(s) provided below.

## List of Examples

- [Single Label, Single Category Classification](https://github.com/am1tyadav/cosmonaut/tree/main/examples/single_label)
- [Multi Label, Single Category Classification](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)
- [Multi Label, Multi Category Classification](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_category)
- [Distributed Predictions with PySpark](https://github.com/am1tyadav/cosmonaut/tree/main/examples/pyspark)

## Development

Contributions are welcome - Please open an issue or submit a pull request. Please feel free to open issues for feature requests as well.

Known issues:

- No test coverage
