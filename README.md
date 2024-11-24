# Cosmonaut

![Cosmonaut](assets/cosmonaut_header.jpg)

Helping you find structure in the cosmos of data.

Cosmonaut is a tool for creating classifiers for unstructured data. Bring you own AI provider, provide minimal configuration, and get started in minutes.

> Cosmonaut is currently in active development, and may not be suitable for production use.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Step by Step Example](#step-by-step-example)
- [More Examples](#more-examples)
- [Development](#development)
- [WIP](#wip)

## Features

### Support Any Classification Scenario

- Single label - Classify an input with a single label
- Multi label - Classify an input with multiple labels
- Multi category, single label - Classify an input with multiple categories, each with a single label
- Multi category, multi label - Classify an input with multiple categories, each with multiple labels

### Produce Structured Outputs

- Predictions are returned in structured outputs
- Supports structured outputs feature provided by the AI Providers, when available

### Mix Unstructured Data Formats

- Text
- Images (coming soon)

### Use Any AI Providers

- Anthropic
- Gemini
- OpenAI
- Support for other providers which support OpenAI REST API

### Easy to Use

- Configuration file generation via CLI
- Save fully built configurations for future use
- Examples provided for common use cases

## Installation

Install Cosmonaut:

```bash
pip install cosmonaut
```

Or install the latest development version:

```bash
pip install git+https://github.com/am1tyadav/cosmonaut.git
```

## Quickstart

Provided you have a configuration file (eg. [config.yml](https://github.com/am1tyadav/cosmonaut/tree/main/examples/single_label/config.yml)) describing your classification problem, and have a Pandas DataFrame for the inputs, you can run Cosmonaut with:

```python
from cosmonaut import Cosmonaut

# Define how to create the prompt for each input
def create_prompt(row: pd.DataFrame) -> str:
    return f"Some context about the input: {row['text']}"

# Create and run a Cosmonaut classifier
predictions = Cosmonaut("/config/filepath", create_prompt).run(...)
```

## Step by Step Example

[Example Source Code](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)

A multi-label, single category classification problem allows each input to have multiple labels within the same category. In our example, we will classify "Gift Suggestions" based on a user's text description of their interests. Given a text input describing a user's preferences, the classifier will generate one or more appropriate gift suggestions as labels within the "Gift Suggestions" category.

### Step 1: Create a configuration file

Cosmonaut expects the classification problem to be _described_ in a configuration file. Additionally, this config file is used to configure the AI provider.

Please take a look at one of the example configuration files to see what is required to describe the classification problem. [config.yml](https://github.com/am1tyadav/cosmonaut/blob/main/examples/multi_label/config.yml)

In order to create the configuration file, we can use the CLI:

```bash
cosmonaut-config
```

Answer the questions, and a configuration file will be created for you.

### Step 2: Create a system instructions file

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

### Step 3: Create a prompt function

While the system prompt is populated automatically when we instantiate a `Cosmonaut` object, we need to create a prompt function that will be used to create the prompt for each input. This prompt function will be passed to the `Cosmonaut` object when it is instantiated. For this example, we will create the following prompt function:

```python
def create_prompt(inputs: pd.DataFrame) -> str:
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


def create_prompt(inputs: pd.DataFrame) -> str:
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

## More Examples

### 1. Single Label, Single Category Classification with a Local Model

You can use a local model to make predictions as long as the model runner is compatible with the OpenAI REST API (eg. LM Studio).

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/single_label)

### 2. Multi Label, Single Category Classification with Anthropic

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)

### 3. Multi Label, Multi Category Classification with Gemini

Following examples uses a configuration that does not provide label descriptions or require reasons for predictions as both of these are optional.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_category)

### 4. Distributed Predictions with PySpark

Cosmonaut uses Pandas to handle the input data - however, this can be a bottleneck when dealing with large datasets. Fortunately, we can use Dask, Ray or Spark to parallelize the data processing. Following example uses PySpark to parallelize the data processing.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/pyspark)

## Development

Contributions are welcome - Please open an issue or submit a pull request. Please feel free to open issues for feature requests as well. Some considerations when developing:

- The project should support Python 3.10+
- [trunk.io](https://trunk.io) is used but only as a linter

## WIP

- Unit Tests
- Return usage information in the response
- Support ingesting image data
- Documentation
  - Code
  - Concepts
- Examples
  - Artefact logging with MLFlow
  - Streaming Predictions with DuckDB
