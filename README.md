# Cosmonaut

![Cosmonaut](assets/cosmonaut_header.jpg)

Helping you find structure in the cosmos of data.

Cosmonaut is a tool for creating classifiers for unstructured data. Bring you own AI provider, provide minimal configuration, and get started in minutes.

> Cosmonaut is currently in active development, and may not be suitable for production use.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Development](#development)
- [WIP](#wip)

## Features

### Support Any Classification Scenario

- Single category, single or multiple label
  - Example: Sentiment classification for user reviews (Positive/Negative/Neutral) where each review can have only one sentiment
  - Example: Tags prediction for blog posts (Sports, Technology, Politics, etc.) where each post can have many tags
- Multiple categories, single or multiple labels
  - Recipe classification for recipes where each recipe can have many ingredients (Chicken, Beef, Lettuce etc.) and one or more cuisines (Italian, Chinese, Indian etc.)

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
- Support for other providers which support OpenAI REST API (eg. LM Studio)

### Easy to Use

- Configuration file generation via CLI
- Save fully built configurations for future use or for deployment
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

## Examples

See the [examples](https://github.com/am1tyadav/cosmonaut/tree/main/examples) folder for more examples. There is also a [step by step](https://github.com/am1tyadav/cosmonaut/tree/main/examples/step_by_step.md) guide if you're just getting started.

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
