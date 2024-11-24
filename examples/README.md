# Examples

There is also a [tutorial](https://github.com/am1tyadav/cosmonaut/tree/main/examples/tutorial.md) if you're just getting started. Or take a look at the following examples:

## Single Label, Single Category Classification (LM Studio)

You can use a local model to make predictions as long as the model runner is compatible with the OpenAI REST API (eg. LM Studio).

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/single_label)

## Multi Label, Single Category Classification (Anthropic)

Predict mulitple labels per category.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)

## Multi Label, Multi Category Classification (Gemini)

Predict multiple labels for multiple categories for each input. Following example uses a configuration that does not provide label descriptions or require reasons for predictions as both of these are optional.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_category)

## Distributed Predictions with PySpark

Cosmonaut uses Pandas to handle the input data - however, this can be a bottleneck when dealing with large datasets. Fortunately, we can use Dask, Ray or Spark to parallelize the data processing. Following example uses PySpark to parallelize the data processing.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/pyspark)

## Batched Predictions with DuckDB (OpenAI)

Instead of applying predictions on the entire dataset at once, we can batch it using DuckDB. This is useful when the input dataset is too large to fit in memory. This example uses OpenAI's `response_schema` based structured outputs available in `gpt-4o-mini`.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/duckdb)
