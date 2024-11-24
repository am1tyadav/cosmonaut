# Examples

There is also a [step by step](https://github.com/am1tyadav/cosmonaut/tree/main/examples/step_by_step.md) guide if you're just getting started.

## 1. Single Label, Single Category Classification with a Local Model

You can use a local model to make predictions as long as the model runner is compatible with the OpenAI REST API (eg. LM Studio).

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/single_label)

## 2. Multi Label, Single Category Classification with Anthropic

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_label)

## 3. Multi Label, Multi Category Classification with Gemini

Following examples uses a configuration that does not provide label descriptions or require reasons for predictions as both of these are optional.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/multi_category)

## 4. Distributed Predictions with PySpark

Cosmonaut uses Pandas to handle the input data - however, this can be a bottleneck when dealing with large datasets. Fortunately, we can use Dask, Ray or Spark to parallelize the data processing. Following example uses PySpark to parallelize the data processing.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/pyspark)

## 5. Streaming Predictions with DuckDB

Instead of trying to apply predictions on the entire dataset at once, we can batch it using DuckDB.

[Link](https://github.com/am1tyadav/cosmonaut/tree/main/examples/duckdb)
