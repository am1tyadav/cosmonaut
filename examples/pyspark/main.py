from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from cosmonaut import Cosmonaut


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["text"]
    return f"Please suggest one or two gifts for the following user: {text}"


def predict_on_batch(classifier: Cosmonaut) -> callable:
    def _predict_on_batch(batch_iterator: Iterator[pd.DataFrame]) -> Iterator:
        for batch in batch_iterator:
            yield classifier.run(batch)

    return _predict_on_batch


def main(config: dict, inputs: DataFrame, schema: StructType) -> DataFrame:
    num_partitions = 100
    classifier = Cosmonaut(config, create_prompt)
    inputs = inputs.rdd.repartition(num_partitions)

    # Distributed inference - only batch dataframes are loaded into memory in the workers
    return inputs.mapInPandas(func=predict_on_batch(classifier), schema=schema)
