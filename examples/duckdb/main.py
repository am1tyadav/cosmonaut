from pathlib import Path

import duckdb
import pandas as pd

from cosmonaut import Cosmonaut


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["sentence"]
    return f"Please classify the following sentence: {text}"


def extract_predicted_label(inputs: pd.Series) -> str:
    try:
        return inputs["predictions"]["predictions"][0]["labels"][0]
    except Exception:
        return "Error"


conn = duckdb.connect()
inputs = pd.read_csv(Path(__file__).parent / "data.csv")
classifier = Cosmonaut(Path(__file__).parent / "config.yml", create_prompt)
result_column = classifier.config.data.result_column


TOTAL_COUNT = len(inputs)
BATCH_SIZE = 5
offset = 0


conn.execute(
    f"CREATE TABLE preds (id BIGINT, language VARCHAR, sentence VARCHAR, {result_column} VARCHAR)"
)


while offset < TOTAL_COUNT:
    df = conn.execute(
        "SELECT * FROM inputs ORDER BY id LIMIT ? OFFSET ?", [BATCH_SIZE, offset]
    ).fetch_df_chunk()
    offset += BATCH_SIZE

    results = classifier.run(df)
    df[result_column] = results[result_column].apply(extract_predicted_label)

    conn.execute("INSERT INTO preds SELECT * FROM df")


query = conn.sql("SELECT * FROM preds")
query.show()
