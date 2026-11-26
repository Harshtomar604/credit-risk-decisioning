from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "credit.duckdb"
OUT_PATH = ROOT / "split_output.txt"
TEST_SIZE = 0.2
SEED = 42


def main():
    con = duckdb.connect(str(DB_PATH))
    ids = con.execute('SELECT client_id, "default" FROM features ORDER BY client_id').df()

    train_ids, test_ids = train_test_split(
        ids["client_id"],
        test_size=TEST_SIZE,
        stratify=ids["default"],
        random_state=SEED,
    )

    split_df = pd.concat([
        pd.DataFrame({"client_id": train_ids, "split": "train"}),
        pd.DataFrame({"client_id": test_ids, "split": "test"}),
    ])
    con.execute("CREATE OR REPLACE TABLE split AS SELECT * FROM split_df")

    summary = con.execute("""
        SELECT s.split,
               COUNT(*)            AS n_clients,
               AVG(f."default")    AS default_rate
        FROM split s
        JOIN features f USING (client_id)
        GROUP BY s.split
        ORDER BY s.split DESC
    """).df()
    overlap = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT client_id FROM split GROUP BY client_id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    con.close()

    text = "\n".join([
        summary.round(4).to_string(index=False),
        f"clients in both sets: {overlap}",
    ])
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
