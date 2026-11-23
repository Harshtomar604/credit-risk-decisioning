from pathlib import Path

import duckdb
from ucimlrepo import fetch_ucirepo

DB_PATH = Path("data/credit.duckdb")

COLUMNS = (
    ["limit_bal", "sex", "education", "marriage", "age"]
    + [f"pay_{i}" for i in range(1, 7)]
    + [f"bill_amt{i}" for i in range(1, 7)]
    + [f"pay_amt{i}" for i in range(1, 7)]
)


def main():
    DB_PATH.parent.mkdir(exist_ok=True)

    ds = fetch_ucirepo(id=350)
    df = ds.data.features.copy()
    df.columns = COLUMNS
    df["default"] = ds.data.targets.iloc[:, 0].astype(int).values
    df.insert(0, "client_id", range(1, len(df) + 1))

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE clients AS SELECT * FROM df")

    n, rate = con.execute('SELECT COUNT(*), AVG("default") FROM clients').fetchone()
    print("rows:", n)
    print("default rate:", round(rate, 4))
    print("columns:", [c[0] for c in con.execute("DESCRIBE clients").fetchall()])

    nulls = con.execute(
        "SELECT " + ", ".join(f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in COLUMNS)
        + " FROM clients"
    ).df().T
    print("total nulls:", int(nulls.sum().iloc[0]))

    con.close()


if __name__ == "__main__":
    main()