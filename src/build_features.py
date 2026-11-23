from pathlib import Path

import duckdb

DB_PATH = Path("data/credit.duckdb")
SQL_PATH = Path("sql/features.sql")
OUT_PATH = Path("features_output.txt")


def main():
    sql = SQL_PATH.read_text()

    con = duckdb.connect(str(DB_PATH))
    con.execute(sql)
    df = con.execute("SELECT * FROM features").df()
    con.close()

    lines = [
        f"shape: {df.shape}",
        f"columns: {list(df.columns)}",
        f"total nulls: {int(df.isna().sum().sum())}",
        f"education values: {sorted(df['education'].unique().tolist())}",
        f"sex/marriage present: {'sex' in df.columns or 'marriage' in df.columns}",
        "",
        "=== describe ===",
        df.drop(columns="client_id").describe().T.round(3).to_string(),
        "",
        "=== default rate by max_delay ===",
        df.groupby("max_delay")["default"].agg(["count", "mean"]).round(3).to_string(),
        "",
        "=== default rate by months_late ===",
        df.groupby("months_late")["default"].agg(["count", "mean"]).round(3).to_string(),
    ]
    text = "\n".join(lines)

    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
