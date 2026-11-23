-- Feature table: one row per client, built from raw `clients`.
-- Applies the cleaning plan from notebooks/eda_1.ipynb. Raw table is never modified.
-- sex and marriage are deliberately excluded (fairness).

CREATE OR REPLACE TABLE features AS
SELECT
    client_id,
    limit_bal,
    age,

    -- cleaning: undocumented education codes (0, 5, 6) -> 4 ("other")
    CASE WHEN education IN (1, 2, 3) THEN education ELSE 4 END        AS education,

    -- A. payment behaviour (delay = max(pay, 0); cap at 3 applied later for LR only)
    GREATEST(pay_1, pay_2, pay_3, pay_4, pay_5, pay_6, 0)              AS max_delay,
    (pay_1 > 0)::INT + (pay_2 > 0)::INT + (pay_3 > 0)::INT
      + (pay_4 > 0)::INT + (pay_5 > 0)::INT + (pay_6 > 0)::INT        AS months_late,
    GREATEST(pay_1, 0)                                                 AS recent_delay,

    -- B. utilisation (limit_bal is never 0, min 10,000)
    bill_amt1 / limit_bal                                              AS util_recent,
    (bill_amt1 + bill_amt2 + bill_amt3
      + bill_amt4 + bill_amt5 + bill_amt6) / 6.0 / limit_bal          AS util_avg,
    GREATEST(bill_amt1, bill_amt2, bill_amt3,
             bill_amt4, bill_amt5, bill_amt6) / limit_bal              AS util_max,

    -- C. repayment and trend
    -- September payment pays August statement; bill <= 0 means nothing owed -> ratio 1
    CASE WHEN bill_amt2 <= 0 THEN 1.0
         ELSE LEAST(pay_amt1 / bill_amt2, 1.0) END                     AS pay_ratio_recent,
    (bill_amt1 - bill_amt6) / limit_bal                                AS bill_trend,
    GREATEST(pay_1, 0) - GREATEST(pay_6, 0)                            AS delay_trend,
    LN(1 + pay_amt1)                                                   AS log_pay_amt1,

    "default"
FROM clients;
