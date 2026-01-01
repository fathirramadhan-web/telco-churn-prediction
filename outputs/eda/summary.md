# EDA — Telco Customer Churn

- Rows: 7043 | Columns: 21
- Duplicated customerID: 0
- Missing cells (raw): 0
- TotalCharges blank strings: 11 -> imputed median
- Churn rate: 26.54% (No 5174, Yes 1869)

## Numeric summary

|                |   count |    mean |     std |   min |    25% |     50% |     75% |     max |
|:---------------|--------:|--------:|--------:|------:|-------:|--------:|--------:|--------:|
| SeniorCitizen  |    7043 |    0.16 |    0.37 |  0    |   0    |    0    |    0    |    1    |
| tenure         |    7043 |   32.37 |   24.56 |  0    |   9    |   29    |   55    |   72    |
| MonthlyCharges |    7043 |   64.76 |   30.09 | 18.25 |  35.5  |   70.35 |   89.85 |  118.75 |
| TotalCharges   |    7043 | 2281.92 | 2265.27 | 18.8  | 402.22 | 1397.48 | 3786.6  | 8684.8  |

## Categorical churn rates

### gender

| gender   |   mean |   size |
|:---------|-------:|-------:|
| Female   |  0.269 |   3488 |
| Male     |  0.262 |   3555 |

### SeniorCitizen

|   SeniorCitizen |   mean |   size |
|----------------:|-------:|-------:|
|               1 |  0.417 |   1142 |
|               0 |  0.236 |   5901 |

### Partner

| Partner   |   mean |   size |
|:----------|-------:|-------:|
| No        |  0.33  |   3641 |
| Yes       |  0.197 |   3402 |

### Dependents

| Dependents   |   mean |   size |
|:-------------|-------:|-------:|
| No           |  0.313 |   4933 |
| Yes          |  0.155 |   2110 |

### Contract

| Contract       |   mean |   size |
|:---------------|-------:|-------:|
| Month-to-month |  0.427 |   3875 |
| One year       |  0.113 |   1473 |
| Two year       |  0.028 |   1695 |

### InternetService

| InternetService   |   mean |   size |
|:------------------|-------:|-------:|
| Fiber optic       |  0.419 |   3096 |
| DSL               |  0.19  |   2421 |
| No                |  0.074 |   1526 |

### PaymentMethod

| PaymentMethod             |   mean |   size |
|:--------------------------|-------:|-------:|
| Electronic check          |  0.453 |   2365 |
| Mailed check              |  0.191 |   1612 |
| Bank transfer (automatic) |  0.167 |   1544 |
| Credit card (automatic)   |  0.152 |   1522 |

### PaperlessBilling

| PaperlessBilling   |   mean |   size |
|:-------------------|-------:|-------:|
| Yes                |  0.336 |   4171 |
| No                 |  0.163 |   2872 |

### TechSupport

| TechSupport         |   mean |   size |
|:--------------------|-------:|-------:|
| No                  |  0.416 |   3473 |
| Yes                 |  0.152 |   2044 |
| No internet service |  0.074 |   1526 |
