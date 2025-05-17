# statistical-analysis

A Python project for statistical analysis and SAT data processing.

## Features

- Data cleaning and preprocessing for tabular and time series data
- Hypothesis testing (Mann-Whitney U, Chi-Square, Kruskal-Wallis)
- Linear regression modeling with multicollinearity checks (VIF)
- Time series analysis and forecasting (ARIMA, SARIMA)
- Automated result and plot generation
- Modular, extensible codebase

## Features

- Data import and export
- Statistical computations (mean, median, standard deviation, correlation, etc.)
- Visualization tools (histograms, scatter plots, box plots)
- Modular and extensible design

## Types of Analysis

- **Hypothesis Testing:** Explore relationships and differences in     student performance data.
- **Linear Regression:** Identify and quantify predictors of student final scores.
- **Time Series Analysis:** Decompose, model, and forecast stock prices.


## Installation

```bash
git clone https://github.com/yourusername/sat-private.git
cd sat-private
pip install -r requirements.txt
```
## Usage

1. Place your datasets in the `Datasets/` directory:
    - `Students_Grading_Dataset.csv`
    - `apple_only.csv`
2. Run the main script:
    ```bash
    python main.py
    ```
3. Results and plots will be saved in the `output/` directory.

## Project Structure

```
sat-private/
├── Data_Cleaning/
│   └── preprocessing.py
├── Regression/
│   └── linear_regression.py
├── Hypothesis/
│   └── hypothesis_testing.py
├── Time_Series/
│   └── time_series_analysis.py
├── Datasets/
│   ├── Students_Grading_Dataset.csv
│   └── apple_only.csv
├── output/
│   └── [results and plots]
├── main.py
└── README.md
```

## License

This project is licensed under the MIT License.