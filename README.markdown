# Cryptocurrency Liquidity Prediction Pipeline

## Project Overview
This project develops a machine learning pipeline to predict cryptocurrency liquidity using historical market data from CoinGecko (snapshot dated March 17, 2022). The pipeline includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, and a Streamlit web application for real-time predictions and data visualization. The goal is to provide actionable insights for market stability by forecasting liquidity metrics.

## Features
- **Data Preprocessing**: Cleans and transforms raw CSV data, handling missing values and converting percentage columns to floats.
- **Exploratory Data Analysis (EDA)**: Visualizes top coins by market cap and volume, price distributions, and correlation heatmaps.
- **Feature Engineering**: Creates liquidity-related features like `price_to_marketcap`, `volume_to_marketcap`, and `volatility_score`.
- **Model Training**: Evaluates multiple regressors (Random Forest, Gradient Boosting, Ridge, SVR) with GridSearchCV for hyperparameter tuning.
- **Model Deployment**: Deploys a Streamlit app for interactive EDA and liquidity predictions.
- **Model Performance**: Best performer is Gradient Boosting Regressor with an R² score of 0.91 and RMSE of 0.18.

## Dataset
- **Source**: CoinGecko snapshot (`coin_gecko_2022-03-17.csv`)
- **Columns**: `coin`, `symbol`, `price`, `1h`, `24h`, `7d`, `24h_volume`, `mkt_cap`, `date`
- **Target**: `target_volume_24h` (synthetic next-day volume for liquidity prediction)

## Project Structure
```
crypto_liquidity_project/
├── data/
│   └── coin_gecko_2022-03-17.csv
├── models/
│   ├── best_crypto_liquidity_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── Streamlit_Preparation.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crypto_liquidity_project.git
   cd crypto_liquidity_project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset (`coin_gecko_2022-03-17.csv`) is placed in the `data/` folder.

## Usage
### Local Execution
1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Access the app at `http://localhost:8501` or the provided ngrok URL for public access.

### Google Colab with ngrok
1. Upload `streamlit_app.py`, `models/`, and `data/` to Colab.
2. Install dependencies:
   ```bash
   !pip install streamlit pyngrok
   ```
3. Run Streamlit with ngrok:
   ```python
   from pyngrok import ngrok
   !streamlit run streamlit_app.py &
   public_url = ngrok.connect(port=8501)
   print("Streamlit App URL:", public_url)
   ```
4. Access the app via the ngrok URL: [https://e62f-34-91-137-46.ngrok-free.app/](https://e62f-34-91-137-46.ngrok-free.app/)

## Pipeline Architecture
1. **Raw CSV Dataset**: CoinGecko snapshot loaded from `data/`.
2. **Data Preprocessing**: Cleans data, handles missing values, and renames columns.
3. **EDA**: Generates visualizations (histograms, heatmaps) and statistical summaries.
4. **Feature Engineering**: Creates features like `price_to_marketcap` and `volatility_score`.
5. **Model Training**: Tests multiple regressors with GridSearchCV, selecting the best based on RMSE and R².
6. **Model Export**: Saves model and scaler as `.pkl` files using `joblib`.
7. **Streamlit App**: Provides EDA dashboard and prediction interface.

## Model Performance
| Model             | R² Score | RMSE  |
|-------------------|----------|-------|
| Linear Regression | 0.72     | 0.39  |
| Random Forest     | 0.88     | 0.21  |
| Gradient Boosting | **0.91** | **0.18** |

## Key Learnings
- Cryptocurrency data is volatile; robust scalers and log transformations improve model stability.
- Feature engineering significantly boosts prediction accuracy.
- Streamlit and ngrok enable rapid deployment for testing and demos.

## Future Enhancements
- Integrate real-time data via CoinGecko or Binance APIs.
- Deploy on Streamlit Cloud, Azure, or Heroku for persistent access.
- Extend to classify liquidity tiers (High, Medium, Low).
- Add logging and monitoring for production use.

## Technologies Used
- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit`
- **Tools**: Google Colab, ngrok, Jupyter Notebooks

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug fixes, feature requests, or improvements.

## License
This project is licensed under the MIT License.

## Author
Pratyush Priyam Kuanr

## Acknowledgments
- CoinGecko for providing the dataset.
- Streamlit and ngrok for enabling easy deployment.