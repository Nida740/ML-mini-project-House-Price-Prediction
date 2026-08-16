# House Price Prediction with Fair Deal Flag

A machine learning mini project that predicts house prices using Multiple Linear Regression, and adds a "Fair Deal / Overpriced" flag by comparing each house's actual listed price to the model's predicted price. Includes an interactive Gradio interface where users can enter house details and instantly see a prediction and deal verdict.

## Project Overview

- **Goal:** Predict house price from features (size, bedrooms) and flag whether a listed price is a fair deal, overpriced, or a good deal.
- **Model:** Multiple Linear Regression (scikit-learn)
- **Interface:** Gradio (live, interactive web form)

## Dataset

`house_data.csv` — 60 house records with:
- `Size_sqft` — size of the house in square feet
- `Bedrooms` — number of bedrooms
- `Price` — sale price (target variable)

## How It Works

1. Load and visualize the dataset
2. Split into training (80%) and test (20%) sets
3. Train a Linear Regression model on `Size_sqft` and `Bedrooms`
4. Evaluate using Mean Absolute Error (MAE) and R² score
5. Compare each test house's actual price to its predicted price
6. Flag it as **Overpriced**, **Fair Price**, or **Good Deal** based on a ±10% threshold
7. Wrap the trained model in a Gradio interface for live predictions

## Results

- **R² Score:** 0.98
- **Mean Absolute Error:** ~16,734

## How to Run

1. Open `ML-miniproj_house-price-prediction.ipynb` in Google Colab
2. Upload `house_data.csv` to the Colab session (or update the file path in the code)
3. Run all cells top to bottom
4. The last cell launches a Gradio interface — enter house details to get a live prediction and deal flag

## Author

**Nida Syed**
 B.TECH |3rd year|AI & DS
from: Anjuman College of Engineering and Technology
