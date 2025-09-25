# ML Price Optimization for OpenEdge

## About the Project

This project combines economic price elasticity theory with machine learning to predict optimal product pricing. The system integrates OpenEdge ABL with a Python Flask API that uses trained ML models to analyze pricing strategies and their revenue impact.

### Economics and ML Combination

The project applies price elasticity of demand concepts through machine learning. By training on historical transaction data, the model learns how quantity demanded responds to price changes for different products. This allows prediction of revenue-maximizing price points.

### Data Source

Dataset: Online Retail Dataset from Kaggle
- 541,909 transactions from UK-based online retailer
- Period: 01/12/2010 to 09/12/2011

### Data Processing Steps

1. Clean raw data 
2. Feature engineering (price buckets, temporal features, elasticity indicators)
3. Split data into three parts:
   - train_data.csv: 70% for model training (379,336 samples)
   - test_data.csv: 20% for model validation (108,436 samples)  
   - unknown_data.csv: 10% mimics live business data for real-time predictions

## Installation

### Prerequisites
- Progress OpenEdge 11.7 or higher
- Python 3.8 or higher
- Git

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/kanchang12/openedgeHackathonKanchan.git
cd openedgeHackathonKanchan
```

2. Install Python dependencies:
```bash
pip install flask flask-cors pandas numpy scikit-learn
```

3. Verify data files are present:
- train_data.csv
- test_data.csv
- unknown_data.csv
- All model files (.pkl)

## How to Run

1. Start the Python ML API:
```bash
python elasticity_prediction_api.py
```
The API will start on http://localhost:5001

2. Open a new terminal and run the OpenEdge ABL script:
```bash
prowin32 -p ml-integration.p
```

3. Access the dashboard at http://localhost:5001 in your browser

## What to Expect

When you run the system:

1. The ABL script will fetch a random product from the API
2. It sends the product data for ML prediction
3. The model returns optimal price and elasticity analysis
4. ABL processes the response and makes a business recommendation
5. Results are logged to ml_session.log
6. Dashboard shows interactive price simulation with real-time profit/loss visualization

## Business Problem Solved

Retailers often set prices based on intuition or simple cost-plus markup, leaving money on the table. This system solves the pricing optimization problem by:

- Identifying which products are price elastic vs inelastic
- Finding the revenue-maximizing price point for each product
- Quantifying the revenue impact of price changes before implementation
- Providing data-driven pricing recommendations

## Why This is Important

1. Revenue Maximization: Even small pricing improvements compound to significant revenue gains
2. Risk Reduction: Test price changes virtually before implementing in market
3. Speed: Analyze thousands of products instantly vs manual analysis
4. Objectivity: Remove emotional bias from pricing decisions
5. Competitive Advantage: Dynamic pricing based on actual demand patterns

## Practical Use Cases

1. Seasonal Pricing: Adjust prices based on historical seasonal patterns
2. Clearance Optimization: Find the optimal discount level to clear inventory
3. New Product Pricing: Use similar products' elasticity to price new items
4. Promotion Planning: Predict the revenue impact of sales campaigns
5. Category Management: Identify which product categories can sustain price increases

## Technical Summary

The system uses a Random Forest Regressor trained on engineered features including price buckets, temporal patterns, and historical price statistics. The model achieves an R² score of 0.291 on test data. The OpenEdge integration demonstrates how legacy ERP systems can leverage modern ML capabilities through API integration, enabling data-driven decision making without replacing core business systems.
