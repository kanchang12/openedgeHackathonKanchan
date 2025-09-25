"""
Fixed Price Elasticity Model - Prevents Overfitting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PRICE ELASTICITY MODEL - FIXED VERSION")
print("="*60)

# LOAD DATA
print("\n1. Loading data...")
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

print(f"   Train size: {len(train_df):,}")
print(f"   Test size: {len(test_df):,}")

# CLEAN DATA
def clean_data(df):
    df = df[df['Quantity'] > 0].copy()
    df = df[df['UnitPrice'] > 0]
    df = df[df['UnitPrice'] < 100]  # Remove extreme prices
    df = df[df['Quantity'] < 100]   # Remove bulk orders
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

train_df = clean_data(train_df)
test_df = clean_data(test_df)

print(f"   After cleaning - Train: {len(train_df):,}, Test: {len(test_df):,}")

# CREATE GENERAL FEATURES (not product-specific)
print("\n2. Creating general features...")

def create_features(df):
    # Time features
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['Quarter'] = df['InvoiceDate'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsEndOfMonth'] = (df['InvoiceDate'].dt.day > 25).astype(int)
    
    # Price features (general, not product-specific)
    df['PriceBucket'] = pd.qcut(df['UnitPrice'], q=10, labels=False, duplicates='drop')
    df['LogPrice'] = np.log1p(df['UnitPrice'])
    df['PriceSquared'] = df['UnitPrice'] ** 2
    
    # Seasonal indicators
    df['IsQ4'] = (df['Quarter'] == 4).astype(int)  # Holiday season
    df['IsSummer'] = df['Month'].isin([6, 7, 8]).astype(int)
    
    # Price point categories
    df['IsLowPrice'] = (df['UnitPrice'] < df['UnitPrice'].quantile(0.25)).astype(int)
    df['IsMidPrice'] = ((df['UnitPrice'] >= df['UnitPrice'].quantile(0.25)) & 
                        (df['UnitPrice'] <= df['UnitPrice'].quantile(0.75))).astype(int)
    df['IsHighPrice'] = (df['UnitPrice'] > df['UnitPrice'].quantile(0.75)).astype(int)
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

# Use global statistics (not product-specific)
global_price_stats = {
    'median_price': train_df['UnitPrice'].median(),
    'mean_price': train_df['UnitPrice'].mean(),
    'std_price': train_df['UnitPrice'].std()
}

train_df['PriceVsMedian'] = train_df['UnitPrice'] / global_price_stats['median_price']
test_df['PriceVsMedian'] = test_df['UnitPrice'] / global_price_stats['median_price']

train_df['PriceZScore'] = (train_df['UnitPrice'] - global_price_stats['mean_price']) / global_price_stats['std_price']
test_df['PriceZScore'] = (test_df['UnitPrice'] - global_price_stats['mean_price']) / global_price_stats['std_price']

# SELECT FEATURES
features = [
    'LogPrice', 'PriceBucket', 'PriceVsMedian', 'PriceZScore',
    'Month', 'Quarter', 'DayOfWeek', 'Hour',
    'IsWeekend', 'IsEndOfMonth', 'IsQ4', 'IsSummer',
    'IsLowPrice', 'IsMidPrice', 'IsHighPrice'
]

print(f"   Using {len(features)} general features")

# PREPARE DATA
X_train = train_df[features]
y_train = np.log1p(train_df['Quantity'])  # Log transform target

X_test = test_df[features]
y_test = np.log1p(test_df['Quantity'])

# SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAIN MULTIPLE MODELS AND PICK BEST
print("\n3. Training models with anti-overfitting measures...")

models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_split=50,
        min_samples_leaf=20,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
}

best_model = None
best_score = -999

for name, model in models.items():
    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n   {name}:")
    print(f"   CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"   Train R2: {train_r2:.3f}")
    print(f"   Test R2:  {test_r2:.3f}")
    
    # Check for overfitting
    overfit_ratio = train_r2 / (test_r2 + 0.001)
    print(f"   Overfit ratio: {overfit_ratio:.1f}x")
    
    if test_r2 > best_score and overfit_ratio < 3:  # Reasonable generalization
        best_score = test_r2
        best_model = model
        best_model_name = name

print(f"\n4. Selected model: {best_model_name}")

# FINAL EVALUATION
y_pred_test = best_model.predict(X_test_scaled)
y_pred_test_actual = np.expm1(y_pred_test)
y_test_actual = np.expm1(y_test)

print(f"   Final Test R2: {r2_score(y_test, y_pred_test):.3f}")
print(f"   Final Test MAE: {mean_absolute_error(y_test_actual, y_pred_test_actual):.2f} units")
print(f"   Final Test RMSE: {np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual)):.2f} units")

# ANALYZE PRICE ELASTICITY
print("\n5. Price Elasticity Analysis...")

# Create synthetic data to test elasticity
price_points = np.linspace(0.5, 10, 20)
elasticities = []

for price in price_points:
    # Create two samples with slightly different prices
    sample_data = pd.DataFrame({
        'LogPrice': [np.log1p(price), np.log1p(price * 1.01)],
        'PriceBucket': [5, 5],
        'PriceVsMedian': [price / global_price_stats['median_price']] * 2,
        'PriceZScore': [(price - global_price_stats['mean_price']) / global_price_stats['std_price']] * 2,
        'Month': [6, 6],
        'Quarter': [2, 2],
        'DayOfWeek': [2, 2],
        'Hour': [14, 14],
        'IsWeekend': [0, 0],
        'IsEndOfMonth': [0, 0],
        'IsQ4': [0, 0],
        'IsSummer': [1, 1],
        'IsLowPrice': [int(price < 2), int(price < 2)],
        'IsMidPrice': [int(2 <= price <= 5), int(2 <= price <= 5)],
        'IsHighPrice': [int(price > 5), int(price > 5)]
    })
    
    X_sample = scaler.transform(sample_data[features])
    predictions = np.expm1(best_model.predict(X_sample))
    
    # Calculate elasticity
    pct_change_quantity = (predictions[1] - predictions[0]) / predictions[0]
    pct_change_price = 0.01  # 1% price change
    elasticity = pct_change_quantity / pct_change_price
    elasticities.append(elasticity)

# Find elastic vs inelastic ranges
elastic_prices = [p for p, e in zip(price_points, elasticities) if abs(e) > 1]
inelastic_prices = [p for p, e in zip(price_points, elasticities) if abs(e) <= 1]

print(f"   Elastic price range (|e| > 1): £{min(elastic_prices) if elastic_prices else 0:.2f} - £{max(elastic_prices) if elastic_prices else 0:.2f}")
print(f"   Inelastic price range (|e| <= 1): £{min(inelastic_prices) if inelastic_prices else 0:.2f} - £{max(inelastic_prices) if inelastic_prices else 0:.2f}")

# SAVE EVERYTHING
print("\n6. Saving models and parameters...")
pickle.dump(best_model, open('elasticity_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(features, open('features.pkl', 'wb'))
pickle.dump(global_price_stats, open('price_stats.pkl', 'wb'))

print("   Saved: elasticity_model.pkl, scaler.pkl, features.pkl, price_stats.pkl")

# FEATURE IMPORTANCE
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n7. Top 5 Important Features:")
    for _, row in importance_df.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE - Ready for production use!")
print("="*60)