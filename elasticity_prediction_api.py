"""
Flask API that matches your existing ABL script endpoints
Endpoints: /product/random and /predict/elasticity
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)
CORS(app)

# Load models and data
try:
    model = pickle.load(open('elasticity_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    price_stats = pickle.load(open('price_stats.pkl', 'rb'))
    print("✓ Models loaded")
except:
    model = None
    scaler = None
    features = ['LogPrice', 'PriceBucket', 'PriceVsMedian', 'PriceZScore', 'Month',
                'Quarter', 'DayOfWeek', 'Hour', 'IsWeekend', 'IsEndOfMonth', 'IsQ4',
                'IsSummer', 'IsLowPrice', 'IsMidPrice', 'IsHighPrice']
    price_stats = {'mean_price': 3.47, 'median_price': 2.95, 'std_price': 4.2}

# Load data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
unknown_df = pd.read_csv('unknown_data.csv')

@app.route('/')
def home():
    """Simple dashboard"""
    return render_template_string(SIMPLE_HTML)

@app.route('/product/random', methods=['GET'])
def get_random_product():
    """Get random product - matches ABL script endpoint"""
    
    # Clean data
    valid_df = unknown_df.dropna(subset=['CustomerID', 'UnitPrice']).copy()
    valid_df = valid_df[valid_df['UnitPrice'] > 0]
    valid_df = valid_df[valid_df['UnitPrice'] < 50]
    
    if valid_df.empty:
        return jsonify({"error": "No valid products"}), 500
    
    product = valid_df.sample(n=1).iloc[0]
    
    return jsonify({
        'stock_code': str(product['StockCode']),
        'description': str(product.get('Description', 'N/A'))[:100],
        'current_price': round(float(product['UnitPrice']), 2),
        'country': str(product.get('Country', 'N/A')),
        'quantity': int(product['Quantity']),
        'customer_id': int(product['CustomerID'])
    })

@app.route('/predict/elasticity', methods=['POST'])
def predict_elasticity():
    """Predict elasticity - matches ABL script endpoint"""
    
    try:
        # Parse JSON - handle various formats
        raw_data = request.get_data(as_text=True)
        print(f"Received: {raw_data[:100]}")
        
        # Try parsing
        try:
            data = json.loads(raw_data)
        except:
            # If parsing fails, try cleaning
            if raw_data.strip():
                # Remove potential quotes
                cleaned = raw_data.strip()
                if cleaned[0] in ['"', "'"] and cleaned[-1] in ['"', "'"]:
                    cleaned = cleaned[1:-1]
                try:
                    data = json.loads(cleaned)
                except:
                    # Last resort - use empty dict
                    data = {}
            else:
                data = {}
        
        # Get product details
        stock_code = data.get('stock_code', 'UNKNOWN')
        current_price = float(data.get('current_price', 10))
        quantity = int(data.get('quantity', 10))
        
        # Test 30% price increase (as mentioned in ABL)
        test_price = current_price * 1.3
        
        # Simple elasticity calculation
        if model and scaler:
            # Create features for current and test price
            features_current = create_features(current_price)
            features_test = create_features(test_price)
            
            # Predict quantities
            X_current = pd.DataFrame([features_current])[features]
            X_test = pd.DataFrame([features_test])[features]
            
            qty_current = np.expm1(model.predict(scaler.transform(X_current))[0])
            qty_test = np.expm1(model.predict(scaler.transform(X_test))[0])
            
            # Calculate elasticity
            price_change = (test_price - current_price) / current_price
            qty_change = (qty_test - qty_current) / qty_current if qty_current > 0 else -0.5
            elasticity = qty_change / price_change if price_change != 0 else -1
        else:
            # Fallback simple model
            elasticity = -1.2  # Assume elastic
            qty_change = elasticity * 0.3  # 30% price increase
            qty_test = quantity * (1 + qty_change)
        
        # Determine if elastic or inelastic
        elasticity_type = "ELASTIC" if abs(elasticity) > 1 else "INELASTIC"
        
        # Calculate revenue impact
        revenue_current = current_price * quantity
        revenue_test = test_price * qty_test
        revenue_change = revenue_test - revenue_current
        
        result = {
            'elasticity': round(elasticity, 2),
            'elasticity_type': elasticity_type,
            'current_price': current_price,
            'test_price': round(test_price, 2),
            'current_quantity': quantity,
            'predicted_quantity': round(qty_test, 2),
            'revenue_change': round(revenue_change, 2),
            'recommendation': f"Product is {elasticity_type} - " + 
                            ("REJECT price increase" if elasticity_type == "ELASTIC" 
                             else "APPROVE price increase")
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_features(price):
    """Create feature vector for a price"""
    return {
        'LogPrice': np.log1p(price),
        'PriceBucket': min(3, int(price / 3)),
        'PriceVsMedian': price / price_stats.get('median_price', 2.95),
        'PriceZScore': (price - price_stats.get('mean_price', 3.47)) / price_stats.get('std_price', 4.2),
        'Month': 10,
        'Quarter': 4,
        'DayOfWeek': 3,
        'Hour': 14,
        'IsWeekend': 0,
        'IsEndOfMonth': 0,
        'IsQ4': 1,
        'IsSummer': 0,
        'IsLowPrice': int(price < 1),
        'IsMidPrice': int(1 <= price <= 5),
        'IsHighPrice': int(price > 5)
    }

SIMPLE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>ML Price Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            background: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            font-size: 24px;
            margin: 0 0 15px 0;
            color: #333;
        }
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }
        .stats {
            display: flex;
            gap: 20px;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 10px;
            color: #666;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background: #5569d8;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .panel h3 {
            font-size: 14px;
            margin: 0 0 10px 0;
            color: #555;
        }
        .panel p {
            margin: 5px 0;
            font-size: 13px;
        }
        .slider {
            width: 100%;
            height: 4px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
            margin: 15px 0;
        }
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        .price-display {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
        }
        #impact-panel {
            transition: all 0.3s ease;
            border-width: 2px;
        }
        #impact-panel.profit {
            background-color: #d4edda !important;
            border-color: #28a745 !important;
        }
        #impact-panel.loss {
            background-color: #f8d7da !important;
            border-color: #dc3545 !important;
        }
        #impact-panel.neutral {
            background-color: #fff3cd !important;
            border-color: #ffc107 !important;
        }
        .chart-container {
            height: 200px;
            margin-top: 10px;
        }
        canvas {
            max-width: 100%;
            height: 180px !important;
        }
        .elastic {
            color: #28a745;
            font-weight: bold;
        }
        .inelastic {
            color: #dc3545;
            font-weight: bold;
        }
        .impact-value {
            font-size: 16px;
            font-weight: bold;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="top-bar">
            <h1>ML Price Optimization</h1>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">379K</div>
                    <div class="stat-label">TRAIN</div>
                </div>
                <div class="stat">
                    <div class="stat-value">108K</div>
                    <div class="stat-label">TEST</div>
                </div>
                <div class="stat">
                    <div class="stat-value">0.291</div>
                    <div class="stat-label">R²</div>
                </div>
            </div>
            <button onclick="testProduct()">Get Random Product</button>
        </div>
        
        <div class="main-content" id="main-content" style="display: none;">
            <!-- Product Info Panel -->
            <div class="panel">
                <h3>Product Info</h3>
                <p><strong>Code:</strong> <span id="stock-code">-</span></p>
                <p><strong>Desc:</strong> <span id="description" style="font-size: 11px;">-</span></p>
                <p><strong>Current Price:</strong> £<span id="current-price">-</span></p>
                <p><strong>Quantity:</strong> <span id="quantity">-</span></p>
                <p><strong>Elasticity:</strong> <span id="elasticity">-</span></p>
                <p>Type: <span id="elasticity-type">-</span></p>
            </div>
            
            <!-- Price Test Panel -->
            <div class="panel">
                <h3>Price Simulator</h3>
                <div class="price-display">
                    <span>Test: £<strong id="test-price">-</strong></span>
                    <span><strong id="price-change">0</strong>%</span>
                </div>
                <input type="range" class="slider" id="price-slider" 
                       min="50" max="150" value="100" oninput="updatePrice()">
                <div class="price-display">
                    <small>-50%</small>
                    <small>Current</small>
                    <small>+50%</small>
                </div>
                <div class="chart-container">
                    <canvas id="revenue-chart"></canvas>
                </div>
            </div>
            
            <!-- Impact Panel -->
            <div class="panel neutral" id="impact-panel">
                <h3>Revenue Impact</h3>
                <p>Current Rev: £<span id="current-rev">-</span></p>
                <p>Predicted Rev: £<span id="pred-rev">-</span></p>
                <p class="impact-value">Impact: <span id="impact">-</span></p>
                <p style="margin-top: 10px;"><strong>Decision:</strong></p>
                <p id="recommendation" style="font-size: 14px;">Move slider to test</p>
            </div>
        </div>
    </div>
    
    <script>
        let currentProduct = null;
        let currentPrediction = null;
        let chart = null;
        
        async function testProduct() {
            try {
                // Get random product
                const productRes = await fetch('/product/random');
                currentProduct = await productRes.json();
                
                // Test elasticity
                const elasticityRes = await fetch('/predict/elasticity', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(currentProduct)
                });
                currentPrediction = await elasticityRes.json();
                
                // Show main content
                document.getElementById('main-content').style.display = 'grid';
                
                // Update product info
                document.getElementById('stock-code').textContent = currentProduct.stock_code;
                document.getElementById('description').textContent = currentProduct.description.substring(0, 50);
                document.getElementById('current-price').textContent = currentProduct.current_price.toFixed(2);
                document.getElementById('quantity').textContent = currentProduct.quantity;
                document.getElementById('elasticity').textContent = currentPrediction.elasticity.toFixed(2);
                
                const typeElement = document.getElementById('elasticity-type');
                typeElement.textContent = currentPrediction.elasticity_type;
                typeElement.className = currentPrediction.elasticity_type.toLowerCase();
                
                document.getElementById('current-rev').textContent = 
                    (currentProduct.current_price * currentProduct.quantity).toFixed(2);
                
                // Reset slider
                document.getElementById('price-slider').value = 100;
                document.getElementById('test-price').textContent = currentProduct.current_price.toFixed(2);
                
                // Draw chart
                drawChart();
                
                // Update display
                updatePrice();
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function updatePrice() {
            if (!currentProduct || !currentPrediction) return;
            
            const sliderValue = parseFloat(document.getElementById('price-slider').value);
            const testPrice = currentProduct.current_price * (sliderValue / 100);
            const priceChangePct = sliderValue - 100;
            
            // Update display
            document.getElementById('test-price').textContent = testPrice.toFixed(2);
            document.getElementById('price-change').textContent = 
                priceChangePct > 0 ? '+' + priceChangePct : priceChangePct;
            
            // Calculate impact
            const elasticity = currentPrediction.elasticity;
            const qtyChangePct = elasticity * (priceChangePct / 100);
            const newQty = Math.max(0, currentProduct.quantity * (1 + qtyChangePct));
            const currentRevenue = currentProduct.current_price * currentProduct.quantity;
            const newRevenue = testPrice * newQty;
            const revenueImpact = newRevenue - currentRevenue;
            const impactPct = (revenueImpact / currentRevenue * 100).toFixed(1);
            
            // Update revenue display
            document.getElementById('pred-rev').textContent = newRevenue.toFixed(2);
            
            const impactElement = document.getElementById('impact');
            impactElement.textContent = 
                (revenueImpact >= 0 ? '+£' : '-£') + Math.abs(revenueImpact).toFixed(2) + 
                ' (' + (impactPct > 0 ? '+' : '') + impactPct + '%)';
            impactElement.className = revenueImpact >= 0 ? 'positive' : 'negative';
            
            // Update panel color based on ANY profit/loss
            const panel = document.getElementById('impact-panel');
            panel.className = 'panel';
            
            if (revenueImpact > 0.01) {  // Any profit above 1 penny
                panel.classList.add('profit');
                document.getElementById('recommendation').textContent = '✅ Increase revenue';
                
                if (chart && chart.data.datasets[1]) {
                    chart.data.datasets[1].backgroundColor = '#28a745';
                }
            } else if (revenueImpact < -0.01) {  // Any loss below 1 penny
                panel.classList.add('loss');
                document.getElementById('recommendation').textContent = '❌ Reduces revenue';
                
                if (chart && chart.data.datasets[1]) {
                    chart.data.datasets[1].backgroundColor = '#dc3545';
                }
            } else {  // Only exactly zero or tiny amounts
                panel.classList.add('neutral');
                document.getElementById('recommendation').textContent = '➖ No change';
                
                if (chart && chart.data.datasets[1]) {
                    chart.data.datasets[1].backgroundColor = '#ffc107';
                }
            }
            
            // Update chart marker
            if (chart) {
                chart.data.datasets[1].data = [{ x: testPrice, y: newRevenue }];
                chart.update('none');
            }
        }
        
        function drawChart() {
            const ctx = document.getElementById('revenue-chart').getContext('2d');
            
            if (chart) chart.destroy();
            
            // Generate curve
            const prices = [];
            const revenues = [];
            
            for (let pct = 50; pct <= 150; pct += 10) {
                const price = currentProduct.current_price * (pct / 100);
                const priceChange = (pct - 100) / 100;
                const qtyChange = currentPrediction.elasticity * priceChange;
                const qty = Math.max(0, currentProduct.quantity * (1 + qtyChange));
                const revenue = price * qty;
                
                prices.push(price);
                revenues.push(revenue);
            }
            
            // Find max revenue for better scaling
            const maxRevenue = Math.max(...revenues);
            const minRevenue = Math.min(...revenues);
            
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: prices.map(p => p.toFixed(1)),
                    datasets: [
                        {
                            label: 'Revenue',
                            data: revenues,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            fill: true,
                            pointRadius: 3,
                            pointHoverRadius: 5
                        },
                        {
                            label: 'Current',
                            data: [{ 
                                x: currentProduct.current_price, 
                                y: currentProduct.current_price * currentProduct.quantity 
                            }],
                            backgroundColor: '#28a745',
                            borderColor: '#28a745',
                            pointRadius: 8,
                            pointHoverRadius: 10,
                            type: 'scatter',
                            showLine: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    if (context.dataset.label === 'Revenue') {
                                        return '£' + context.parsed.y.toFixed(2);
                                    }
                                    return context.dataset.label + ': £' + context.parsed.y.toFixed(2);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            grid: {
                                display: false
                            },
                            ticks: {
                                display: true,
                                font: {
                                    size: 10
                                },
                                callback: function(value) {
                                    return '£' + value.toFixed(1);
                                }
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            },
                            ticks: {
                                display: true,
                                font: {
                                    size: 10
                                },
                                callback: function(value) {
                                    return '£' + value.toFixed(0);
                                }
                            },
                            suggestedMin: minRevenue * 0.9,
                            suggestedMax: maxRevenue * 1.1
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════╗
    ║  ML API - Compatible with Your ABL Script         ║
    ╚════════════════════════════════════════════════════╝
    
    Starting on port 5001...
    
    Endpoints (matching your ABL):
    - GET  /product/random     - Get random product
    - POST /predict/elasticity - Test elasticity
    
    Dashboard: http://localhost:5001/
    
    Your ABL script should work with these endpoints!
    """)
    
    app.run(host='127.0.0.1', port=5001, debug=True)