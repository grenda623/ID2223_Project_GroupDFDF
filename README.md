# Electricity Price Predictor - API Version

Real-time electricity price prediction system based on ENTSO-E Transparency Platform API.

## Project Overview

This is an API-based electricity price prediction system that fetches real-time data from the ENTSO-E official API, eliminating the need for manually downloaded CSV files.

**Key Features**:
- Real-time data fetching from ENTSO-E API
- Automated data processing and feature engineering
- Machine learning model training (XGBoost, LightGBM)
- Interactive web application (Streamlit)
- Support for SE3 region (Stockholm, Sweden)
- 24-72 hour price forecasting
- Automatic daily updates at 08:00 Stockholm time

---

## Project Structure

```
api_version/
│
├── Configuration Files
│   ├── config/
│   │   ├── __init__.py              # Python package initialization
│   │   └── api_config.py            # API configuration (Token, regions, paths, etc.)
│   │
│   ├── requirements.txt             # Python dependencies
│   └── .gitignore                   # Git ignore file (optional)
│
├── Data Files
│   └── data/
│       └── api_data/                # Raw data from ENTSO-E API
│           └── entsoe_se3_latest.csv
│
├── Core Scripts
│   ├── scripts/
│   │   └── fetch_data.py            # ENTSO-E API data fetching script
│   │
│   ├── features/
│   │   ├── create_features.py       # Feature engineering script
│   │   └── features_latest.csv      # Generated feature file
│   │
│   └── models/
│       ├── train_models.py          # Model training script
│       ├── xgboost_latest.pkl       # XGBoost model
│       ├── lightgbm_latest.pkl      # LightGBM model
│       ├── xgboost_metrics_latest.json
│       ├── lightgbm_metrics_latest.json
│       └── model_comparison_latest.csv
│
├── Web Application
│   └── app_api.py                   # Streamlit interactive application
│
├── Documentation
│   ├── docs/
│   │   └── AUTO_UPDATE_GUIDE.md     # Automatic update guide
│   └── README.md                    # This file
│
└── Run Scripts
    ├── run_pipeline.bat             # Windows one-click run script
    ├── run_pipeline.sh              # macOS/Linux one-click run script
    ├── setup_daily_update.bat       # Windows auto-update setup
    └── auto_update.bat              # Automatic update script
```

---

## Quick Start Guide

### Step 1: Get ENTSO-E API Token

1. Open your browser and visit: https://transparency.entsoe.eu/
2. Click **Login / Register** in the top right corner
3. Register a new account (or log in with an existing account)
4. After logging in, click your username in the top right
5. Select **My Account** > **Web API Access**
6. Click the green **Generate Token** button
7. Copy the generated token (a long string of characters)


### Step 2: Configure API Token

**Option A: Environment Variable**

Windows PowerShell:
```powershell
$env:ENTSOE_API_TOKEN = "your_token_here"
```

Windows CMD:
```cmd
set ENTSOE_API_TOKEN=your_token_here
```

macOS/Linux:
```bash
export ENTSOE_API_TOKEN="your_token_here"
```

**Option B: Direct Configuration**

Edit `config/api_config.py`:
```python
API_TOKEN = 'your_token_here'  # Replace with your token
```

### Step 3: Install Dependencies

```bash
# Using Anaconda Python
D:\anaconda\python.exe -m pip install -r requirements.txt

# Or using standard Python
pip install -r requirements.txt
```

### Step 4: Run the Complete Pipeline

#### Method A: One-Click Run (Recommended)

**Windows**:
```batch
run_pipeline.bat
```

**macOS/Linux**:
```bash
chmod +x run_pipeline.sh  # First time only
./run_pipeline.sh
```

#### Method B: Step-by-Step Execution

```bash
# Step 1: Fetch API data (1-2 minutes)
D:\anaconda\python.exe scripts/fetch_data.py

# Step 2: Create features (10-30 seconds)
D:\anaconda\python.exe features/create_features.py

# Step 3: Train models (1-2 minutes)
D:\anaconda\python.exe models/train_models.py

# Step 4: Launch web application
D:\anaconda\python.exe -m streamlit run app_api.py
```

---

## Pipeline Workflow

The one-click script automatically completes these 4 steps:

### 1 Fetch API Data
- Retrieves last 30 days of SE3 region electricity price data from ENTSO-E
- Saves to `data/api_data/entsoe_se3_latest.csv`
- Handles XML parsing and error retry

### 2 Create Features
- Generates 62 machine learning features from raw price data
- Saves to `features/features_latest.csv`

**Feature Categories** (Total: 62 features):
1. **Lagged Features** (8): price_lag_1h ~ price_lag_72h
2. **Rolling Statistics** (24):
   - mean, std, min, max
   - Windows: 6h, 12h, 24h, 48h, 72h, 168h
3. **Calendar Features** (16):
   - hour, day_of_week, month, quarter
   - Cyclical encoding (sin/cos)
   - is_weekend, is_weekday
4. **Price Changes** (5):
   - price_diff_1h, price_diff_2h, price_diff_24h
   - price_pct_change_1h, price_pct_change_24h
5. **Statistical Features** (3):
   - price_vs_hour_avg
   - price_vs_dow_avg
   - price_deviation_from_7d_mean
6. **Interaction Features** (2):
   - price_hour_interaction
   - weekend_hour_interaction

### 3 Train Models
- Trains XGBoost and LightGBM models
- Saves models to `models/xgboost_latest.pkl` and `models/lightgbm_latest.pkl`
- Displays model performance comparison

**Models Trained**:

**Baseline Models** (3):
- **Persistence**: Uses price from 24h ago
- **Historical Average**: Average price by hour
- **Rolling Average**: 24-hour rolling average

**Machine Learning Models** (2):
- **XGBoost**: Gradient boosted trees
- **LightGBM**: Light gradient boosted machines

**Expected Performance**:
- R² > 0.99
- MAE < 3 EUR/MWh
- RMSE < 5 EUR/MWh

### 4 Launch Web Application
- Starts Streamlit web app
- Automatically opens browser at `http://localhost:8501`
- View 24-hour electricity price forecasts

---

## Using the Web Application

After successful launch, you will see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Main Features:

1. **Real-time Predictions**: View price forecasts for the next 24 hours
2. **Price Trends**: 7 days historical + 24 hours forecast trend chart
3. **Daily Patterns**: Price distribution across different hours of the day
4. **Usage Recommendations**:
   - Best usage time (lowest price - ideal for washing machine, dryer, EV charging)
   - Avoid usage time (highest price - avoid high-energy appliances)
   - Low price period statistics
   - Price volatility analysis
5. **Detailed Data**: View hourly prediction prices in table format

### Sidebar Options:

- **Select Model**: Choose between XGBoost or LightGBM
- **Prediction Horizon**: Adjust from 6 to 72 hours
- **Refresh Data**: Re-fetch latest API data

---

## Automatic Daily Updates

### Setup Automatic Updates (Windows)

Run the setup script as Administrator to enable daily automatic updates at 08:00 Stockholm time:

```batch
cd D:\KTH\ID2223\Project\api_version
setup_daily_update.bat
```

Or right-click `setup_daily_update.bat` and select **"Run as Administrator"**.

### Verify Task Created

```batch
schtasks /query /tn "ElectricityPriceAutoUpdate"
```

### What It Does

Every day at 08:00 Stockholm time, the system automatically:

1. Fetches latest data from ENTSO-E API (`scripts/fetch_data.py`)
2. Creates features (`features/create_features.py`)
3. Trains models (`models/train_models.py`)
4. Saves logs to `logs/auto_update_YYYYMMDD_HHMMSS.log`

### Managing the Scheduled Task

**View task status**:
```batch
schtasks /query /tn "ElectricityPriceAutoUpdate"
```

**View task details**:
```batch
schtasks /query /tn "ElectricityPriceAutoUpdate" /fo LIST /v
```

**Run task manually (test)**:
```batch
schtasks /run /tn "ElectricityPriceAutoUpdate"
```

**Delete task**:
```batch
schtasks /delete /tn "ElectricityPriceAutoUpdate" /f
```

**Change schedule time**:

Edit `setup_daily_update.bat` and modify this line:
```batch
set START_TIME=08:00  # Change to your desired time (24-hour format)
```

Then run the setup script again.

### Checking Logs

Logs are saved in the `logs/` directory with timestamps:

```
logs/auto_update_YYYYMMDD_HHMMSS.log
```

Example log content:
```
=====================================================
Automatic Update Started
Time: 2026-01-11 08:00:00
=====================================================

[08:00:05] Step 1/3: Fetching latest data from ENTSO-E API...
[08:00:35] SUCCESS: Data fetched

[08:00:35] Step 2/3: Creating features...
[08:00:45] SUCCESS: Features created

[08:00:45] Step 3/3: Training models...
[08:02:15] SUCCESS: Models trained

=====================================================
Automatic Update Completed Successfully
Time: 2026-01-11 08:02:15
=====================================================
```

### macOS/Linux Alternative (Cron)

**Edit crontab**:
```bash
crontab -e
```

**Add cron job** (08:00 Stockholm time):
```bash
0 8 * * * cd /path/to/api_version && bash auto_update.sh >> logs/auto_update.log 2>&1
```

**Create `auto_update.sh`**:
```bash
#!/bin/bash
# Auto-update script for macOS/Linux

cd "$(dirname "$0")"

echo "====================================================="
echo "Automatic Update Started: $(date)"
echo "====================================================="

# Step 1: Fetch data
echo "Step 1/3: Fetching data..."
python3 scripts/fetch_data.py

# Step 2: Create features
echo "Step 2/3: Creating features..."
python3 features/create_features.py

# Step 3: Train models
echo "Step 3/3: Training models..."
python3 models/train_models.py

echo "====================================================="
echo "Automatic Update Completed: $(date)"
echo "====================================================="
```

Make executable:
```bash
chmod +x auto_update.sh
```

---

## Configuration

### Change Prediction Region

Edit `config/api_config.py`:

```python
# Change default zone
DEFAULT_ZONE = 'SE3'  # Options: SE1, SE2, SE3, SE4

# Swedish electricity price zones:
# SE1: Northern Sweden - Luleå
# SE2: Central-North Sweden - Sundsvall
# SE3: Central-South Sweden - Stockholm
# SE4: Southern Sweden - Malmö
```

### Adjust Data Fetch Days

Edit `config/api_config.py`:

```python
# Change default fetch days
DEFAULT_DAYS = 30  # Fetch last 30 days
# Can set to any number of days
```

### Modify Model Parameters

Edit the parameter dictionaries in `models/train_models.py`.

---

## Troubleshooting

### Q1: API Token Invalid

**Symptom**: `API request failed: HTTP 401`

**Solution**:
1. Check if token is correctly copied (no extra spaces)
2. Verify token status shows `ACTIVE` on the website
3. Try regenerating the token
4. Check token at: https://transparency.entsoe.eu/myAccount/webApiAccess

### Q2: No pip Command

**Symptom**: `'pip' is not recognized as an internal or external command`

**Solution**:
```batch
# Windows - use full path
D:\anaconda\python.exe -m pip install -r requirements.txt

# macOS/Linux - use pip3
pip3 install -r requirements.txt
```

### Q3: Port 8501 Already in Use

**Symptom**: `Port 8501 is in use by another program`

**Solution**:
```batch
# Specify a different port
streamlit run app_api.py --server.port 8502
```

### Q4: Data Fetching Is Slow

**Reason**: API server may be overseas, causing network latency

**Solution**:
- Be patient, typically completes in 1-2 minutes
- Reduce fetch days (modify `DEFAULT_DAYS` in `config/api_config.py`)

### Q5: Model Prediction Inaccurate

**Reason**: May need more training data

**Solution**:
1. Increase fetch days: modify `DEFAULT_DAYS = 60`
2. Re-run the complete pipeline
3. Model will improve with more data

### Q6: Request Timeout

**Error**: `Request timeout`

**Solution**:
- Check network connection
- Reduce fetch days (modify `DEFAULT_DAYS`)
- Increase timeout (modify `REQUEST_TIMEOUT` in config)

### Q7: XML Parsing Error

**Error**: `XML parsing error`

**Solution**:
- Check if requested date range is valid
- Verify region code is correct
- Review API error messages in logs

### Q8: Feature File Not Found

**Error**: `Feature file not found`

**Solution**:
```bash
# Run feature engineering script first
D:\anaconda\python.exe features/create_features.py
```

### Q9: Model File Not Found

**Error**: `Model file not found`

**Solution**:
```bash
# Run model training script first
D:\anaconda\python.exe models/train_models.py
```

### Q10: Task Not Running Automatically

**Check if task exists**:
```bash
schtasks /query /tn "ElectricityPriceAutoUpdate"
```

**Recreate task**:
```bash
setup_daily_update.bat
```

### Q11: API Token Issues in Auto-Update

If logs show "API Token invalid":

1. Check `config/api_config.py`
2. Verify API_TOKEN is correct
3. Check token status at: https://transparency.entsoe.eu/myAccount/webApiAccess

### Q12: Python Path Issues in Auto-Update

If logs show "Python not found":

1. Edit `auto_update.bat`
2. Update the Python path:
   ```batch
   set PYTHON=D:\anaconda\python.exe
   ```

---

## Performance Optimization

### Data Caching

Streamlit app uses 1-hour TTL caching:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
```

### API Request Optimization

- Batch requests (7 days per batch) to avoid timeout
- 1-second delay between requests to avoid rate limiting
- Maximum 3 retry attempts with exponential backoff

### Model Optimization

Run hyperparameter tuning scripts in `scripts/analysis/` (optional).

---

## Data Format

### Raw Data (entsoe_se3_latest.csv)
```csv
timestamp,price_eur_mwh
2024-01-01 00:00:00,45.23
2024-01-01 01:00:00,42.15
...
```

### Feature Data (features_latest.csv)
```csv
timestamp,price_eur_mwh,price_lag_1h,price_lag_24h,...,hour_sin,hour_cos
2024-01-01 00:00:00,45.23,44.50,43.20,...,0.0,1.0
...
```

### Model Metrics (xgboost_metrics_latest.json)
```json
{
    "mae": 2.23,
    "rmse": 3.45,
    "r2": 0.9913,
    "mape": 5.67
}
```

---

## API Documentation

ENTSO-E API Official Documentation:
- [API Guide](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
- [API Docs](https://transparency.entsoe.eu/api-guide)

---

## Comparison with Manual Version

| Feature | Manual Version | API Version |
|---------|---------------|-------------|
| Data Source | Manual CSV download | ENTSO-E API auto-fetch |
| Data Update | Manual download | One-click auto-update |
| Real-time | Depends on download frequency | Can fetch latest data anytime |
| Automation | Requires manual operation | Supports scheduled auto-run |
| Data Range | Limited by downloaded file | Flexible date range specification |
| Dependencies | No external dependencies | Requires API Token |

---

## Security Notes

1. **Don't commit API Token to Git**
   - Use environment variables
   - Add `config/api_config.py` to `.gitignore`

2. **Protect API Token**
   - Don't share with others
   - Rotate token periodically

3. **Backup Important Files**
   - Trained models (`*.pkl`)
   - Feature files (`features_latest.csv`)

---

## Dependencies

- **pandas**: Data processing
- **numpy**: Numerical computation
- **scikit-learn**: Machine learning tools
- **xgboost**: XGBoost model
- **lightgbm**: LightGBM model
- **requests**: HTTP requests
- **streamlit**: Web application framework
- **plotly**: Interactive charts

Install all:
```bash
pip install -r requirements.txt
```

---

## Next Steps

After completing the quick start, you can:

1. **Adjust Parameters**: Modify configurations in `config/api_config.py`
2. **Optimize Models**: Run hyperparameter tuning scripts
3. **Analyze Features**: Examine feature importance
4. **Switch Regions**: Change `DEFAULT_ZONE` to SE1, SE2, or SE4
5. **Customize App**: Modify `app_api.py` to add new features
6. **Set Up Auto-Update**: Enable daily automatic updates

---

## View Results

After successful execution, find generated files at:

```
api_version/
├── data/api_data/
│   └── entsoe_se3_latest.csv        # Raw API data
├── features/
│   └── features_latest.csv          # Feature data
├── models/
│   ├── xgboost_latest.pkl           # XGBoost model
│   ├── lightgbm_latest.pkl          # LightGBM model
│   ├── xgboost_metrics_latest.json  # XGBoost performance
│   ├── lightgbm_metrics_latest.json # LightGBM performance
│   └── model_comparison_latest.csv  # Model comparison
└── logs/
    └── auto_update_*.log            # Auto-update logs
```

---

## Technical Support

Having issues?

1. Review this documentation
2. Check error messages in log output
3. Verify all dependencies are correctly installed
4. Refer to ENTSO-E official documentation: https://transparency.entsoe.eu/api-guide
5. Check auto-update logs in `logs/` directory
