# Automatic Daily Update Guide

## Overview

This guide shows you how to set up automatic daily updates for the Electricity Price Predictor at 08:00 Stockholm time.

## Quick Setup (Windows)

### Step 1: Run Setup Script

Right-click `setup_daily_update.bat` and select **"Run as Administrator"**

Or in PowerShell/CMD (as Administrator):
```bash
cd D:\KTH\ID2223\Project\api_version
setup_daily_update.bat
```

### Step 2: Verify Task Created

```bash
schtasks /query /tn "ElectricityPriceAutoUpdate" /fo LIST /v
```

### Done! ✅

The system will now automatically:
1. Fetch latest data from ENTSO-E API at 08:00 Stockholm time
2. Create features
3. Train models

## How It Works

### Scheduled Task Details

- **Task Name**: `ElectricityPriceAutoUpdate`
- **Schedule**: Daily at 08:00 (Stockholm local time)
- **Script**: `auto_update.bat`
- **Logs**: Saved to `logs/auto_update_YYYYMMDD_HHMMSS.log`

### Time Zone Handling

The task uses your **local system time**. Since you're in Stockholm:
- Windows automatically handles CET (UTC+1) / CEST (UTC+2)
- The task runs at 08:00 local time, regardless of daylight saving changes
- No manual adjustment needed!

### What Gets Updated

Every day at 08:00, the system:

1. **Fetches Latest Data** (`scripts/fetch_data.py`)
   - Gets last 30 days of SE3 electricity prices
   - Includes today's day-ahead prices
   - Saves to `data/api_data/entsoe_se3_latest.csv`

2. **Creates Features** (`features/create_features.py`)
   - Generates 62 machine learning features
   - Saves to `features/features_latest.csv`

3. **Trains Models** (`models/train_models.py`)
   - Trains XGBoost and LightGBM models
   - Updates `models/xgboost_latest.pkl` and `models/lightgbm_latest.pkl`

## Managing the Task

### View Task Status

```bash
schtasks /query /tn "ElectricityPriceAutoUpdate"
```

### View Task Details

```bash
schtasks /query /tn "ElectricityPriceAutoUpdate" /fo LIST /v
```

### Run Task Manually (Test)

```bash
schtasks /run /tn "ElectricityPriceAutoUpdate"
```

### Delete Task

```bash
schtasks /delete /tn "ElectricityPriceAutoUpdate" /f
```

Or run:
```bash
setup_daily_update.bat
```
and it will recreate the task.

### Change Schedule Time

To change from 08:00 to another time, edit `setup_daily_update.bat`:

```batch
REM Change this line:
set START_TIME=08:00

REM To your desired time (24-hour format):
set START_TIME=06:00  # For 06:00 AM
set START_TIME=20:00  # For 08:00 PM
```

Then run the setup script again.

## Checking Logs

### View Latest Log

```bash
cd logs
dir /o-d
# Open the most recent log file
```

### Log File Format

```
auto_update_YYYYMMDD_HHMMSS.log
```

Example:
```
auto_update_20260111_080000.log
```

### Log Content Example

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

## Troubleshooting

### Task Not Running

**Check if task exists:**
```bash
schtasks /query /tn "ElectricityPriceAutoUpdate"
```

**Recreate task:**
```bash
setup_daily_update.bat
```

### API Token Issues

If you see "API Token invalid" in logs:

1. Check `config/api_config.py`
2. Verify API_TOKEN is correct
3. Check token status at: https://transparency.entsoe.eu/myAccount/webApiAccess

### Python Path Issues

If logs show "Python not found":

1. Edit `auto_update.bat`
2. Update the Python path:
   ```batch
   set PYTHON=D:\anaconda\python.exe
   ```

### Permission Issues

The scheduled task needs to run even when you're not logged in:

1. Open Task Scheduler (`taskschd.msc`)
2. Find "ElectricityPriceAutoUpdate"
3. Right-click → Properties
4. Check "Run whether user is logged on or not"
5. Enter your Windows password when prompted

## macOS/Linux Alternative

For macOS/Linux, use cron instead:

### Edit Crontab

```bash
crontab -e
```

### Add Cron Job (08:00 Stockholm time)

```bash
# Run at 08:00 Stockholm time (adjust for timezone)
0 8 * * * cd /path/to/api_version && bash auto_update.sh >> logs/auto_update.log 2>&1
```

### Create `auto_update.sh`

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

## Monitoring

### Email Notifications (Optional)

To receive email notifications when updates complete/fail, you can:

1. Add email sending logic to `auto_update.bat`
2. Use Windows Task Scheduler email settings (requires SMTP server)
3. Use third-party monitoring tools

### Dashboard Integration

The Streamlit app automatically shows the latest data:
- Check "Last Update" time in sidebar
- Should show today's date after 08:00 AM

## Best Practices

1. **Check logs weekly** to ensure updates are running
2. **Monitor API usage** to stay within ENTSO-E limits
3. **Keep API token secure** - don't commit to Git
4. **Backup models** before major changes
5. **Test task manually** after setup to verify it works

## FAQ

**Q: Will it run if my computer is off?**
A: No, the computer must be on at 08:00. Consider:
   - Using a server/cloud VM
   - Keeping your computer on
   - Deploying to a cloud platform

**Q: What if the update fails?**
A: Check the log file in `logs/` folder for error details. The task will try again the next day.

**Q: Can I run updates more frequently?**
A: Yes! Edit the scheduled task frequency:
```bash
# Example: Run every 6 hours
schtasks /create /tn "ElectricityPriceAutoUpdate" /tr "D:\KTH\ID2223\Project\api_version\auto_update.bat" /sc hourly /mo 6 /st 08:00 /f
```

**Q: Does this work with daylight saving time?**
A: Yes! Windows automatically adjusts for CET/CEST transitions.

---

**Created**: 2026-01-11
**Last Updated**: 2026-01-11
