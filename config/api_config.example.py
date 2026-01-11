"""
ENTSO-E API Configuration
Configuration for ENTSO-E Transparency Platform API access
"""

import os
from pathlib import Path

# =====================================================
# API Configuration
# =====================================================

# ENTSO-E API Token
# Set ENTSOE_API_TOKEN in environment variable, or fill in directly here
# How to get: https://transparency.entsoe.eu/myAccount/webApiAccess
API_TOKEN = os.environ.get('ENTSOE_API_TOKEN', 'YOUR_API_TOKEN_HERE')

# ENTSO-E API Base URL
API_BASE_URL = 'https://web-api.tp.entsoe.eu/api'

# =====================================================
# Region Configuration (Bidding Zones)
# =====================================================

# Swedish electricity price zones (ENTSO-E EIC codes)
BIDDING_ZONES = {
    'SE1': '10Y1001A1001A44P',  # Northern Sweden - Luleå
    'SE2': '10Y1001A1001A45N',  # Central-North Sweden - Sundsvall
    'SE3': '10Y1001A1001A46L',  # Central-South Sweden - Stockholm
    'SE4': '10Y1001A1001A47J',  # Southern Sweden - Malmö
}

# Default zone
DEFAULT_ZONE = 'SE3'

# =====================================================
# Data Type Configuration
# =====================================================

# ENTSO-E Document Types
DOCUMENT_TYPES = {
    'day_ahead_prices': 'A44',  # Day-ahead prices
    'actual_load': 'A65',        # Actual total load
    'forecasted_load': 'A65',    # Day-ahead total load forecast
    'generation': 'A75',         # Actual generation per type
}

# Default document type
DEFAULT_DOC_TYPE = 'day_ahead_prices'

# =====================================================
# Time Configuration
# =====================================================

# Default number of days to fetch
DEFAULT_DAYS = 30

# Timezone setting
TIMEZONE = 'Europe/Stockholm'

# =====================================================
# File Path Configuration
# =====================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data storage directory
DATA_DIR = PROJECT_ROOT / 'data' / 'api_data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Feature files directory
FEATURES_DIR = PROJECT_ROOT / 'features'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Model files directory
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Data Processing Configuration
# =====================================================

# Resample frequency (original data is 15min, resample to 1 hour)
RESAMPLE_FREQ = '1H'

# Missing value handling method
MISSING_VALUE_METHOD = 'interpolate'  # 'interpolate', 'ffill', 'bfill', 'drop'

# =====================================================
# API Request Configuration
# =====================================================

# Request timeout (seconds)
REQUEST_TIMEOUT = 30

# Maximum retry attempts
MAX_RETRIES = 3

# Request delay (seconds) - avoid rate limiting
REQUEST_DELAY = 1

# =====================================================
# Logging Configuration
# =====================================================

# Log level
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# Log format
LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# =====================================================
# Helper Functions
# =====================================================

def get_zone_code(zone_name):
    """
    Get EIC code for a zone

    Args:
        zone_name: Zone name (e.g., 'SE3')

    Returns:
        EIC code string
    """
    return BIDDING_ZONES.get(zone_name.upper(), BIDDING_ZONES[DEFAULT_ZONE])


def get_doc_type(type_name):
    """
    Get document type code

    Args:
        type_name: Document type name

    Returns:
        Document type code
    """
    return DOCUMENT_TYPES.get(type_name, DOCUMENT_TYPES[DEFAULT_DOC_TYPE])


def validate_api_token():
    """
    Validate if API Token is configured

    Returns:
        bool: Whether token is valid
    """
    if API_TOKEN == 'YOUR_API_TOKEN_HERE' or not API_TOKEN:
        print("ERROR: Please configure ENTSO-E API Token!")
        print("Method 1: Modify API_TOKEN in config/api_config.py")
        print("Method 2: Set environment variable ENTSOE_API_TOKEN")
        return False
    return True


if __name__ == '__main__':
    print("=== ENTSO-E API Configuration Info ===")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Default Zone: {DEFAULT_ZONE} ({get_zone_code(DEFAULT_ZONE)})")
    print(f"Default Doc Type: {DEFAULT_DOC_TYPE} ({get_doc_type(DEFAULT_DOC_TYPE)})")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Features Directory: {FEATURES_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"\nAPI Token Status: {'Configured [OK]' if validate_api_token() else 'Not Configured [ERROR]'}")
