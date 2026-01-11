"""
ENTSO-E API Data Fetching Script
Fetch electricity price data from ENTSO-E Transparency Platform
"""

import sys
import os
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import logging

# Import configuration
from config.api_config import (
    API_TOKEN, API_BASE_URL, DEFAULT_ZONE, get_zone_code,
    get_doc_type, DATA_DIR, TIMEZONE, REQUEST_TIMEOUT,
    MAX_RETRIES, REQUEST_DELAY, validate_api_token,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


class ENTSOEDataFetcher:
    """ENTSO-E Data Fetcher"""

    def __init__(self, api_token=None):
        """
        Initialize data fetcher

        Args:
            api_token: ENTSO-E API token
        """
        self.api_token = api_token or API_TOKEN
        self.base_url = API_BASE_URL
        self.session = requests.Session()

    def _format_datetime(self, dt):
        """
        Format datetime to ENTSO-E required format

        Args:
            dt: datetime object

        Returns:
            Formatted time string (YYYYMMDDHHMM)
        """
        return dt.strftime('%Y%m%d%H%M')

    def _parse_xml_response(self, xml_content):
        """
        Parse ENTSO-E XML response

        Args:
            xml_content: XML string

        Returns:
            DataFrame containing timestamp and price data
        """
        try:
            root = ET.fromstring(xml_content)

            # Define XML namespace
            ns = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3'}

            data_points = []

            # Iterate through all time series
            for timeseries in root.findall('.//ns:TimeSeries', ns):
                # Get time interval
                period = timeseries.find('.//ns:Period', ns)

                if period is None:
                    continue

                # Get start time
                start_time_str = period.find('ns:timeInterval/ns:start', ns).text
                start_time = pd.to_datetime(start_time_str)

                # Get resolution (PT60M = 60 minutes)
                resolution = period.find('ns:resolution', ns).text
                if resolution == 'PT60M':
                    freq = '1H'
                elif resolution == 'PT15M':
                    freq = '15min'
                else:
                    freq = '1H'  # Default

                # Get all data points
                points = period.findall('ns:Point', ns)

                for point in points:
                    position = int(point.find('ns:position', ns).text)
                    price = float(point.find('ns:price.amount', ns).text)

                    # Calculate timestamp
                    timestamp = start_time + pd.Timedelta(hours=position - 1)

                    data_points.append({
                        'timestamp': timestamp,
                        'price_eur_mwh': price
                    })

            # Create DataFrame
            df = pd.DataFrame(data_points)

            if len(df) > 0:
                df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info(f"Successfully parsed {len(df)} data records")
            else:
                logger.warning("No data points found")

            return df

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            logger.debug(f"XML content: {xml_content[:500]}...")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Data parsing error: {e}")
            return pd.DataFrame()

    def fetch_day_ahead_prices(self, zone, start_date, end_date):
        """
        Fetch day-ahead price data

        Args:
            zone: Price zone (e.g., 'SE3')
            start_date: Start date (datetime)
            end_date: End date (datetime)

        Returns:
            DataFrame containing price data
        """
        zone_code = get_zone_code(zone)
        doc_type = get_doc_type('day_ahead_prices')

        # Build request parameters
        params = {
            'securityToken': self.api_token,
            'documentType': doc_type,
            'in_Domain': zone_code,
            'out_Domain': zone_code,
            'periodStart': self._format_datetime(start_date),
            'periodEnd': self._format_datetime(end_date)
        }

        logger.info(f"Fetching {zone} zone data: {start_date.date()} to {end_date.date()}")

        # Send request (with retry mechanism)
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    logger.info(f"API request successful [OK]")
                    return self._parse_xml_response(response.content)

                elif response.status_code == 401:
                    logger.error("Invalid or expired API Token [ERROR]")
                    return pd.DataFrame()

                elif response.status_code == 429:
                    logger.warning(f"Rate limited, waiting {REQUEST_DELAY * 2} seconds before retry...")
                    time.sleep(REQUEST_DELAY * 2)
                    continue

                else:
                    logger.error(f"API request failed: HTTP {response.status_code}")
                    logger.debug(f"Response content: {response.text[:500]}")

                    if attempt < MAX_RETRIES - 1:
                        logger.info(f"Waiting {REQUEST_DELAY} seconds before retry ({attempt + 1}/{MAX_RETRIES})...")
                        time.sleep(REQUEST_DELAY)
                    else:
                        return pd.DataFrame()

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout, retrying ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(REQUEST_DELAY)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY)
                else:
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_multiple_days(self, zone, days=30):
        """
        Fetch multiple days of data (batch requests to avoid timeout)

        Args:
            zone: Price zone
            days: Number of days

        Returns:
            Combined DataFrame
        """
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)

        logger.info(f"Starting to fetch {days} days of data...")

        all_data = []

        # Request 7 days per batch
        batch_days = 7
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=batch_days), end_date)

            df_batch = self.fetch_day_ahead_prices(zone, current_start, current_end)

            if len(df_batch) > 0:
                all_data.append(df_batch)

            current_start = current_end

            # Avoid rate limiting
            if current_start < end_date:
                time.sleep(REQUEST_DELAY)

        # Combine all data
        if all_data:
            df_combined = pd.concat(all_data)
            df_combined = df_combined.sort_index().drop_duplicates()
            logger.info(f"Total fetched {len(df_combined)} data records [OK]")
            return df_combined
        else:
            logger.warning("No data fetched [ERROR]")
            return pd.DataFrame()

    def save_to_csv(self, df, filename=None):
        """
        Save data to CSV file

        Args:
            df: DataFrame
            filename: Filename (auto-generated if None)

        Returns:
            Saved file path
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'entsoe_prices_{timestamp}.csv'

        filepath = DATA_DIR / filename

        df.to_csv(filepath)
        logger.info(f"Data saved to: {filepath} [OK]")

        return filepath


def main():
    """Main function"""
    print("=" * 60)
    print("ENTSO-E Electricity Price Data Fetcher")
    print("=" * 60)

    # Validate API Token
    if not validate_api_token():
        return

    # Create data fetcher
    fetcher = ENTSOEDataFetcher()

    # Set parameters
    zone = DEFAULT_ZONE  # SE3
    days = 30  # Fetch last 30 days

    print(f"\nStarting data fetch...")
    print(f"Zone: {zone}")
    print(f"Days: {days}")
    print(f"Time range: {(datetime.now() - timedelta(days=days)).date()} to {datetime.now().date()}")
    print("-" * 60)

    # Fetch data
    df = fetcher.fetch_multiple_days(zone, days=days)

    if len(df) > 0:
        # Display data summary
        print("\n" + "=" * 60)
        print("Data fetch successful!")
        print("=" * 60)
        print(f"Records: {len(df)}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: {df['price_eur_mwh'].min():.2f} - {df['price_eur_mwh'].max():.2f} EUR/MWh")
        print(f"Average price: {df['price_eur_mwh'].mean():.2f} EUR/MWh")

        print("\nFirst 5 records:")
        print(df.head())

        # Save data
        print("\n" + "-" * 60)
        filename = f'entsoe_{zone.lower()}_latest.csv'
        filepath = fetcher.save_to_csv(df, filename)

        print("=" * 60)
        print("Complete! [OK]")
        print("=" * 60)

    else:
        print("\n" + "=" * 60)
        print("Data fetch failed [ERROR]")
        print("=" * 60)
        print("Please check:")
        print("1. API Token is correct")
        print("2. Network connection is working")
        print("3. Date range is valid")
        print("=" * 60)


if __name__ == '__main__':
    main()
