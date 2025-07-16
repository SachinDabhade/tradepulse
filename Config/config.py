import os
import configparser
from Utilities.utilities import display_market_status

# Config file path
config = configparser.ConfigParser()

# Save the current working directory in the config under a section
config['PATHS'] = {}
config['VARIABLES'] = {}
config['DATABASE'] = {}
config['MARKET'] = {}

config['PATHS']['HOME_DIR'] = os.getcwd()

config['PATHS']['CONFIG_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Config')
config['PATHS']['DATA_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Data')
config['PATHS']['EXECUTION_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Execution')
config['PATHS']['RISK_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Risk')  
config['PATHS']['STRATEGIES_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Strategies')
config['PATHS']['UTILITIES_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Utilities')
config['PATHS']['PAGES_DIR'] = os.path.join(config['PATHS']['HOME_DIR'], 'Pages')

config['PATHS']['HISTORICAL_DIR'] = os.path.join(config['PATHS']['DATA_DIR'], 'HISTORICAL')
config['PATHS']['INDEXES_DIR'] = os.path.join(config['PATHS']['DATA_DIR'], 'INDEXES')
config['PATHS']['LOG_DIR'] = os.path.join(config['PATHS']['DATA_DIR'], 'LOGS')
config['PATHS']['REALTIME_DIR'] = os.path.join(config['PATHS']['DATA_DIR'], 'REALTIME')
config['PATHS']['STATIC'] = os.path.join(config['PATHS']['CONFIG_DIR'], 'STATIC')

config['PATHS']['STOCK_DATABASE'] = os.path.join(config['PATHS']['HISTORICAL_DIR'], 'stock_data.db')
config['PATHS']['INDEX_LINKS'] = os.path.join(config['PATHS']['STATIC'], 'nifty_index_links.csv')
config['PATHS']['INDEX_JSON'] = os.path.join(config['PATHS']['STATIC'], 'nifty_indices_json.json')
config['PATHS']['TODAY_OPENING_JSON'] = os.path.join(config['PATHS']['REALTIME_DIR'], 'today_opening_prices.json')
config['PATHS']['DHAN_SECRETS'] = os.path.join(config['PATHS']['STATIC'], 'dhan_secrets.ini')
config.read(config['PATHS']['DHAN_SECRETS'])


# IMPORTANT VARIABLES
config['VARIABLES']['INTERVALS'] = "1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo"
# config['VARIABLES']['INTERVALS'] = "1d,5d,1wk,1mo,3mo"

config['DATABASE']['1m'] = 'one_minute_stock_prices'
config['DATABASE']['2m'] = 'two_minute_stock_prices'
config['DATABASE']['5m'] = 'five_minute_stock_prices'
config['DATABASE']['15m'] = 'fifteen_minute_stock_prices'
config['DATABASE']['30m'] = 'thirty_minute_stock_prices'
config['DATABASE']['60m'] = 'sixty_minute_stock_prices'
config['DATABASE']['90m'] = 'ninty_minute_stock_prices'
config['DATABASE']['1h'] = 'hourly_stock_prices'
config['DATABASE']['1d'] = 'daily_stock_prices'
config['DATABASE']['5d'] = 'five_day_stock_prices'
config['DATABASE']['1wk'] = 'weekly_stock_prices'
config['DATABASE']['1mo'] = 'monthly_stock_prices'
config['DATABASE']['3mo'] = 'quarterly_stock_prices'

# USAGE
"""
from config import config
print(config.get('PATHS', 'HOME_DIR', fallback=None))
print(config['PATHS']['HOME_DIR'])

"""

# Images file path
config['PATHS']['QUANTLOGO'] = os.path.join(config['PATHS']['STATIC'], 'Quant Logo.png')



config['MARKET']['STATUS'] = str(display_market_status())  # display_market_status() returns a bool
