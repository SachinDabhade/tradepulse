from dhanhq import DhanContext, dhanhq
from Config.config import config

class DhanExecutionEngine:
    # --- Dhan API ENUMS & CONSTANTS ---
    # Exchange Segments
    IDX_I = 0  # Index Value
    NSE_EQ = 1  # NSE Equity Cash
    NSE_FNO = 2  # NSE Futures & Options
    NSE_CURRENCY = 3  # NSE Currency
    BSE_EQ = 4  # BSE Equity Cash
    MCX_COMM = 5  # MCX Commodity
    BSE_CURRENCY = 7  # BSE Currency
    BSE_FNO = 8  # BSE Futures & Options

    # Product Types
    CNC = 'CNC'  # Cash & Carry
    INTRADAY = 'INTRADAY'  # Intraday
    MARGIN = 'MARGIN'  # Carry Forward
    CO = 'CO'  # Cover Order
    BO = 'BO'  # Bracket Order

    # Order Status
    STATUS_TRANSIT = 'TRANSIT'
    STATUS_PENDING = 'PENDING'
    STATUS_CLOSED = 'CLOSED'
    STATUS_TRIGGERED = 'TRIGGERED'
    STATUS_REJECTED = 'REJECTED'
    STATUS_CANCELLED = 'CANCELLED'
    STATUS_PART_TRADED = 'PART_TRADED'
    STATUS_TRADED = 'TRADED'

    # AMO Times
    AMO_PRE_OPEN = 'PRE_OPEN'
    AMO_OPEN = 'OPEN'
    AMO_OPEN_30 = 'OPEN_30'
    AMO_OPEN_60 = 'OPEN_60'

    # Expiry Codes
    EXPIRY_CURRENT = 0
    EXPIRY_NEXT = 1
    EXPIRY_FAR = 2

    # Instrument Types
    INSTR_INDEX = 'INDEX'
    INSTR_FUTIDX = 'FUTIDX'
    INSTR_OPTIDX = 'OPTIDX'
    INSTR_EQUITY = 'EQUITY'
    INSTR_FUTSTK = 'FUTSTK'
    INSTR_OPTSTK = 'OPTSTK'
    INSTR_FUTCOM = 'FUTCOM'
    INSTR_OPTFUT = 'OPTFUT'
    INSTR_FUTCUR = 'FUTCUR'
    INSTR_OPTCUR = 'OPTCUR'

    # Feed Request Codes
    FEED_CONNECT = 11
    FEED_DISCONNECT = 12
    FEED_SUBSCRIBE_TICKER = 15
    FEED_UNSUBSCRIBE_TICKER = 16
    FEED_SUBSCRIBE_QUOTE = 17
    FEED_UNSUBSCRIBE_QUOTE = 18
    FEED_SUBSCRIBE_FULL = 21
    FEED_UNSUBSCRIBE_FULL = 22
    FEED_SUBSCRIBE_DEPTH = 23
    FEED_UNSUBSCRIBE_DEPTH = 24

    # Feed Response Codes
    FEED_INDEX_PACKET = 1
    FEED_TICKER_PACKET = 2
    FEED_QUOTE_PACKET = 4
    FEED_OI_PACKET = 5
    FEED_PREV_CLOSE_PACKET = 6
    FEED_MARKET_STATUS_PACKET = 7
    FEED_FULL_PACKET = 8
    FEED_DISCONNECT_PACKET = 50

    # Error Codes (partial, for reference)
    ERROR_CODES = {
        'DH-901': 'Invalid Authentication: Client ID or access token is invalid or expired.',
        'DH-902': 'Invalid Access: Not subscribed to Data APIs or no access to Trading APIs.',
        'DH-903': "User Account Error: Check segment activation or requirements.",
        'DH-904': 'Rate Limit: Too many requests. Throttle API calls.',
        'DH-905': 'Input Exception: Missing/bad fields.',
        'DH-906': 'Order Error: Incorrect order request.',
        'DH-907': 'Data Error: Incorrect parameters or no data.',
        'DH-908': 'Internal Server Error.',
        'DH-909': 'Network Error.',
        'DH-910': 'Other Error.'
    }
    DATA_ERROR_CODES = {
        800: 'Internal Server Error',
        804: 'Requested number of instruments exceeds limit',
        805: 'Too many requests/connections',
        806: 'Data APIs not subscribed',
        807: 'Access token expired',
        808: 'Authentication Failed',
        809: 'Access token invalid',
        810: 'Client ID invalid',
        811: 'Invalid Expiry Date',
        812: 'Invalid Date Format',
        813: 'Invalid SecurityId',
        814: 'Invalid Request',
    }

    def __init__(self):
        self.dhan_context = DhanContext(config['DHAN']['CLIENT_ID'], config['DHAN']['API_SECRET_TOKEN'])
        self.dhan = dhanhq(dhan_context=self.dhan_context)
        
    # --- Order Management ---
    def place_order(self, **kwargs):
        """Place an order. Use class constants for enums."""
        return self.dhan.place_order(**kwargs)

    def modify_order(self, order_id, **kwargs):
        return self.dhan.modify_order(order_id, **kwargs)

    def cancel_order(self, order_id):
        return self.dhan.cancel_order(order_id)

    def get_order_list(self):    # done
        return self.dhan.get_order_list()

    def get_order_by_id(self, order_id):
        return self.dhan.get_order_by_id(order_id)

    def get_order_by_correlationID(self, correlationID):
        return self.dhan.get_order_by_correlationID(correlationID)

    # --- Portfolio/Account ---
    def get_positions(self):           # done
        return self.dhan.get_positions()

    def get_holdings(self):            # done
        return self.dhan.get_holdings()

    def get_fund_limits(self):         # done
        return self.dhan.get_fund_limits()

    # --- Trade/Order Book ---
    def get_trade_book(self, order_id=None):
        return self.dhan.get_trade_book(order_id)

    def get_trade_history(self, from_date, to_date, page_number=0):
        return self.dhan.get_trade_history(from_date, to_date, page_number)

    # --- Instrument/Market Data ---
    def fetch_security_list(self, mode="compact"):
        return self.dhan.fetch_security_list(mode)

    def expiry_list(self, under_security_id, under_exchange_segment):
        return self.dhan.expiry_list(under_security_id, under_exchange_segment)

    def option_chain(self, under_security_id, under_exchange_segment, expiry):
        return self.dhan.option_chain(under_security_id, under_exchange_segment, expiry)

    def ohlc_data(self, securities):
        return self.dhan.ohlc_data(securities=securities)

    def historical_daily_data(self, security_id, exchange_segment, instrument_type, from_date, to_date):
        return self.dhan.historical_daily_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date
        )

    def intraday_minute_data(self, security_id, exchange_segment, instrument_type, from_date, to_date):
        return self.dhan.intraday_minute_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date
        )

    # --- Forever Orders ---
    def place_forever(self, **kwargs):
        return self.dhan.place_forever(**kwargs)

    # --- eDIS/TPIN ---
    def generate_tpin(self):
        return self.dhan.generate_tpin()

    def open_browser_for_tpin(self, **kwargs):
        return self.dhan.open_browser_for_tpin(**kwargs)

    def edis_inquiry(self, isin=None):
        return self.dhan.edis_inquiry(isin=isin)

    # --- Utility ---
    def convert_to_date_time(self, epoch_date):
        return self.dhan.convert_to_date_time(epoch_date)

    @staticmethod
    def check_status(response):
        """
        Checks the status of a Dhan API response dict.
        Returns (True, data) if status is 'success', else (False, remarks or error message).
        """
        if isinstance(response, dict):
            if response.get('status') == 'success':
                return True, response.get('data')
            else:
                return False, response.get('remarks', response.get('errorMessage', 'Unknown error'))
        return False, 'Invalid response format'

    # --- Error Handling Helper ---
    @classmethod
    def interpret_error(cls, code):
        """Return a user-friendly error message for a given error code."""
        if code in cls.ERROR_CODES:
            return cls.ERROR_CODES[code]
        if code in cls.DATA_ERROR_CODES:
            return cls.DATA_ERROR_CODES[code]
        return f"Unknown error code: {code}"