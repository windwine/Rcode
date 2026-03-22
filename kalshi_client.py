"""
Kalshi API client — wraps all HTTP communication with the public trade API.
No authentication required for public market data.
"""

import time
import requests
from typing import Optional


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_DELAY = 0.2   # seconds between requests to avoid rate limiting
TIMEOUT = 30          # request timeout in seconds
MAX_RETRIES = 3       # max retries for network errors


class KalshiClient:
    """Kalshi API client for public market data endpoints."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "kalshi-finance-fetcher/1.0",
        })

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Send a GET request with retry logic.
        - 429 rate limit: wait 5s, retry up to 10 times
        - Timeout / connection error: exponential backoff, up to 3 retries
        """
        url = f"{BASE_URL}{endpoint}"
        rate_limit_retries = 0
        max_rate_limit_retries = 10
        connect_attempt = 0

        while True:
            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT)
                if resp.status_code == 429:
                    rate_limit_retries += 1
                    if rate_limit_retries > max_rate_limit_retries:
                        print(f"  [rate limit] exceeded max retries for: {endpoint}")
                        return {}
                    wait = 5
                    print(f"  [rate limit] attempt {rate_limit_retries}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                connect_attempt += 1
                print(f"  [timeout] attempt {connect_attempt}/{MAX_RETRIES}: {url}")
                if connect_attempt >= MAX_RETRIES:
                    raise
                time.sleep(2 ** connect_attempt)
            except requests.exceptions.ConnectionError as e:
                connect_attempt += 1
                print(f"  [connection error] attempt {connect_attempt}/{MAX_RETRIES}: {e}")
                if connect_attempt >= MAX_RETRIES:
                    raise
                time.sleep(2 ** connect_attempt)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500:
                    print(f"  [server error] {e}")
                raise

    def get_all_events(self, status: str = "open") -> list:
        """
        Paginate through all events (cursor-based, 200 per page).
        status: "open" | "closed" | "settled"
        """
        events = []
        cursor = None
        page = 1

        while True:
            params = {"limit": 200, "status": status}
            if cursor:
                params["cursor"] = cursor

            data = self._get("/events", params=params)
            page_events = data.get("events", [])
            events.extend(page_events)
            print(f"  page {page}: fetched {len(page_events)} (total {len(events)})")

            cursor = data.get("cursor")
            if not cursor or not page_events:
                break

            page += 1
            time.sleep(REQUEST_DELAY)

        return events

    def get_markets_for_event(self, event_ticker: str) -> list:
        """
        Fetch all markets under a given event (paginated).
        """
        markets = []
        cursor = None

        while True:
            params = {"event_ticker": event_ticker, "limit": 100}
            if cursor:
                params["cursor"] = cursor

            data = self._get("/markets", params=params)
            page_markets = data.get("markets", [])
            markets.extend(page_markets)

            cursor = data.get("cursor")
            if not cursor or not page_markets:
                break

            time.sleep(REQUEST_DELAY)

        return markets

    def get_candles_batch(self, tickers: list, start_ts: int, end_ts: int,
                          period_interval: int) -> dict:
        """
        Batch candlestick fetch for active markets.
        Max 100 tickers per request, max 10,000 candles total returned.
        Returns: {ticker: [candle, ...]}
        """
        params = {
            "market_tickers": ",".join(tickers),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        data = self._get("/markets/candlesticks", params=params)
        result = {}
        for item in data.get("markets", []):
            result[item["market_ticker"]] = item.get("candlesticks", [])
        return result

    def get_candles_historical(self, ticker: str, start_ts: int, end_ts: int,
                               period_interval: int) -> list:
        """
        Fetch candlestick data for a settled (historical) market.
        Returns empty list if no data (404).
        """
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        try:
            data = self._get(f"/historical/markets/{ticker}/candlesticks", params=params)
            return data.get("candlesticks", [])
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return []
            raise

    def get_event(self, event_ticker: str) -> dict:
        """
        Fetch a single event with its nested markets.
        Returns the raw API response dict.
        """
        return self._get(f"/events/{event_ticker}")
