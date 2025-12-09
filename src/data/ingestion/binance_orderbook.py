"""
Binance Order Book WebSocket collector.
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import aiohttp
import polars as pl
import websockets
from loguru import logger

from src.data.ingestion.base import DataSource


class OrderBook:
    """Local order book state."""

    def __init__(self, symbol: str, depth: int = 100):
        self.symbol = symbol
        self.depth = depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.timestamp: datetime | None = None

    def update(self, data: dict) -> None:
        """Update order book with depth update."""
        for bid in data.get("b", []):
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for ask in data.get("a", []):
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = data.get("u", self.last_update_id)
        self.timestamp = datetime.now(timezone.utc)

    def snapshot(self, n_levels: int = 10) -> dict:
        """Get snapshot of top N levels."""
        sorted_bids = sorted(self.bids.items(), reverse=True)[:n_levels]
        sorted_asks = sorted(self.asks.items())[:n_levels]

        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bids": sorted_bids,
            "asks": sorted_asks,
            "best_bid": sorted_bids[0] if sorted_bids else (0, 0),
            "best_ask": sorted_asks[0] if sorted_asks else (0, 0),
            "mid_price": (
                (sorted_bids[0][0] + sorted_asks[0][0]) / 2
                if sorted_bids and sorted_asks
                else 0
            ),
            "spread": (
                sorted_asks[0][0] - sorted_bids[0][0]
                if sorted_bids and sorted_asks
                else 0
            ),
        }


class BinanceOrderBookCollector(DataSource):
    """Collector for Binance Futures Order Book data via WebSocket."""

    WS_URL = "wss://fstream.binance.com/stream"
    REST_URL = "https://fapi.binance.com/fapi/v1/depth"

    def __init__(self, symbols: list[str], depth: int = 100, testnet: bool = False):
        super().__init__("binance_orderbook")
        self.symbols = symbols
        self.depth = depth
        self.testnet = testnet

        if testnet:
            self.ws_url = "wss://stream.binancefuture.com/stream"
            self.rest_url = "https://testnet.binancefuture.com/fapi/v1/depth"
        else:
            self.ws_url = self.WS_URL
            self.rest_url = self.REST_URL

        self.order_books: dict[str, OrderBook] = {
            symbol: OrderBook(symbol, depth) for symbol in symbols
        }
        self.snapshots: list[dict] = []
        self._running = False

    async def fetch(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch historical order book snapshots.

        For real-time streaming, use start_streaming() instead.
        """
        if symbol:
            return await self._fetch_depth_snapshot(symbol)
        else:
            dfs = []
            for sym in self.symbols:
                df = await self._fetch_depth_snapshot(sym)
                dfs.append(df)
            return pl.concat(dfs) if dfs else pl.DataFrame()

    async def _fetch_depth_snapshot(self, symbol: str) -> pl.DataFrame:
        """Fetch current order book snapshot via REST."""
        async with aiohttp.ClientSession() as session:
            params = {"symbol": symbol, "limit": self.depth}
            async with session.get(self.rest_url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

        # Initialize local order book
        ob = self.order_books.get(symbol, OrderBook(symbol, self.depth))
        ob.bids = {float(b[0]): float(b[1]) for b in data["bids"]}
        ob.asks = {float(a[0]): float(a[1]) for a in data["asks"]}
        ob.last_update_id = data["lastUpdateId"]
        ob.timestamp = datetime.now(timezone.utc)

        return self._snapshot_to_df(ob.snapshot(self.depth))

    def _snapshot_to_df(self, snapshot: dict) -> pl.DataFrame:
        """Convert order book snapshot to DataFrame."""
        rows = []

        for i, (price, qty) in enumerate(snapshot["bids"]):
            rows.append({
                "timestamp": snapshot["timestamp"],
                "symbol": snapshot["symbol"],
                "side": "bid",
                "level": i,
                "price": price,
                "quantity": qty,
            })

        for i, (price, qty) in enumerate(snapshot["asks"]):
            rows.append({
                "timestamp": snapshot["timestamp"],
                "symbol": snapshot["symbol"],
                "side": "ask",
                "level": i,
                "price": price,
                "quantity": qty,
            })

        return pl.DataFrame(rows)

    async def start_streaming(
        self,
        callback: callable | None = None,
        snapshot_interval: int = 60,
    ) -> None:
        """
        Start streaming order book updates.

        Args:
            callback: Optional callback for each update
            snapshot_interval: Seconds between saved snapshots
        """
        self._running = True

        # Initialize with REST snapshots
        for symbol in self.symbols:
            await self._fetch_depth_snapshot(symbol)
            await asyncio.sleep(0.1)

        # Build stream URL
        streams = "/".join(
            [f"{s.lower()}@depth@100ms" for s in self.symbols]
        )
        ws_url = f"{self.ws_url}?streams={streams}"

        self.logger.info(f"Connecting to WebSocket: {ws_url}")

        last_snapshot_time = datetime.now(timezone.utc)

        try:
            async with websockets.connect(ws_url) as ws:
                while self._running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)

                        if "data" in data:
                            stream_data = data["data"]
                            symbol = stream_data["s"]

                            if symbol in self.order_books:
                                self.order_books[symbol].update(stream_data)

                                if callback:
                                    await callback(
                                        symbol,
                                        self.order_books[symbol].snapshot(10),
                                    )

                        # Save periodic snapshots
                        now = datetime.now(timezone.utc)
                        if (now - last_snapshot_time).seconds >= snapshot_interval:
                            for symbol, ob in self.order_books.items():
                                self.snapshots.append(ob.snapshot(self.depth))
                            last_snapshot_time = now

                    except asyncio.TimeoutError:
                        self.logger.debug("WebSocket timeout, sending ping")
                        await ws.ping()

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            raise

    def stop_streaming(self) -> None:
        """Stop streaming."""
        self._running = False

    def get_snapshots_df(self) -> pl.DataFrame:
        """Get all saved snapshots as DataFrame."""
        if not self.snapshots:
            return pl.DataFrame()

        rows = []
        for snapshot in self.snapshots:
            rows.append({
                "timestamp": snapshot["timestamp"],
                "symbol": snapshot["symbol"],
                "best_bid_price": snapshot["best_bid"][0],
                "best_bid_qty": snapshot["best_bid"][1],
                "best_ask_price": snapshot["best_ask"][0],
                "best_ask_qty": snapshot["best_ask"][1],
                "mid_price": snapshot["mid_price"],
                "spread": snapshot["spread"],
            })

        return pl.DataFrame(rows)

    def validate(self, data: pl.DataFrame) -> bool:
        """Validate order book data."""
        if data.is_empty():
            return False

        required_cols = ["timestamp", "symbol", "side", "price", "quantity"]
        return all(col in data.columns for col in required_cols)

    def calculate_metrics(self, symbol: str) -> dict:
        """Calculate order book metrics for a symbol."""
        ob = self.order_books.get(symbol)
        if not ob:
            return {}

        snapshot = ob.snapshot(self.depth)
        bids = snapshot["bids"]
        asks = snapshot["asks"]

        if not bids or not asks:
            return {}

        # Imbalance
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)

        # Depth
        bid_depth = sum(price * qty for price, qty in bids)
        ask_depth = sum(price * qty for price, qty in asks)

        # Microprice
        best_bid_p, best_bid_q = bids[0]
        best_ask_p, best_ask_q = asks[0]
        microprice = (
            (best_bid_p * best_ask_q + best_ask_p * best_bid_q)
            / (best_bid_q + best_ask_q)
        )

        return {
            "symbol": symbol,
            "timestamp": ob.timestamp,
            "bid_ask_spread": snapshot["spread"],
            "bid_ask_spread_bps": snapshot["spread"] / snapshot["mid_price"] * 10000,
            "mid_price": snapshot["mid_price"],
            "microprice": microprice,
            "imbalance": imbalance,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_imbalance": (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10),
        }
