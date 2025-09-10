import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from utils.debug import logger
import sys
import os

CSV_PATH = os.path.join("data", "fundamentals_wide.csv")
MAX_POSITION = 100000


class valueStrat(autoTrader):
    """
    Quarterly long/short Value-Growth strategy:
    - Rebalance only on actual post-quarter fundamental release dates
    - Long bottom-N value stocks, short top-N growth stocks
    - Exit all other positions during each rebalance
    """

    def __init__(self, name, universe, start: str, end: str, n_positions: int = 10):
        super().__init__(name)
        self.n_positions = n_positions
        self.start_ts = pd.to_datetime(start)
        self.end_ts = pd.to_datetime(end)

        self.ESSENTIALS = ["pe_ratio", "pb_ratio"]
        self.EXTRA_FEATS = ["ev_ebitda", "ev_sales", "fcf_yield"]
        self.MAX_TICK_MISS = 0.40
        self.MIN_TKRS = 20

        self._load_fundamentals()
        self._detect_features()
        self._compute_rebalance_dates()
        self._initialize_universe(universe)

    def _load_fundamentals(self):
        df_wide = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0, parse_dates=True)
        df_wide = df_wide.sort_index().loc[self.start_ts: self.end_ts].ffill()
        self.fund_panel = df_wide

    def _detect_features(self):
        cols = self.fund_panel.columns.get_level_values(1)
        feats = [f for f in (self.ESSENTIALS + self.EXTRA_FEATS) if f in cols]
        missing = set(self.ESSENTIALS) - set(feats)
        if missing:
            raise ValueError(f"Missing essential metrics: {missing}")
        self.features = feats

    def _compute_rebalance_dates(self):
        # Use true release dates closest after each calendar quarter-end
        q_ends = pd.date_range(self.start_ts, self.end_ts, freq="Q").normalize()
        raw = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0, parse_dates=False).sort_index()
        rel_df = raw.xs("release_date", level=1, axis=1)

        rel_series = (
            rel_df
            .stack()
            .pipe(lambda s: pd.to_datetime(s, errors="coerce"))
            .dropna()
            .dt.normalize()
        )

        reb_dates = []
        for q in q_ends:
            future = rel_series[rel_series >= q]
            if not future.empty:
                reb_dates.append(future.min())
        self.reb_dates = sorted(set(reb_dates))

    def _initialize_universe(self, universe):
        tickers = self.fund_panel.columns.get_level_values(0).unique()
        self.universe = [t for t in (universe or OBData.assets) if t in tickers]
        self.assets = self.universe

    def strategy(self, orderClass):
        today = pd.to_datetime(OBData.Date[OBData.step]).normalize()

        if today not in self.reb_dates:
            # No rebalance today, carry forward AUM
            self.historical_AUM.append(self.AUM_available)
            orderClass.filled_order(self)
            return

        # Retrieve fundamental data for today
        s = self.fund_panel.loc[today]
        df = s.unstack(level=1)[self.features]
        df = df.reindex(self.universe).copy()

        # Impute missing values with cross-sectional median
        df = df.fillna(df.median())

        # Drop assets with too many missing values
        tick_imp = df.isna().mean(axis=1)
        df = df[tick_imp <= self.MAX_TICK_MISS]
        if len(df) < self.MIN_TKRS:
            print(f"  ⚠️ Only {len(df)} tickers available, skip")
            return

        # Normalize features: compute z-scores
        arr = df.values
        mu = np.nanmean(arr, axis=0)
        sigma = np.nanstd(arr, axis=0) + 1e-6
        z_np = (arr - mu) / sigma
        scores = np.nanmean(z_np, axis=1)

        # Select top-N high (growth) and bottom-N low (value) scores
        idx_long = np.argpartition(scores, self.n_positions)[:self.n_positions]
        idx_short = np.argpartition(scores, -self.n_positions)[-self.n_positions:]
        longs = [df.index[i] for i in idx_long]
        shorts = [df.index[i] for i in idx_short]

        # Equal-weight sizing
        w_long = 1.0 / len(longs) if longs else 0
        w_short = 1.0 / len(shorts) if shorts else 0

        # Place long orders
        for asset in longs:
            price = OBData.currentPrice(asset)
            quantity = (MAX_POSITION * w_long)
            orderClass.send_order(self, asset, price, quantity)
            self.AUM_available -= quantity
            self.orderID += 1

        # Place short orders
        for asset in shorts:
            price = OBData.currentPrice(asset)
            quantity = (MAX_POSITION * w_short)
            orderClass.send_order(self, asset, price, -quantity)
            self.AUM_available += quantity
            self.orderID += 1

        # Flatten unselected positions
        keep = set(longs + shorts)
        for asset in self.inventory.keys():
            position = self.inventory[asset]["quantity"]
            if asset not in keep and position != 0:
                price = OBData.currentPrice(asset)
                orderClass.send_order(self, asset, price, -position)
                self.AUM_available += position if position < 0 else -position
                self.orderID += 1

        # Record AUM and process fills
        self.historical_AUM.append(self.AUM_available)
        orderClass.filled_order(self)
