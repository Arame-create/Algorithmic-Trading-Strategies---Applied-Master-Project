import pandas as pd
import numpy as np
from collections import deque

from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from utils.debug import logger

MAX_INVENTORY = 10000  # Max allowed inventory for any asset

class momentumStrat(autoTrader):
    
    def __init__(self, name,
                short_window: int = 20, long_window: int = 200,
                RSI_window=50, sellThreshold=70,
                buyThreshold=30, alpha=0.05):
        
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window

        self.asset_list = OBData.assets
        self.asset_count = len(self.asset_list)

        # Store price history for moving average computation
        self.prices = [deque(maxlen=self.long_window+1) for _ in range(self.asset_count)]
        self.historical_short_ma = [[] for _ in range(self.asset_count)]
        self.historical_long_ma = [[] for _ in range(self.asset_count)]
        self.short_sums = np.zeros(self.asset_count)
        self.long_sums = np.zeros(self.asset_count)

        # RSI settings and buffers
        self.windowLengt = RSI_window
        self.windowRSI = [deque(maxlen=self.windowLengt) for _ in range(self.asset_count)]
        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha  # EMA smoothing factor
        self.historical_RSI = [[] for _ in range(self.asset_count)]

    def compute_RSI(self, asset):
        # Compute RSI using EMA of gains and losses
        idx = OBData.assetIdx[asset]-1
        self.windowRSI[idx].append(OBData.currentPrice(asset))

        if len(self.windowRSI[idx]) > 1:
            delta = self.windowRSI[idx][-1] - self.windowRSI[idx][-2]
        else:
            delta = 0

        if not hasattr(self, "avg_gain"):
            self.avg_gain = np.zeros(self.asset_count)
            self.avg_loss = np.zeros(self.asset_count)

        gain = max(delta, 0)
        loss = max(-delta, 0)

        self.avg_gain[idx] = (1 - self.alpha) * self.avg_gain[idx] + self.alpha * gain
        self.avg_loss[idx] = (1 - self.alpha) * self.avg_loss[idx] + self.alpha * loss

        if self.avg_loss[idx] == 0:
            rsi = 100
        else:
            rs = self.avg_gain[idx] / self.avg_loss[idx]
            rsi = 100 - (100 / (1 + rs))

        if len(self.windowRSI[idx]) < self.windowLengt:
            self.historical_RSI[idx].append(None)
            return None
        else:
            self.historical_RSI[idx].append(rsi)
            return rsi

    def calculate_moving_averages(self, asset):
        # Compute short and long moving averages with fast rolling sum
        idx = OBData.assetIdx[asset]-1
        new_price = OBData.currentPrice(asset)
        price_queue = self.prices[idx]
        price_queue.append(new_price)

        if len(price_queue) <= self.short_window:
            self.short_sums[idx] += new_price
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            return None, None

        elif len(price_queue) < self.long_window:
            self.short_sums[idx] += new_price - price_queue[-self.short_window-1]
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            return None, None

        self.short_sums[idx] += new_price - price_queue[-self.short_window-1]
        self.long_sums[idx] += new_price - price_queue[0]

        short_ma = self.short_sums[idx] / self.short_window
        long_ma = self.long_sums[idx] / self.long_window

        self.historical_short_ma[idx].append(short_ma)
        self.historical_long_ma[idx].append(long_ma)

        return short_ma, long_ma

    def strategy(self, orderClass):
        """
        Momentum strategy using moving average crossovers and RSI signals.
        Includes stop-loss logic.
        """
        for asset in self.asset_list:
            current_price = OBData.currentPrice(asset)
            entry_price = self.inventory[asset]["price"]
            position = self.inventory[asset]["quantity"]

            # Stop loss for long or short position
            if position > 0 and entry_price > 0:
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct < -0.5:
                    price, quantity = current_price, abs(position)
                    orderClass.send_order(self, asset, price, -quantity)
                    self.AUM_available += quantity
                    self.orderID += 1
                    continue
            elif position < 0 and entry_price > 0:
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct < -0.5:
                    price, quantity = current_price, abs(position)
                    orderClass.send_order(self, asset, price, quantity)
                    self.AUM_available -= quantity
                    self.orderID += 1
                    continue

            rsi = self.compute_RSI(asset)
            short_ma, long_ma = self.calculate_moving_averages(asset)

            if rsi is None or short_ma is None or long_ma is None:
                pass
            else:
                # Entry signals
                if short_ma > long_ma and rsi < self.sellThreshold:
                    if self.inventory[asset]["quantity"] < MAX_INVENTORY and self.AUM_available > 0:
                        price, quantity = current_price, min(1000, self.AUM_available)
                        orderClass.send_order(self, asset, price, quantity)
                        self.AUM_available -= quantity
                        self.orderID += 1

                elif short_ma < long_ma and rsi > self.buyThreshold:
                    if self.inventory[asset]["quantity"] > -MAX_INVENTORY:
                        price, quantity = current_price, 1000
                        orderClass.send_order(self, asset, price, -quantity)
                        self.AUM_available += quantity
                        self.orderID += 1

                # Exit signals based on RSI reverting toward neutral (50)
                elif rsi <= 50 + (self.sellThreshold - 50) / 2 and self.inventory[asset]["quantity"] < 0:
                    price, quantity = current_price, abs(self.inventory[asset]["quantity"])
                    orderClass.send_order(self, asset, price, quantity)
                    self.AUM_available -= quantity
                    self.orderID += 1

                elif rsi <= 50 - (50 - self.buyThreshold) / 2 and self.inventory[asset]["quantity"] > 0:
                    price, quantity = current_price, abs(self.inventory[asset]["quantity"])
                    orderClass.send_order(self, asset, price, -quantity)
                    self.AUM_available += quantity
                    self.orderID += 1

        self.historical_AUM.append(self.AUM_available)
        orderClass.filled_order(self)
