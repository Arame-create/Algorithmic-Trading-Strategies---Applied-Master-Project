import pandas as pd
import numpy as np
from collections import deque

from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from utils.debug import logger
import sys

MAX_INVENTORY = 10000  # Max allowed inventory per asset


class rsiStrat(autoTrader):

    def __init__(self, name, window=50, sellThreshold=70, buyThreshold=40, alpha=0.02):
        super().__init__(name)
        self.name = name
        self.asset_list = OBData.assets
        self.asset_count = len(self.asset_list)

        self.windowLengt = window
        self.windowRSI = [deque(maxlen=self.windowLengt) for _ in range(self.asset_count)]

        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha  # Smoothing factor for EMA
        self.historical_RSI = [[] for _ in range(self.asset_count)]

    def compute_RSI(self, asset):
        idx = OBData.assetIdx[asset] - 1
        self.windowRSI[idx].append(OBData.currentPrice(asset))

        # Price difference (delta) from previous step
        if len(self.windowRSI[idx]) > 1:
            delta = self.windowRSI[idx][-1] - self.windowRSI[idx][-2]
        else:
            delta = 0

        # Initialize average gain/loss arrays if not done yet
        if not hasattr(self, "avg_gain"):
            self.avg_gain = np.zeros(self.asset_count)
            self.avg_loss = np.zeros(self.asset_count)

        gain = max(delta, 0)
        loss = max(-delta, 0)

        # EMA smoothing for gain/loss
        self.avg_gain[idx] = (1 - self.alpha) * self.avg_gain[idx] + self.alpha * gain
        self.avg_loss[idx] = (1 - self.alpha) * self.avg_loss[idx] + self.alpha * loss

        # RSI calculation
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

    def strategy(self, orderClass):
        """
        RSI-based trading strategy:
        - Buy when RSI is below buyThreshold
        - Sell when RSI is above sellThreshold
        - Exit positions near RSI mean reversion
        - Includes stop-loss
        """
        for asset in self.asset_list:
            current_price = OBData.currentPrice(asset)
            entry_price = self.inventory[asset]["price"]
            position = self.inventory[asset]["quantity"]

            # Stop-loss logic
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
            if rsi is None:
                continue

            # Buy signal
            if rsi <= self.buyThreshold:
                if self.inventory[asset]["quantity"] < MAX_INVENTORY and self.AUM_available:
                    price, quantity = current_price, 1000
                    orderClass.send_order(self, asset, price, quantity)
                    self.AUM_available -= quantity
                    self.orderID += 1

            # Sell signal
            elif rsi >= self.sellThreshold:
                if self.inventory[asset]["quantity"] > -MAX_INVENTORY:
                    price, quantity = current_price, 1000
                    orderClass.send_order(self, asset, price, -quantity)
                    self.AUM_available += quantity
                    self.orderID += 1

            # Exit short position near neutral RSI
            elif rsi <= 50 + (self.sellThreshold - 50) / 2 and self.inventory[asset]["quantity"] < 0:
                price, quantity = current_price, abs(self.inventory[asset]["quantity"])
                orderClass.send_order(self, asset, price, quantity)
                self.AUM_available -= quantity
                self.orderID += 1

            # Exit long position near neutral RSI
            elif rsi <= 50 - (50 - self.buyThreshold) / 2 and self.inventory[asset]["quantity"] > 0:
                price, quantity = current_price, abs(self.inventory[asset]["quantity"])
                orderClass.send_order(self, asset, price, -quantity)
                self.AUM_available += quantity
                self.orderID += 1

        # Update filled orders and AUM history
        self.historical_AUM.append(self.AUM_available)
        orderClass.filled_order(self)
