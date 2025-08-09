#!/usr/bin/env python3

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=["Weighted_Price"])

df = df.rename(columns={"Timestamp": "Date"})

df["Date"] = pd.to_datetime(df["Date"], unit="s")

df = df.set_index("Date")

df["Close"] = df["Close"].ffill()
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

df = df[df.index >= "2017-01-01"]

daily_df = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
    })

daily_df[["High", "Low", "Open", "Close"]].plot(figsize=(12, 6))
plt.title("Daily Bitcoin Prices (2017 and Beyond)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend(["High", "Low", "Open", "Close"])

plt.savefig("bitcoin_prices.png")

try:
    plt.show()
except Exception:
    print("Plot saved as 'bitcoin_prices.png' since the environment does not support interactive plotting.")

print(daily_df)
