from sec_edgar_downloader import Downloader
import os

email = "aakashsangwan024@gmail.com"
dl = Downloader("./downloaded_filings",email)

tickers = ["AAPL"] 

for ticker in tickers:
    dl.get("10-K", ticker, after="1995-01-01", before="2023-12-31")

print("Download completed.")
