# lstm_cpp

### LSTM In C++ Code 
1. realize the training procedure: BPTT in C++ 
2. ToDo list


### LSTM In Pyhton Compared with Moving Average  
- Dataset from : [New York Stock Exchange S&P 500 companies historical prices with fundamental data](https://www.kaggle.com/dgawlik/nyse)


#### Dataset consists of following files:
1. prices.csv: raw, as-is daily prices. Most of data spans from 2010 to the end 2016, for companies new on stock market date range is shorter. There have been approx. 140 stock splits in that time, this set doesn't account for that.
2. prices-split-adjusted.csv: same as prices, but there have been added adjustments for splits.
3. securities.csv: general description of each company with division on sectors
4. fundamentals.csv: metrics extracted from annual SEC 10K fillings (2012-2016), should be enough to derive most of popular fundamental indicators.


### Inspiration

Here is couple of things one could try out with this data:

- One day ahead prediction: Rolling Linear Regression, ARIMA, Neural Networks, LSTM
- Momentum/Mean-Reversion Strategies
- Security clustering, portfolio construction/hedging

Which company has biggest chance of being bankrupt? Which one is undervalued (how prices behaved afterwards), what is Return on Investment?
