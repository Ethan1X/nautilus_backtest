# Usage

Four kinds of data are required for the construction of a class `StrategyStatsRecord` object: a symbol, an initial balance, a price list and an order list.

## Symbol

The symbol is an object of class `SymbolInfo`, which can be read from the config files.

To read from the config files, you need to specify the `token`, `quote`, `exchange` and `symbol_type`, with an optional argument `path` indicating the customized base directory of config files.

The `token`, `quote`, `exchange` are all of type `str`.

The symbol type is of enum `SymbolType`.

Then you simply call function `load_symbol_info`:
```{python}
from strategystats.stat_func import load_symbol_info

symbol = load_symbol_info(token, quote, symbol_type, exchange)
# Or:
# symbol = load_symbol_info(token, quote, symbol_type, exchange, path)
```

## Initial Balance

The initial balance is of class `SymbolBalance`.

Its constructor requires you pass the [symbol](#symbol) and the start timestamp of your gathering of statistics.

## Price List

The member `price_list` in class `StrategyStatsRecord` should be an list of `MarketPriceInfo` objects.

This `MarketPriceInfo` object records the ticker data, namely the timestamp, the ask/bid price and the ask/bid size.

Simply append the `MarketPriceInfo` objects to `price_list` is fine.

## Order List

The member `order_list` in class `StrategyStatsRecord` should be an list of `OrderInfo` objects.

This `OrderInfo` object records your order status data, namely its submmission, fill, cancellation, etc.

You'll also need to provide other info of the order, like:
```{python}
from strategystats.stat_data import OrderInfo

OrderInfo(
    id = id, # unique order id
    symbol_info = symbol_info, # symbol info of order
    price = price, # price of order
    trade_side = trade_side, # INVALID_SIDE/BUY/SELL
    order_type = order_type, # INVALID_ORDER/LIMIT/MARKET
    trade_type = trade_type, # INVALID_TRADE/MAKER/TAKER
    amount = amount, # amount of order
    tp = tp, # timestamp of your update
    status = status, # INVALID_STATUS/OPENED/PARTLIFILLED/FILLED/CANCELED
)
```

Simply append the `OrderInfo` objects to `order_list` is fine.

## Plotting

Once all the data above are collected, call the `plot` method of class `StrategyStatsRecord` object, while providing two additional arguments: `plot_config` and `run_dir`.

`plot_config` is a dict like:
```{python}
plot_config = {
    "net_value": True, # plot net value curve
    "position": True, # plot position curve
    "stat_metrics": True, # other statistical metrics
    "order_mark": False, # mark opening & closing of positions (UNIMPLEMENTED)
    "holding_spot_value": False, # plot the value of naively holding spot instrument
}
```

`run_dir` is where the HTML plot will be saved.
