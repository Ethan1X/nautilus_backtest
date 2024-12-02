# Usage with Nautilus Trader

假设 `StrategyStatsRecord` 对象为策略类对象的成员 `self.stats_record`.



首先，常规引用：
from strategystats.stat_data import SymbolInfo
from strategystats.stra_stat import StrategyStatsRecord, plot_from_npz
from strategystats.stat_load import initial_balance, get_capital_list
from strategystats.utils.nautilus import *
from strategystats.utils.nautilus_tool import *


## Initial Balance
其次，策略类对象的初始化

在初始化函数中，

starting_balance = ["10 BTC", "10000 USDT"] (样例)
`start_time` 是一个时间戳（int）
`symbol_info` 是一个 `SymbolInfo` 对象

capital = get_capital_list(starting_balance)
init_balance = initial_balance(self.symbol_info, capital, start_time)
self.stats_record = StrategyStatsRecord(init_balance)
self.stats_record.order_list = NautilusOrderList(symbol_info)


## Order Record
接下来，记录订单数据

新订单：通常在发出新订单的时候
self.stats_record.order_list.register(order, self.symbol_info, is_maker=True)

订单状态：通常在任何订单事件的响应中（如 on_order_filled）
self.stats_record.order_list.update(event)


## Price Record
以及记录价格数据，通常在on_quote_tick的事件响应中

self.stats_record.price_list.append(from_quote_tick(ticker))