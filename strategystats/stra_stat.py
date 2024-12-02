import numpy as np
import tomli
import tomli_w
import time

from .stat_data import *
from .stat_func import *
from .stat_indicator import Indicators
from .report2pdf import Report2Pdf


####################################################
# 统计接口
####################################################
def plot_from_npz(plot_config, npz_path, run_dir=None, period="all", plot_title=None, interval=1):
    if run_dir is None:
        run_dir = npz_path
    with open(f"./{npz_path}/symbol_info.toml", "rb") as f:
        symbol_info = tomli.load(f)
    with open(f"./{npz_path}/stat_info.toml", "rb") as f:
        stat_info = tomli.load(f)
        
    price_npz = np.load(f"./{npz_path}/market_price.npz", allow_pickle=True)
    balance_npz = np.load(f"./{npz_path}/balance.npz", allow_pickle=True)
    net_value_npz = np.load(f"./{npz_path}/net_value.npz", allow_pickle=True)
    hedge_npz = np.load(f"./{npz_path}/hedge.npz", allow_pickle=True)
    signal_npz = np.load(f"./{npz_path}/signal.npz", allow_pickle=True)
    
    # draw_stat_plot_html(symbol_info, stat_info, price_npz, balance_npz, net_value_npz, hedge_npz, signal_npz, plot_config, run_dir, period, plot_title, interval)

    report_png = Report2Pdf()
    report_png.draw_stat_plot_png(symbol_info, stat_info, price_npz, balance_npz, net_value_npz, hedge_npz, signal_npz, plot_config, run_dir, period, plot_title, interval)


class StrategyStatsRecord(object):
    def __init__(self, initial_balance: SymbolBalance):
        self.initial_balance = duplicate_data(initial_balance)           # 初始balance
        # self.order_list: list[OrderInfo] = []
        self.order_list: OrderList = OrderList(initial_balance.trading_symbol)
        self.price_list: list[MarketPriceInfo] = []
        self.signal_list: list[SignalInfo] = []
        self._terminated_orders = None
        self._balance_list = None
        self._hedge_list = None
        self._net_value_list = None
        self._indicators = None

        self.start_time = None
        self.end_time = None

    @property
    def terminated_orders(self):
        if self._terminated_orders is None:
            self._terminated_orders = select_terminated_orders(self.order_list)

        # print(f'_terminated_orders: {self._terminated_orders}')
        # print(f'terminated: {len(self.order_list.order_id)}, {len(self.order_list.index.keys())}, {len(self._terminated_orders)}')
        return self._terminated_orders
    
    @property
    def balance_list(self):
        if self._balance_list is None:
            self._balance_list = adjust_balance(self.initial_balance, self.terminated_orders, self.order_list)
        return self._balance_list

    def set_balance_list(self, balance_list: BalanceList):
        self._balance_list = balance_list

    @property
    def hedge_list(self):
        # print(f'_hedge_list: {self._hedge_list}')

        if self._hedge_list is None:
            self._hedge_list = matching_orders(self.terminated_orders, self.order_list, self.initial_balance.trading_symbol)
        return self._hedge_list

    @property
    def net_value_list(self):
        if self._net_value_list is None:
            # self._net_value_list = calculate_net_value(self.balance_list, self.price_list, self.order_list["ts"][self.terminated_orders[0]], self.order_list["ts"][self.terminated_orders[-1]])
            self._net_value_list = calculate_net_value(self.balance_list, self.price_list)
        return self._net_value_list

    @property
    def indicators(self, period=360*24*60*60):
        if self._indicators is None:
            self._indicators = Indicators(self.balance_list, self.hedge_list, self.net_value_list, period)
        return self._indicators
        
    @classmethod
    def from_npz(cls, init_balance: SymbolBalance, run_dir: str, start_time=None, end_time=None):
        # self.initial_balance = init_balance
        
        stats_record = StrategyStatsRecord(init_balance)
        symbol_info = init_balance.trading_symbol
        # print("from_npz start")
        
        price_npz = np.load(f"{run_dir}/market_price.npz", allow_pickle=True)
        order_npz = np.load(f"{run_dir}/orders.npz", allow_pickle=True)
        signal_npz = np.load(f"{run_dir}/signal.npz", allow_pickle=True)
        price_list = {}
        order_list = {}
        signal_list = {}

        start_ts = start_time.value // 1e6 if start_time is not None else -np.inf
        end_ts = end_time.value // 1e6 if end_time is not None else np.inf
        # print(f'{start_ts} {end_ts}')


        # idxs = [idx for idx, ts in enumerate(order_npz["ts"]) if ts < start_ts or ts > end_ts]
        _begin = time.time()
        for key, arr in order_npz.items():
            # order_list[key] = np.delete(arr, idxs)
            order_list[key] = order_npz[key][(order_npz["ts"] >= start_ts) & (order_npz["ts"] <= end_ts)]
            # print(f'load orders: {key} {len(order_list[key])}')
        _end = time.time()
        # odd_count = np.array([1 for i, v in enumerate(order_list["status"]) if order_list["filled_amount"][i] > 0]).sum()
        # print(f'select orders: {_end-_begin}, odd orders: {odd_count}; {order_list["status"][0]}, {type(order_list["status"][0])}')
        _begin = _end
        
        # idxs = [idx for idx, ts in enumerate(price_npz["ts"]) if ts < start_ts or ts > end_ts]
        for key, arr in price_npz.items():
            # price_list[key] = np.delete(arr, idxs)
            price_list[key] = price_npz[key][(price_npz["ts"] >= start_ts) & (price_npz["ts"] <= end_ts)]

        # idxs = [idx for idx, ts in enumerate(signal_npz["ts"]) if ts < start_ts or ts > end_ts]
        for key, arr in signal_npz.items():
            # signal_list[key] = np.delete(arr, idxs)
            signal_list[key] = signal_npz[key][(signal_npz["ts"] >= start_ts) & (signal_npz["ts"] <= end_ts)]

        # NOTE: 加载npz之后才可使用各计算功能
        # stats_record.order_list = order_list
        stats_record.order_list = OrderList(symbol_info)
        stats_record.order_list.load_npz(order_list)
        stats_record.price_list = price_list
        stats_record.signal_list = signal_list
        stats_record.start_time = start_time
        stats_record.end_time = end_time

        # print(f'order_list: {order_list}')
        # print(f'price_list: {price_list}')
        # print(f'signal_list: {signal_list}')

        # stats_record.signal_list = []
        # for ts, price, amount, name in zip(signal_npz["ts"], signal_npz["price"], signal_npz["amount"], signal_npz["name"]):
        #     if start_ts > ts:
        #         continue
        #     if end_ts < ts:
        #         break
        #     stats_record.signal_list.append(SignalInfo(ts=ts, price=price, amount=amount, name=name))

        # stats_record.price_list = []
        # for ts, mid_price in zip(price_npz["ts"], price_npz["mid_price"]):
        #     if start_ts > ts:
        #         continue
        #     if end_ts < ts:
        #         break
        #     stats_record.price_list.append(MarketPriceInfo(mid_price=mid_price, tp=ts))
            
        # stats_record.order_list = []
        # for order_id, price, amount, filled_price, filled_amount, trade_side, order_type, tp, commission, trade_type, status in zip(order_npz["order_id"], order_npz["price"], 
        #                                                                                                                             order_npz["amount"], order_npz["filled_price"],
        #                                                                                                                             order_npz["filled_amount"],
        #                                                                                                                             order_npz["side"],order_npz["order_type"], order_npz["ts"],
        #                                                                                                                             order_npz["commission"], order_npz["liquidity_side"],
        #                                                                                                                             order_npz["status"]):
        #     if start_ts > tp:
        #         continue
        #     if end_ts < tp:
        #         break
        #     stats_record.order_list.append(OrderInfo(order_id=order_id, symbol_info=symbol_info, price=price, amount=amount, filled_price=filled_price, filled_amount=filled_amount, 
        #                                      trade_side=getattr(TradeSide, trade_side),
        #                                      order_type=getattr(OrderType, order_type), 
        #                                      tp=tp, commission=commission, 
        #                                      trade_type=getattr(TradeType, trade_type), 
        #                                      status=getattr(OrderStatus, status)
        #                                     ))

        return stats_record

    def statistics(self, run_dir):
        # 先close
        price_info = PriceInfo(price=self.price_list["mid_price"][-1], tp=self.price_list["ts"][-1])
        last_balance = self.balance_list.get_at(-1)
        close_balance, _profit = close_all_positions(last_balance, price_info)
        self.balance_list.update(close_balance.capitals, close_balance.position, close_balance.holding_price, close_balance.ts_event)

        # 存symbol
        symbol = self.initial_balance.trading_symbol
        with open(f"{run_dir}/symbol_info.toml", "wb") as f:
            tomli_w.dump(symbol.to_dict(), f)

        # 存stat_info
        print(' ***** calculate_all_indicators *****')
        stat_info, freq_returns = self.indicators.calculate_all_indicators()
        f = open(f"{run_dir}/stat_analysis_info.text", "w")
        print(f'年化夏普率: {stat_info.sharpe_ratio:.3f}%; 年化calmar比率: {stat_info.calmar_ratio:.3f}%; 单日胜率: {stat_info.daily_win_rate:.3f}%;')
        print(f'回测时长：{stat_info.trading_days:.2}天；总盈利：{stat_info.total_returns-stat_info.total_commissions:.3f}u；手续费：{stat_info.total_commissions:.3f}u；年化收益率：{stat_info.annual_returns:.3f}%；最大回撤率：{stat_info.maxdrawdown_rate:.3f}%；最大回撤时长：{stat_info.drawdown_interval:.3}h；日换手率：{stat_info.daily_turnover_rate:.3f}%；平均持仓时间：{stat_info.average_holding_time:.3f}s;')
        print(f'不含手续费：胜率：{stat_info.win_percentage_without_commission:.3f}%；单次盈利率：{stat_info.average_win_percentage_without_commission:.5f}%；单次亏损率：{stat_info.average_loss_percentage_without_commission:.5f}%；总体单次盈亏率：{stat_info.average_returns_without_commission:.5f}%;')
        print(f'含手续费：胜率：{stat_info.win_percentage_with_commission_without_zero:.3f}%；单次盈利率：{stat_info.average_win_percentage_with_commission_without_zero:.5f}%；单次亏损率：{stat_info.average_loss_percentage_with_commission_without_zero:.5f}%；总体单次盈亏率：{stat_info.average_returns_with_commission_without_zero:.5f}%;')
        print(stat_info)

        with open(f"{run_dir}/stat_info.toml", "wb") as f:
            tomli_w.dump(stat_info.__dict__, f)
        freq_returns.to_csv(f"./{run_dir}/freq_returns.csv")

        # 存balance
        np.savez(f"{run_dir}/balance.npz", 
                 quote_capital=np.array(self.balance_list.capitals[symbol.quote]), 
                 token_capital=np.array(self.balance_list.capitals[symbol.token]), 
                 position=np.array(self.balance_list.position), 
                 holding_price=np.array(self.balance_list.holding_price), 
                 ts=np.array(self.balance_list.ts_event)
                )
        
        # 存net value
        np.savez(f"{run_dir}/net_value.npz", 
                 net_value=np.array(self.net_value_list.net_values), 
                 ts=np.array(self.net_value_list.ts_event),
                )

        # 存hedge
        np.savez(f"{run_dir}/hedge.npz",
                 open_side=np.array([hedge.open_side.name for hedge in self.hedge_list]),
                 open_order_id=np.array([str(hedge.open_order_id) for hedge in self.hedge_list]),
                 close_order_id=np.array([str(hedge.close_order_id) for hedge in self.hedge_list]),
                 amount_close=np.array([hedge.amount_close for hedge in self.hedge_list]),
                 avg_price_close=np.array([hedge.avg_price_close for hedge in self.hedge_list]),
                 ts_finish=np.array([hedge.ts_finish for hedge in self.hedge_list]),
                 avg_price_open=np.array([hedge.avg_price_open for hedge in self.hedge_list]),
                 amount_open=np.array([hedge.amount_open for hedge in self.hedge_list]),
                 commissions=np.array([hedge.commissions for hedge in self.hedge_list]),
                 ts_create=np.array([hedge.ts_create for hedge in self.hedge_list]),
                 total_value=np.array([hedge.total_value for hedge in self.hedge_list]),
                 total_returns=np.array([hedge.total_returns for hedge in self.hedge_list]),
                )
        
        # 存terminated orders
        # np.savez(f"{run_dir}/terminated_orders.npz",
        #         order_id = np.array([self.order_list["order_id"][idx] for idx in self.terminated_orders]),
        #         status = np.array([self.order_list["status"][idx] for idx in self.terminated_orders]),
        #         price = np.array([self.order_list["price"][idx] for idx in self.terminated_orders]),
        #         amount = np.array([self.order_list["amount"][idx] for idx in self.terminated_orders]),
        #         side = np.array([self.order_list["side"][idx]  for idx in self.terminated_orders]),
        #         liquidity_side = np.array([self.order_list["liquidity_side"][idx] for idx in self.terminated_orders]),
        #         order_type = np.array([self.order_list["order_type"][idx] for idx in self.terminated_orders]),
        #         filled_price = np.array([self.order_list["filled_price"][idx] for idx in self.terminated_orders]),
        #         filled_amount = np.array([self.order_list["filled_amount"][idx] for idx in self.terminated_orders]),
        #         commission = np.array([self.order_list["commission"][idx] for idx in self.terminated_orders]),
        #         ts = np.array([self.order_list["ts"][idx] for idx in self.terminated_orders]),
        #         )
        np.savez(f"{run_dir}/terminated_orders.npz",
                order_id = np.array([self.order_list.order_id[idx] for idx in self.terminated_orders]),
                status = np.array([self.order_list.status[idx] for idx in self.terminated_orders]),
                price = np.array([self.order_list.price[idx] for idx in self.terminated_orders]),
                amount = np.array([self.order_list.amount[idx] for idx in self.terminated_orders]),
                side = np.array([self.order_list.side[idx]  for idx in self.terminated_orders]),
                # liquidity_side = np.array([self.order_list.liquidity_side[idx] for idx in self.terminated_orders]),
                trade_type = np.array([self.order_list.trade_type[idx] for idx in self.terminated_orders]),
                order_type = np.array([self.order_list.order_type[idx] for idx in self.terminated_orders]),
                filled_price = np.array([self.order_list.filled_price[idx] for idx in self.terminated_orders]),
                filled_amount = np.array([self.order_list.filled_amount[idx] for idx in self.terminated_orders]),
                commission = np.array([self.order_list.commission[idx] for idx in self.terminated_orders]),
                ts = np.array([self.order_list.ts[idx] for idx in self.terminated_orders]),
                )

        np.savez(f"./{run_dir}/market_price.npz", 
                 **self.price_list
                )

        orders = self.order_list.to_dict()
        del orders['unique_order']
        del orders['unique_index']
        np.savez(f"./{run_dir}/orders.npz",
                 **orders
                )
        np.savez(f"./{run_dir}/signal.npz", 
                 **self.signal_list
                )
    
    def to_npz(self, run_dir):        
        # # 先close
        # close_balance, _profit = close_all_positions(self.balance_list[-1], self.price_list[-1])
        # self.balance_list.append(close_balance)
        
        # 存symbol
        symbol = self.initial_balance.trading_symbol
        with open(f"./{run_dir}/symbol_info.toml", "wb") as f:
            tomli_w.dump(symbol.to_dict(), f)

        # # 存stat_info
        # stat_info, freq_returns = self.indicators.calculate_all_indicators()
        # print(stat_info)
        # with open(f"./{run_dir}/stat_info.toml", "wb") as f:
        #     tomli_w.dump(stat_info.__dict__, f)
        # freq_returns.to_csv(f"./{run_dir}/freq_returns.csv")
        
        # 存price
        # ask_price = np.array([price_info.ask_info.price for price_info in self.price_list])
        # ask_amount = np.array([price_info.ask_info.amount for price_info in self.price_list])
        # bid_price = np.array([price_info.bid_info.price for price_info in self.price_list])
        # bid_amount = np.array([price_info.bid_info.amount for price_info in self.price_list])
        mid_price = np.array([price_info.mid_price.price for price_info in self.price_list])
        ts = np.array([price_info.ts_event for price_info in self.price_list])
        # np.savez(f"./{run_dir}/market_price.npz", ask_price=ask_price, ask_amount=ask_amount, bid_price=bid_price, bid_amount=bid_amount, mid_price=mid_price, ts=ts)
        np.savez(f"./{run_dir}/market_price.npz", mid_price=mid_price, ts=ts)

        # 存balance
        # quote_capital = np.array([balance_info.capitals[symbol.quote] for balance_info in self.balance_list])
        # token_capital = np.array([balance_info.capitals[symbol.token] for balance_info in self.balance_list])
        # position = np.array([balance_info.position for balance_info in self.balance_list])
        # holding_price = np.array([balance_info.holding_price for balance_info in self.balance_list])
        # ts = np.array([balance_info.ts_event for balance_info in self.balance_list])
        # np.savez(f"./{run_dir}/balance.npz", quote_capital=quote_capital, token_capital=token_capital, position=position, holding_price=holding_price, ts=ts)
        
        # 存net value
        # net_value = np.array([nv.net_value for nv in self.net_value_list])
        # ts = np.array([nv.ts for nv in self.net_value_list])
        # np.savez(f"./{run_dir}/net_value.npz", net_value=net_value, ts=ts)

        # 存signal
        name = np.array([signal.name for signal in self.signal_list])
        price = np.array([signal.price for signal in self.signal_list])
        amount = np.array([signal.amount for signal in self.signal_list])
        ts = np.array([signal.ts for signal in self.signal_list])
        np.savez(f"./{run_dir}/signal.npz", name=name, price=price, amount=amount, ts=ts)

        # 存hedge
        # open_side = np.array([hedge.open_side.name for hedge in self.hedge_list])
        # open_order_id = np.array([str(hedge.open_order_id) for hedge in self.hedge_list]) # TODO: can be improved
        # close_order_id = np.array([str(hedge.close_order_id) for hedge in self.hedge_list]) # TODO: can be improved
        # amount_close = np.array([hedge.amount_close for hedge in self.hedge_list])
        # avg_price_close = np.array([hedge.avg_price_close for hedge in self.hedge_list])
        # ts_finish = np.array([hedge.ts_finish for hedge in self.hedge_list])
        # avg_price_open = np.array([hedge.avg_price_open for hedge in self.hedge_list])
        # amount_open = np.array([hedge.amount_open for hedge in self.hedge_list])
        # commissions = np.array([hedge.commissions for hedge in self.hedge_list])
        # ts_create = np.array([hedge.ts_create for hedge in self.hedge_list])
        # total_value = np.array([hedge.total_value for hedge in self.hedge_list])
        # total_returns = np.array([hedge.total_returns for hedge in self.hedge_list])
        # np.savez(f"./{run_dir}/hedge.npz",
        #          open_side=open_side,
        #          open_order_id=open_order_id,
        #          close_order_id=close_order_id,
        #          amount_close=amount_close,
        #          avg_price_close=avg_price_close,
        #          ts_finish=ts_finish,
        #          avg_price_open=avg_price_open,
        #          amount_open=amount_open,
        #          commissions=commissions,
        #          ts_create=ts_create,
        #          total_value=total_value,
        #          total_returns=total_returns,
        #         )
        
        # 存terminated orders
        # order_id = np.array([order.order_id for order in self.terminated_orders])
        # status = np.array([order.status.name for order in self.terminated_orders])
        # price = np.array([order.price for order in self.terminated_orders])
        # amount = np.array([order.amount for order in self.terminated_orders])
        # side = np.array([order.side.name for order in self.terminated_orders])
        # liquidity_side = np.array([order.trade_type.name for order in self.terminated_orders])
        # order_type = np.array([order.order_type.name for order in self.terminated_orders])
        # filled_price = np.array([order.filled_price for order in self.terminated_orders])
        # filled_amount = np.array([order.filled_amount for order in self.terminated_orders])
        # commission = np.array([order.commission for order in self.terminated_orders])
        # ts = np.array([order.ts for order in self.terminated_orders])
        # np.savez(f"./{run_dir}/terminated_orders.npz",
        #          order_id=order_id,
        #          status=status,
        #          price=price,
        #          amount=amount,
        #          side=side,
        #          liquidity_side=liquidity_side,
        #          order_type=order_type,
        #          filled_price=filled_price,
        #          filled_amount=filled_amount,
        #          commission=commission,
        #          ts=ts,
        #         )

        # 存orders
        # order_id = np.array([order.order_id for order in self.order_list])
        # status = np.array([order.status.name for order in self.order_list])
        # price = np.array([order.price for order in self.order_list])
        # amount = np.array([order.amount for order in self.order_list])
        # side = np.array([order.side.name for order in self.order_list])
        # liquidity_side = np.array([order.trade_type.name for order in self.order_list])
        # order_type = np.array([order.order_type.name for order in self.order_list])
        # filled_price = np.array([order.filled_price for order in self.order_list])
        # filled_amount = np.array([order.filled_amount for order in self.order_list])
        # commission = np.array([order.commission for order in self.order_list])
        # ts = np.array([order.ts for order in self.order_list])
        # np.savez(f"./{run_dir}/orders.npz",
        #          order_id=order_id,
        #          status=status,
        #          price=price,
        #          amount=amount,
        #          side=side,
        #          liquidity_side=liquidity_side,
        #          order_type=order_type,
        #          filled_price=filled_price,
        #          filled_amount=filled_amount,
        #          commission=commission,
        #          ts=ts,
        #         )
        orders = self.order_list.to_dict()
        del orders['unique_order']
        del orders['unique_index']
        np.savez(f"./{run_dir}/orders.npz",
                 **orders
                )