import pandas  as pd
import numpy as np
from .stat_data import *
from datetime import datetime, timedelta


def convert_ts(ts):
    return datetime.fromtimestamp(ts_to_second(ts))


class Indicators():
    """指标计算"""

    def __init__(self, balance_list, hedge_list, net_value_list, period=365*24*60*60):

        # 改用npz
        # self.period = period
        # self.convert_info(balance_list, net_value_list)
        # self.balance_list = balance_list['position']
        # self.hedge_list = hedge_list
        # self.init_flag = False
        # self.stat_info = StatisticInfo()
        # self.returns = self.calculate_returns()

        self.period = period
        self.convert_info(balance_list, net_value_list)
        self.balance_list = balance_list
        self.hedge_list = hedge_list
        self.init_flag = False
        self.stat_info = StatisticInfo(balance_list.trading_symbol, balance_list.get_at(0))
        self.returns = self.calculate_returns()
        self.frequency = '1h'

    def convert_info(self, balance_list, net_value_list):
        # 转换数据格式
        """
        :param net_values: 资产净值序列（按时间戳）
        :param capital: 初始资金
        :param total_trading_value: 总交易额
        :param trade_time: 交易起止时间间隔（按时间戳）
        :param timestamps: 时间戳序列
        :param period: 年交易周期（默认s）
        """
        self.trading_symbol = balance_list.trading_symbol.quote
        self.init_capital = balance_list.capitals[self.trading_symbol][0]
        self.timestamps = [datetime.fromtimestamp(ts_to_second(ts)) for ts in net_value_list.ts_event]
        self.trade_time = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        self.net_values = pd.Series(net_value_list.net_values)
        self.trading_days = self.trade_time / (60 * 60 * 24)


        # # 改用npz
        # self.init_capital = balance_list['quote_capital'][0]
        # vectorized_function = np.vectorize(convert_ts)
        # self.timestamps = vectorized_function(net_value_list['ts'])  # todo:修改时间格式
        # self.trade_time = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        # self.net_values = pd.Series(net_value_list['net_value'])
        # self.trading_days = self.trade_time / (60 * 60 * 24)
    
    def calculate_returns(self):
        # 计算收益率
        return self.net_values.pct_change()

    def calculate_frequency_returns(self):
        # 计算给定频率下的收益率分布
        sr = pd.Series(self.net_values.values, index=self.timestamps)
        sr = sr.resample(self.frequency).last()
        freq_returns = sr.pct_change()
        return freq_returns

    def calculate_daily_returns(self):
        sr = pd.Series(self.net_values.values, index=self.timestamps)
        sr = sr.resample('1d').last()
        self.daily_returns = sr.pct_change()

    def calculate_daily_win_rate(self):
        return (self.daily_returns.dropna() > 0).mean()

    def calculate_sharpe_ratio(self):
        # 计算夏普比率
        sharpe_ratio = self.stat_info.annual_returns / (np.nanstd(self.daily_returns) * np.sqrt(365) * 100)
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, target_return=0):
        # 计算下行比率
        average_return = self.returns.mean()
        downside_returns = self.returns[self.returns < target_return]
        # downside_deviation = np.sqrt((downside_returns ** 2).mean())
        downside_deviation = downside_returns.std()
        sortino_ratio = (average_return - target_return) / downside_deviation if downside_deviation != 0 else 0
        return sortino_ratio

    def calculate_maxdrawdown(self):
        # 计算最大回撤, 最大回撤比例，最大回撤时间
        i = np.argmax((np.maximum.accumulate(self.net_values) - self.net_values))
        if i == 0:
            j = 0
        else:
            j = np.argmax(self.net_values[:i])
        maxdrawdown = self.net_values[j] - self.net_values[i] 
        maxdrawdown_rate = maxdrawdown / self.net_values[j] 
        drawdown_interval = (self.timestamps[i] - self.timestamps[j]).total_seconds() / 3600
        return maxdrawdown, maxdrawdown_rate, drawdown_interval

    def calculate_total_returns_rate(self):
        # 计算总收益率
        return self.net_values.iloc[-1] / self.net_values.iloc[0] - 1

    def calculate_annual_returns(self):
        # 计算年化收益率
        total_returns = self.net_values.iloc[-1] / self.net_values.iloc[0] - 1
        # annual_returns = (1 + total_returns) ** (self.period / self.trade_time) - 1  # 计算复利
        annual_returns = total_returns * (self.period / self.trade_time)  # 计算单利
        
        return annual_returns
    
    def calculate_turnover_rate(self):
        # 计算换手率
        turnover_rate = self.stat_info.total_trading_value / self.init_capital
        return turnover_rate

    def calculate_daily_turnover_rate(self):
        # 计算日度换手率
        daily_turnover_rate = (self.stat_info.total_trading_value / self.trading_days) / self.init_capital
        return daily_turnover_rate

    def calculate_calmar_ratio(self):
        # 计算卡尔马比率
        if not self.init_flag:
            maxdrawdown_rate = self.calculate_maxdrawdown()[1]
            annual_return = self.calculate_annual_return()
            calmar_ratio = annual_return / maxdrawdown_rate
        else:
            calmar_ratio = self.stat_info.annual_returns / self.stat_info.maxdrawdown_rate
        return calmar_ratio

    def calculate_trading_indicators_without_commission(self):
        win_counts, loss_counts = 0, 0
        win_amounts, loss_amounts = 0, 0
        win_value, loss_value = 0, 0
        sum_holding_time = 0
        win_vol, loss_vol = 0, 0

        # print(f'calculate_trading_indicators_without_commission _hedge_list: {self.hedge_list}')

        for hedge in self.hedge_list:

            self.stat_info.total_trading_value += hedge.total_value
            # if hedge.commissions == 0:
                # print(f"Zero commission: {hedge}")
            self.stat_info.total_commissions += hedge.commissions
            self.stat_info.total_returns += hedge.total_returns
            if hedge.ts_finish != 0:
                sum_holding_time += (ts_to_second(hedge.ts_finish) - ts_to_second(hedge.ts_create))
            if hedge.total_returns > 0:
                win_counts += 1
                win_amounts += hedge.total_returns
                win_value += hedge.total_value
                win_vol += hedge.amount_open
            elif hedge.total_returns <= 0:
                loss_counts += 1
                loss_amounts += abs(hedge.total_returns)
                loss_value += hedge.total_value
                loss_vol += hedge.amount_open

        average_win_amount = win_amounts / win_counts if win_counts > 0 else 0  # 单次盈利均值
        average_win_percentage = win_amounts / win_value if win_value > 0 else 0  # 单次盈利率
        average_loss_amount = loss_amounts / loss_counts if loss_counts > 0 else 0  # 单次亏损均值
        average_loss_percentage = loss_amounts / loss_value if loss_value > 0 else 0  # 单次亏损率
        win_loss_rate = average_win_percentage / average_loss_percentage if average_loss_percentage != 0 else 1  # 盈亏比
        win_percentage = win_vol / (win_vol + loss_vol) if (win_vol + loss_vol) != 0 else 0 # 胜率
        order_list = []
        for hedge in self.hedge_list:
            order_list.extend(hedge.open_order_id)
            order_list.extend(hedge.close_order_id)
        order_set = set(order_list)
        trading_counts = len(order_set)  # 交易次数
        daily_trading_counts = trading_counts / self.trading_days  # 日度交易次数
        if len(self.hedge_list) > 0:
            average_holding_time = sum_holding_time / len(self.hedge_list)  # 平均持仓时间
        else:
            average_holding_time = 0
    
        self.stat_info.win_counts_without_commission = win_counts
        self.stat_info.loss_counts_without_commission = loss_counts
        self.stat_info.win_percentage_without_commission = win_percentage * 100
        self.stat_info.win_loss_rate_without_commission = win_loss_rate
        self.stat_info.average_win_amount_without_commission = average_win_amount
        self.stat_info.average_loss_amount_without_commission = -average_loss_amount
        self.stat_info.average_win_percentage_without_commission = average_win_percentage * 100
        self.stat_info.average_loss_percentage_without_commission = -average_loss_percentage * 100
        self.stat_info.average_returns_without_commission = self.stat_info.total_returns / self.stat_info.total_trading_value * 100
        self.stat_info.average_holding_time = average_holding_time
        self.stat_info.trading_counts = trading_counts
        self.stat_info.daily_trading_counts = daily_trading_counts

    def calculate_trading_indicators_with_commission(self):
        win_counts, loss_counts = 0, 0
        win_amounts, loss_amounts = 0, 0
        win_value, loss_value = 0, 0
        win_vol, loss_vol = 0, 0

        for hedge in self.hedge_list:
            if hedge.total_returns - hedge.commissions > 0:
                win_counts += 1
                win_amounts += (hedge.total_returns - hedge.commissions)
                win_value += hedge.total_value
                win_vol += hedge.amount_open
            elif hedge.total_returns - hedge.commissions <= 0:
                loss_counts += 1
                loss_amounts += abs(hedge.total_returns - hedge.commissions)
                loss_value += hedge.total_value
                loss_vol += hedge.amount_open

        average_win_amount = win_amounts / win_counts if win_counts > 0 else 0  # 单次盈利均值
        average_win_percentage = win_amounts / win_value if win_value > 0 else 0  # 单次盈利率
        average_loss_amount = loss_amounts / loss_counts if loss_counts > 0 else 0  # 单次亏损均值
        average_loss_percentage = loss_amounts / loss_value if loss_value > 0 else 0  # 单次亏损率
        win_loss_rate = average_win_percentage / average_loss_percentage if average_loss_percentage != 0 else 1  # 盈亏比
        win_percentage = win_vol / (win_vol + loss_vol) if (win_vol + loss_vol) != 0 else 0 # 胜率
    
        self.stat_info.win_counts_with_commission_without_zero = win_counts
        self.stat_info.loss_counts_with_commission_with_zero = loss_counts
        self.stat_info.win_percentage_with_commission_without_zero = win_percentage * 100
        self.stat_info.win_loss_rate_with_commission_without_zero = win_loss_rate
        self.stat_info.average_win_amount_with_commission_without_zero = average_win_amount
        self.stat_info.average_loss_amount_with_commission_with_zero = -average_loss_amount
        self.stat_info.average_win_percentage_with_commission_without_zero = average_win_percentage * 100
        self.stat_info.average_loss_percentage_with_commission_with_zero = -average_loss_percentage * 100
        self.stat_info.average_returns_with_commission_without_zero = (self.stat_info.total_returns - self.stat_info.total_commissions)  / self.stat_info.total_trading_value * 100

        win_counts, loss_counts = 0, 0
        win_amounts, loss_amounts = 0, 0
        win_value, loss_value = 0, 0
        win_vol, loss_vol = 0, 0

        for hedge in self.hedge_list:
            if hedge.total_returns - hedge.commissions >= 0:
                win_counts += 1
                win_amounts += (hedge.total_returns - hedge.commissions)
                win_value += hedge.total_value
                win_vol += hedge.amount_open
            elif hedge.total_returns - hedge.commissions < 0:
                loss_counts += 1
                loss_amounts += abs(hedge.total_returns - hedge.commissions)
                loss_value += hedge.total_value
                loss_vol += hedge.amount_open

        average_win_amount = win_amounts / win_counts if win_counts > 0 else 0  # 单次盈利均值
        average_win_percentage = win_amounts / win_value if win_value > 0 else 0  # 单次盈利率
        average_loss_amount = loss_amounts / loss_counts if loss_counts > 0 else 0  # 单次亏损均值
        average_loss_percentage = loss_amounts / loss_value if loss_value > 0 else 0  # 单次亏损率
        win_loss_rate = average_win_percentage / average_loss_percentage if average_loss_percentage != 0 else 1  # 盈亏比
        win_percentage = win_vol / (win_vol + loss_vol) if (win_vol + loss_vol) != 0 else 0 # 胜率
    
        self.stat_info.win_counts_with_commission_with_zero = win_counts
        self.stat_info.loss_counts_with_commission_without_zero = loss_counts
        self.stat_info.win_percentage_with_commission_with_zero = win_percentage * 100
        self.stat_info.win_loss_rate_with_commission_with_zero = win_loss_rate
        self.stat_info.average_win_amount_with_commission_with_zero = average_win_amount
        self.stat_info.average_loss_amount_with_commission_without_zero = -average_loss_amount
        self.stat_info.average_win_percentage_with_commission_with_zero = average_win_percentage * 100
        self.stat_info.average_loss_percentage_with_commission_without_zero = -average_loss_percentage * 100

    def calculate_all_indicators(self):
        # 汇总所有统计指标
        print(' ***** calculate_trading_indicators_without_commission *****')
        self.calculate_trading_indicators_without_commission()  # 交易统计指标

        print(' ***** calculate_trading_indicators_with_commission *****')
        self.calculate_trading_indicators_with_commission()  # 交易统计指标(考虑手续费)

        self.init_flag = True
        
        self.stat_info.trading_days = self.trading_days

        print(' ***** calculate_daily_returns *****')
        self.calculate_daily_returns()  # 日度收益率

        print(' ***** calculate_daily_win_rate *****')
        daily_win_rate = self.calculate_daily_win_rate()  # 日地胜率
        self.stat_info.daily_win_rate = daily_win_rate * 100
        
        print(' ***** calculate_total_returns_rate *****')
        total_returns_rate = self.calculate_total_returns_rate()  # 总收益率
        self.stat_info.total_returns_rate = total_returns_rate * 100
        
        print(' ***** calculate_annual_returns *****')
        annual_returns = self.calculate_annual_returns()  # 年化收益率
        self.stat_info.annual_returns = annual_returns * 100
        
        print(' ***** calculate_maxdrawdown *****')
        maxdrawdown, maxdrawdown_rate, drawdown_interval = self.calculate_maxdrawdown()  # 最大回撤
        self.stat_info.maxdrawdown = maxdrawdown
        self.stat_info.maxdrawdown_rate = maxdrawdown_rate * 100
        self.stat_info.drawdown_interval = drawdown_interval
        
        print(' ***** calculate_sharpe_ratio *****')
        sharpe_ratio = self.calculate_sharpe_ratio()  # 夏普比率
        self.stat_info.sharpe_ratio = sharpe_ratio * 100

        print(' ***** calculate_sortino_ratio *****')
        sortino_ratio = self.calculate_sortino_ratio()  # 下行风险比率
        self.stat_info.sortino_ratio = sortino_ratio * 100
        
        print(' ***** calculate_calmar_ratio *****')
        calmar_ratio = self.calculate_calmar_ratio()  # 卡尔马比率
        self.stat_info.calmar_ratio = calmar_ratio * 100
        
        print(' ***** calculate_turnover_rate *****')
        turnover_rate = self.calculate_turnover_rate()  # 换手率
        self.stat_info.turnover_rate = turnover_rate * 100
        
        print(' ***** calculate_daily_turnover_rate *****')
        daily_turnover_rate = self.calculate_daily_turnover_rate()  # 换手率
        self.stat_info.daily_turnover_rate = daily_turnover_rate * 100

        print(' ***** calculate_frequency_returns *****')
        freq_returns = self.calculate_frequency_returns()  # 给定频率下的收益率

        # print(f'含手续费且盈利包含0：胜率：{self.stat_info.win_percentage_with_commission_with_zero:.3f}%；单次盈利率：{self.stat_info.average_win_percentage_with_commission_with_zero:.5f}%；')

        return self.stat_info, freq_returns

    def load_stat_info(self):
        return [self.stat_info.__dict__]


if __name__ == "__main__":
    # 测试一下match_orders
    test_symbol = SymbolInfo("USDT", "USDT", SymbolType.SPOT_NORMAL, Market.SPOT, "OKEX", -0.00005, 0.00015)
    test_balance_list, test_hedge_list = [], []
    # tp = datetime.now()
    tp = 0
    print("balance:")
    for i in range(1, 10):
        # tp = tp + timedelta(seconds=5)
        tp = tp + 1
        balance = SymbolBalance("USDT", "USDT", SymbolType.SPOT_NORMAL, Market.SPOT, "OKEX", -0.00005, 0.00015, tp)
        balance.capitals["USDT"] = i * 1000
        balance.position = i * 0.5
        balance.holding_price = i * 100
        test_balance_list.append(balance)
        print(balance)
    # tp = datetime.now()
    tp = 0
    print("hedge:")
    for i in range(1, 5):
        # tp = tp + timedelta(seconds=2)
        tp = tp + 2
        # tp_ = tp + timedelta(seconds=1)
        tp_ = tp + 1
        hedge = HedgeInfo(TradeSide.BUY)
        hedge.total_value = i * 10
        hedge.total_returns = i * 5
        hedge.commissions = i * 1
        hedge.ts_create = tp
        hedge.ts_finish = tp_
        test_hedge_list.append(hedge)
        print(hedge)

    indicators = Indicators(test_balance_list, test_hedge_list)

    stat_info = indicators.calculate_all_indicators()
    result = {
        # net worth value indicator
        'Sharpe Ratio(%)': stat_info.sharpe_ratio,
        'Sortino Ratio(%)': stat_info.sortino_ratio,
        'Calmar Ratio(%)': stat_info.calmar_ratio,
        'Maximum Drawdown(U)': stat_info.maxdrawdown,
        'Max Drawdown Rate(%)': stat_info.maxdrawdown_rate,
        'Drawdown Interval(s)': stat_info.drawdown_interval,
        'Total Returns(%)': stat_info.total_returns_rate,
        'Annual Returns(%)': stat_info.annual_returns,
        # turnover and profit
        'Turnover Rate(%)': stat_info.turnover_rate,
        'Total Trading Value(U)': stat_info.total_trading_value,
        'Total Gain/Loss without Commissions(U)': stat_info.total_returns - stat_info.total_commissions,
        'Total Commissions(U)': stat_info.total_commissions,
        'Total Gain/Loss(U)': stat_info.total_returns,
        # win and loss indicator
        'Win Percentage(%)': stat_info.win_percentage,
        'Win Counts': stat_info.win_counts,
        'Lose Counts': stat_info.loss_counts,
        'Win/Loss Rate': stat_info.win_loss_rate,
        'Average Win Amount(U)': stat_info.average_win_amount,
        'Average Loss Amount(U)': stat_info.average_loss_amount,
        'Average Win Percentage(%)': stat_info.average_win_percentage,
        'Average Loss Percentage(%)': stat_info.average_loss_percentage,
        # holding time
        'Average Holding Time(s)': stat_info.average_holding_time,
    }
    print("stat_info: {}".format(result))
