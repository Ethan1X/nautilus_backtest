时间对象格式：
start_time = pd.Timestamp("2024-03-01 00:00:00", tz="HONGKONG")
end_time =  pd.Timestamp("2024-03-04 00:00:00", tz="HONGKONG")


在本地目录下建立etl目录的软链接：
（终端命令行操作）
ln -s /data/backtest_nt/etl .


【重要】
在第一次运行自己的回测程序之前，执行下述命令，在本地目录下建立catalog目录，保存自己的交易对配置
（终端命令行操作）
sh /data/backtest_nt/init_catalog.sh /data/catalog
注意：runner程序中对于数据路径的指定，改为指向本地目录
    catalog = ChronoDataCatalog("./catalog", show_query_paths=False)
                                ～～
【重要】
请在终端下启动回测程序。
在启动前，执行命令
conda activate nautilus


查看自己的程序进程
ps aux | grep 程序名称
可以查找到自己程序的进程号

查看自己程序的运行情况
top -p 自己程序的进程号
（在top的显示界面，按c查看进程的具体执行对象，按m查看总体的内存占用情况）


============================================================================
on_order_filled 事件响应函数中

event（OrderFilled对象）的主要数据：
            amount_filled = float(event.last_qty.as_decimal())
            price_filled = float(event.last_px.as_decimal())
            fee = float(event.commission.as_decimal())
            update_timestamp = event.ts_event
(订单买卖方向可能是event.aggressor_side，取值为AggressorSide.BUYER或AggressorSide.SELLER)


新下订单时
        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=side,
            price=Price(price, precision=self.instrument.price_precision),
            # quantity=self.instrument.make_qty(self.trade_size),
            quantity=self.instrument.make_qty(qty),
            post_only=True,
            time_in_force=TimeInForce.GTD,
            expire_time=expire_time,
            emulation_trigger=TriggerType["NO_TRIGGER"]
        )
这是限价单，
post_only 为True时表示“只挂单：如果能直接成交，则撤单”，maker；如果想做taker，此项设为False
订单类型有多种，限价单（limit）只是其中一种，其参数也可能有更多。
side表示订单买卖方向，取值范围：OrderSide.BUY，OrderSide.SELL


生成并发送订单样例：
ask_order_id = self.new_order(ap, sell_amount, OrderSide.SELL, self.hedge_index)
bid_order_id = self.new_order(bp, buy_amount, OrderSide.BUY, self.hedge_index)


核心数据类型提要
OrderBook
bids(), .asks()
ts_event

QuoteTick
ask_price：Price, ask_size：Quantity
bid_price：Price,ask_size：Quantity
ts_event：int

TradeTick
price：Price, size：Quantity
trade_id, aggressor_side
ts_event


***** strategystats结构 *****

stra_stat层级调用顺序:
    - 加载生成报告 from stat_load
        - load_period_result 

    - 策略状态 from stra_stat

    - 功能函数 from stat_func
        - 计算净值 （资产相关）
        - 匹配订单（订单相关）
        - adjust_balance（订单相关）

    - 指标计算功能函数 from stat_indicator
    
    - 回测数据结构 from stat_data
        - 市场信息
        - 交易对信息
        - 资产信息
        - 订单数据
        - 订单匹配数据
        - 行情数据（价格，市场价格，净值，净值列表）
        - 统计指标名称

    - 绘图功能 from stat_plot