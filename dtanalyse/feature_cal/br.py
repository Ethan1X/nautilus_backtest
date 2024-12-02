import sys, os
import pandas as pd
from decimal import Decimal

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.config import BacktestRunConfig, BacktestVenueConfig, BacktestDataConfig, BacktestEngineConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.instruments import *
from nautilus_trader.model.data import QuoteTick,OrderBookDelta,TradeTick
from nautilus_trader.model.identifiers import InstrumentId,Venue
from nautilus_trader.persistence.catalog import ParquetDataCatalog

sys.path.append('.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from etl.runner import ChronoRunner
from etl.catalog import ChronoDataCatalog
from etl.config import ChronoDataConfig

from fm import start_time, end_time

'''
Simple backtest process
'''
catalog = ChronoDataCatalog("/data/catalog", show_query_paths=False)
START_BALANCE = ["0 BTC", "0 ETH", "0 FIL", "0 MATIC", "100000 USDT"]
EXCHANGE = "BINANCE"

instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, "BTCUSDT.BINANCE", maker_fee="-0.001", taker_fee="0.002")
# instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, "ETHSDT.BINANCE", maker_fee="-0.001", taker_fee="0.002")
# instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, "FILUSDT.BINANCE", maker_fee="-0.001", taker_fee="0.002")
# instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, "MATICUSDT.BINANCE", maker_fee="-0.001", taker_fee="0.002")

# instrument = catalog.clone_instrument("/data/catalog", CryptoPerpetual, "BTCUSDT.BINANCE_PERP", rewrite=True, maker_fee="-0.001", taker_fee="0.002")
# instrument = catalog.clone_instrument("/data/catalog", CryptoPerpetual, "ETHSDT.BINANCE_PERP", rewrite=True, maker_fee="-0.001", taker_fee="0.002")
# instrument = catalog.clone_instrument("/data/catalog", CryptoPerpetual, "FILUSDT.BINANCE_PERP", rewrite=True, maker_fee="-0.001", taker_fee="0.002")
# instrument = catalog.clone_instrument("/data/catalog", CryptoPerpetual, "MATICUSDT.BINANCE_PERP", rewrite=True, maker_fee="-0.001", taker_fee="0.002")


print(instrument)
data_configs = [
    # ChronoDataConfig(
    #     catalog_path=str(catalog.path),
    #     data_cls=OrderBookDelta,
    #     instrument_id=instrument.id,
    #     start_time=start_time,
    #     end_time=end_time,
    #     file_batch_size=5,
    # ),
    # ChronoDataConfig(
    #     catalog_path=str(catalog.path),
    #     data_cls=QuoteTick,
    #     instrument_id=instrument.id,
    #     start_time=start_time,
    #     end_time=end_time,
    #     file_batch_size=5,
    # ),
    ChronoDataConfig(
        catalog_path=str(catalog.path),
        data_cls=TradeTick,
        instrument_id=instrument.id,
        start_time=start_time,
        end_time=end_time,
        file_batch_size=5,
    )
]

book_type = "L2_MBP"  # Ensure data book type matches venue book type
venues_configs = [
   # BacktestVenueConfig(
   #     name=f'{EXCHANGE}_PERP',
   #     oms_type="NETTING",
   #     account_type="MARGIN",
   #     base_currency=None,
   #     starting_balances=START_BALANCE,
   #     book_type=book_type, # Venues book type
   # ),
    BacktestVenueConfig(
        name=EXCHANGE,
        oms_type="NETTING",
        account_type="CASH",
        base_currency=None,
        starting_balances=START_BALANCE,
        book_type=book_type, # Venues book type
    )
]

strategies = [
    ImportableStrategyConfig(
        strategy_path="fm:FeatureManager",
        config_path="fm:FeatureManagerConfig",
        config=dict(
            instrument_id=instrument.id,
            data_mode="central",
            # trade_conf_fp="/Users/jungle/TradingSystem/BackTestSys/ntrader/tests/as_maker/trade_confs/binance/asm_conf_BTC_usdt.toml",
        ),
    ),
]

# NautilusTrader currently exceeds the rate limit for Jupyter notebook logging (stdout output),
# this is why the `log_level` is set to "ERROR". If you lower this level to see
# more logging then the notebook will hang during cell execution. A fix is currently
# being investigated which involves either raising the configured rate limits for
# Jupyter, or throttling the log flushing from Nautilus.
# https://github.com/jupyterlab/jupyterlab/issues/12845
# https://github.com/deshaw/jupyterlab-limit-output
config = BacktestRunConfig(
    engine=BacktestEngineConfig(
        trader_id="BACKTESTER-808",
        strategies=strategies,
        logging=LoggingConfig(log_level="INFO"),
    ),
    data=data_configs,
    venues=venues_configs,
)

node = ChronoRunner(configs=[config])
result = node.run()
# print(result)

engine: BacktestEngine = node.get_engine(config.id)
#
#engine.trader.generate_order_fills_report()
#engine.trader.generate_positions_report()
#engine.trader.generate_account_report(Venue("BINANCE"))
