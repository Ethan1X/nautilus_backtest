import datetime
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pytz
import random
import boto3
import io

from collections import deque
from typing import List, NamedTuple
from recover_depth import recoveryDepth


NUM_SAMPLES_MEMORY_LIMIT = 20_000  # 35%
NUM_SAMPLES_PER_FILE = 100
PRICE_FLUCTUATION = 4/10000

BUCKET = 'dp4intern'
S3_PATH = 'vinci_data'


def load_depth_iter(exchange, symbol, date, num_depths):
    predata = None
    # Read one more hour to get pre_data & last 137 seconds of data
    for hour in range(-1, 24):
        hour_time = date + datetime.timedelta(hours=hour)
        hour_str = hour_time.strftime("%Y%m%d%H")
        hour_depth, predata = recoveryDepth(
            exchange, symbol, hour_str, num_depths, pre_data=predata)
        logging.info("%s depth size %s" % (hour_str, len(hour_depth)))
        for depth in hour_depth:
            yield depth


class JointDepth(NamedTuple):
    timestamp: int
    depth: list


class Depth(NamedTuple):
    bids: list
    asks: list


class JointTradePrices(NamedTuple):
    timestamp: int
    depth: list


class TradePrices(NamedTuple):
    bids: list
    asks: list


class DepthPointer(object):
    def __init__(self, iterator):
        self._iter = iterator
        self.current = iterator.__next__()
        self.next = iterator.__next__()

    def step(self):
        self.current = self.next
        self.next = self._iter.__next__()


def join_depth(exchange_symbol_list, date, num_depths, time_list):
    pointers = [DepthPointer(load_depth_iter(exchange, symbol, date, num_depths))
        for exchange, symbol in exchange_symbol_list]
    joint_depth = []

    try:
        for t in time_list:
            lastest = max(pointer.current["time"] for pointer in pointers)
            if t < lastest:
                continue
            current_depth_list = []
            for pointer in pointers:
                while pointer.next["time"] <= t:
                    pointer.step()
                if pointer.current["time"] - t > 10 * 1000:
                    raise

                bids = [(d["p"], d["s"]) for d in pointer.current["bids"]]
                asks = [(d["p"], d["s"]) for d in pointer.current["asks"]]
                current_depth_list.append(Depth(bids, asks))
            joint_depth.append(JointDepth(t, current_depth_list))
    except StopIteration:
      # StopIteration will be raised if one of the DepthPointers ends.
      return joint_depth

    return joint_depth


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(formatter)
    # logger.addHandler(streamHandler)


def datetime_range(time_begin, time_end, step):
    ret = []
    while time_begin < time_end:
        ret.append(int(time_begin.timestamp() * 1000))
        time_begin = time_begin + step
    return ret


def load_data(exchange_list, symbol_prefix_list, date, step_in_second, num_depths, preceding_len):
    suffix_list = [""]
    symbol_list = [symbol_prefix + suffix for symbol_prefix in symbol_prefix_list for suffix in suffix_list]

    step = datetime.timedelta(seconds=step_in_second)
    time_list = datetime_range(date-preceding_len*step, date + datetime.timedelta(days=1), step)

    exchange_symbol_list = [(e, s) for e in exchange_list for s in symbol_list]
    return join_depth(exchange_symbol_list, date, num_depths, time_list)


def get_average_prices_of_given_queries(price_amount_tuples: list, 
                                        order_quantity_query_list: list):
    logger = logging.getLogger('depth_statistics')
    cumulative_values = np.cumsum([price * amount for price, amount in price_amount_tuples])
    depth_overflow = cumulative_values[-1] - order_quantity_query_list[-1]
    logger.info('Remaining value is: %s', depth_overflow)
    cumulative_amount = np.cumsum([amount for _, amount in price_amount_tuples])
    average_price_list = []
    previous_order_amount, previous_order_value, i = 0, 0, 0

    for order_quantity in order_quantity_query_list:
        
        while cumulative_values[i] < order_quantity:
            previous_order_value = cumulative_values[i]
            previous_order_amount = cumulative_amount[i]
            if i < len(price_amount_tuples)-1:
                i+=1
            else:
                break
        
        remaining_order_quantity = order_quantity - previous_order_value
        next_order_price = price_amount_tuples[i][0]
        total_order_amount = previous_order_amount + (remaining_order_quantity / next_order_price)
        average_price = order_quantity / total_order_amount
        average_price_list.append(average_price)
    
    return np.array(average_price_list)


def calculate_prices_of_query_on_row(depth_data: Depth, 
                                     order_quantity_query_list: list):
    bids = depth_data.bids
    asks = depth_data.asks
    bids_avg_price = get_average_prices_of_given_queries(bids, order_quantity_query_list)
    asks_avg_price = get_average_prices_of_given_queries(asks, order_quantity_query_list)
    avg_prices = np.concatenate((np.flip(bids_avg_price), asks_avg_price), axis=None)
    return avg_prices


def apply_average_price_transform_on_window(joint_depth_window: list, order_quantity_query_list: list):
    joint_average_price_list = []

    for _, joint_depth in joint_depth_window:
        joint_average_price = [calculate_prices_of_query_on_row(depth_data, 
                                                                order_quantity_query_list) for depth_data in joint_depth]
        joint_average_price_list.append(joint_average_price)
    
    return np.array(joint_average_price_list)


def get_last_midprice_in_window(joint_depth_list: list) -> float:
    best_quoted_prices = [quote[0][0] for depth in joint_depth_list[-1][1] 
                          for quote in (depth.bids, depth.asks)]
    return np.mean(best_quoted_prices)


def get_mean_in_window(array):
    return np.mean(array)


def get_variance_in_window(array):
    return np.var(array)


def normalize_price_range(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def grayscale_transform_window(array):
    return np.clip((array * 255.9999).astype(np.uint8), a_min=0, a_max=255)


def create_plot(array):
    # get the number of channels we need to plot
    n_pictures = array.shape[1]

    # calculate the number of rows and columns for the subplots
    rows = int(np.sqrt(n_pictures))
    cols = n_pictures // rows
    if rows * cols < n_pictures:
        cols += 1

    # create the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # plot each 2d array as a grayscale image on its own subplot
    for i in range(n_pictures):
        axes[i].imshow(array[:, i, :], cmap='gray')
        axes[i].axis('off')
    
    # remove remaining empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def draw(joint_depth_list: list, order_quantity_query_list:list):
    average_price_window = apply_average_price_transform_on_window(joint_depth_list, 
                                                                   order_quantity_query_list)
    mean = get_mean_in_window(average_price_window)
    variance = get_variance_in_window(average_price_window)
    print(f"The mean is {mean:.4f} and the variance is {variance:.4f}.")
    last_midprice = get_last_midprice_in_window(joint_depth_list)
    mean_removed_window = average_price_window - last_midprice
    normalized_window = normalize_price_range(mean_removed_window)
    grayscale_window = grayscale_transform_window(normalized_window)
    create_plot(grayscale_window)
    return grayscale_window


class DepthSample(NamedTuple):
    feature: np.array
    label: np.array
    bid_direction: List[int]
    ask_direction: List[int]
    mean: float
    variance: float


def create_sample(joint_prices_window: List[JointTradePrices], feature_len: int, label_len: int):
    
    window_array = np.array([[np.concatenate((np.flip(depth.bids), depth.asks)) 
                              for depth in joint_depth.depth] for joint_depth in joint_prices_window])
    feature_array = window_array[:feature_len,:,:]
    label_array = window_array[feature_len:feature_len+label_len]
    
    mean = np.mean(feature_array)
    variance = np.var(feature_array)
    last_midprice = np.mean([quote[0] for depth in joint_prices_window[feature_len].depth 
                             for quote in depth])

    feature_mean_removed_window = feature_array - last_midprice
    min_value = np.min(feature_mean_removed_window)
    range_value = np.max(feature_mean_removed_window) - min_value
    feature_normalized_window = (feature_mean_removed_window - min_value) / range_value

    # label calculation
    label_mean_removed_window = label_array - last_midprice
    label_normalized_window = (label_mean_removed_window - min_value) / range_value

    # Statistics to log to extreme values in each window
    logger = logging.getLogger('sample_window_statistics')
    feature_percent_change = feature_array / last_midprice - 1
    logger.info("Feature | Max percent change is %s | Min percent change is %s", 
                np.max(feature_percent_change), np.min(feature_percent_change))
    
    label_percent_change = label_array / last_midprice - 1
    logger.info("Label | Max percent change is %s | Min percent change is %s", 
            np.max(label_percent_change), np.min(label_percent_change))
    
    feature_best_bids = np.fromiter((channel.bids[0] 
                                     for channel in joint_prices_window[feature_len-1].depth), 
                                     dtype=float)
    label_best_asks = np.fromiter((channel.asks[0] 
                                   for channel in joint_prices_window[feature_len+label_len-1].depth), 
                                   dtype=float)
    best_bids_percent_change = label_best_asks/feature_best_bids - 1
    bids_direction = np.select([best_bids_percent_change > PRICE_FLUCTUATION, 
                                best_bids_percent_change < -PRICE_FLUCTUATION], 
                                [1, -1], default=0).tolist()

    feature_best_asks = np.fromiter((channel.asks[0] 
                                     for channel in joint_prices_window[feature_len-1].depth), 
                                     dtype=float)
    label_best_bids = np.fromiter((channel.bids[0] 
                                   for channel in joint_prices_window[feature_len+label_len-1].depth), 
                                   dtype=float)
    best_asks_percent_change = label_best_bids/feature_best_asks - 1
    asks_direction = np.select([best_asks_percent_change > PRICE_FLUCTUATION, 
                                best_asks_percent_change < -PRICE_FLUCTUATION], 
                                [1, -1], default=0).tolist()

    return DepthSample(feature_normalized_window, label_normalized_window, 
                       bids_direction, asks_direction, mean, variance)


def construct_depth(exchange_list, symbol_prefix_list):   
    timezone = pytz.timezone("Asia/Shanghai")
    start_date = timezone.localize(datetime.datetime(2023, 6, 1))
    num_days = 40
    num_train = int(num_days * 0.6)
    num_val = int(num_days * 0.2)
    num_test = num_days - num_train - num_val

    for i in range(num_days):
        current_date = start_date + datetime.timedelta(days=i)
        filename = current_date.strftime('%Y-%m-%d')
        if i < num_train:
            directory_name = 'train'
        elif i < num_train + num_val:
            directory_name = 'validation'
        else:
            directory_name = 'test'
        
        joint_depth_list = load_data(exchange_list, symbol_prefix_list, current_date, 1, 20, 128+10-1)
        save_npz_to_s3(directory_name, filename, joint_depth_list)


def joint_depth_to_price(joint_depth: JointDepth, order_quantity_query_list: list):
    price_list = []

    for depth in joint_depth.depth:
        asks_price = get_average_prices_of_given_queries(depth.asks, order_quantity_query_list)
        bids_price = get_average_prices_of_given_queries(depth.bids, order_quantity_query_list)
        price_list.append(TradePrices(bids_price, asks_price))
    
    return JointTradePrices(joint_depth.timestamp, price_list)


def depth_generator(folder_name, feature_len, label_len, order_quantity_query_list: list):
    total_len = label_len + feature_len
    npz_files = get_npzfile_list(folder_name)
    logger = logging.getLogger('run')

    # Find the files in the format "2023-05-01.npz"
    date_files = []
    for filename in npz_files:
        basename = filename.split('/')[-1]
        try:
            date = datetime.datetime.strptime(basename.split('.')[0], '%Y-%m-%d')
            date_files.append((date, filename))
        except ValueError as e:
            logger.error(f"Invalid File: {filename}", exc_info=True)
            continue
    # Sort by date since we need to manage a cache of the depth data (by time)
    date_files.sort()

    # Queue to store the data
    queue = deque(maxlen=total_len)

    # Yield the next depth data if queue is full
    for _, filename in date_files:
        data = load_npz_from_s3(filename)

        for i, data_array in enumerate(data['data']):
            depth_data = JointDepth(data_array[0], data_array[1])

            # We want to check if the next file can be added onto the previous file's cache
            if i == 0 and queue:
                prev_timestamp = queue[-1].timestamp
                current_timestamp = depth_data.timestamp

                if current_timestamp != prev_timestamp + 1000:
                    queue.clear()
            
            queue.append(joint_depth_to_price(depth_data, order_quantity_query_list))

            if len(queue) < total_len:
                continue
            yield list(queue)


def save_npz(depth_folder_name: str, file_index: str, output_data):
    sample_folder_name = f'{depth_folder_name}_samples'
    file_name = f'{depth_folder_name}_{file_index}_samples.npz'
    sample_file_path = os.path.join(os.getcwd(), sample_folder_name, file_name)
    np.savez_compressed(sample_file_path, data=np.array(output_data, dtype=object))
    logger = logging.getLogger('run')
    logger.info('Saved to %s', sample_file_path)
    
    
def get_npzfile_list(folder_name: str):
    client = boto3.client('s3')
    res = client.list_objects_v2(Bucket=BUCKET, Prefix=f'{S3_PATH}/{folder_name}')
    
    return [obj['Key'] for obj in res.get('Contents', []) if obj.get('Key', '').endswith('.npz')]


def load_npz_from_s3(file_name: str):
    client = boto3.client('s3')
    buffer = io.BytesIO()
    client.download_fileobj(BUCKET, file_name, buffer)
    buffer.seek(0)
    
    return np.load(buffer, allow_pickle=True)


def save_npz_to_s3(folder_name: str, file_name: str, output_data):
    file_path = f'{S3_PATH}/{folder_name}/{file_name}.npz'
    client = boto3.client('s3')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.array(output_data, dtype=object))
    buffer.seek(0)
    client.upload_fileobj(buffer, BUCKET, file_path)
    logger = logging.getLogger('run')
    logger.info('Saved to %s', file_path)


def shuffle_and_slice_samples(sample_list: list, size: int):
    random.shuffle(sample_list)
    return [sample_list[i:i+size] for i in range(0, len(sample_list), size)]


def sample_generator(depth_folder_name, feature_len, label_len, order_quantity_query_list):
    # Generator for depth data
    data_generator = depth_generator(depth_folder_name, feature_len, label_len, order_quantity_query_list)
    sample_list = []
    file_index = 1
    sample_folder_name = f'{depth_folder_name}_samples'

    # For each possible depth window, we create a sample
    for data in data_generator:
        sample = create_sample(data, feature_len, label_len)
        sample_list.append(sample)

        if len(sample_list) >= NUM_SAMPLES_MEMORY_LIMIT:
            random.shuffle(sample_list)
            output_data = sample_list[:NUM_SAMPLES_PER_FILE]
            file_name = f'{depth_folder_name}_{file_index}_samples'
            save_npz_to_s3(sample_folder_name, file_name, output_data)
            file_index += 1
            del sample_list[:NUM_SAMPLES_PER_FILE]
   
    for file in shuffle_and_slice_samples(sample_list, NUM_SAMPLES_PER_FILE):
        file_name = f'{depth_folder_name}_{file_index}_samples'
        save_npz_to_s3(sample_folder_name, file_name, file)
        file_index += 1


def main():
    exchange_list = ('binance', 'okex')
    symbol_prefix_list = ('btc_usdt', 'eth_usdt', 'bch_usdt', 'xrp_usdt', 'ltc_usdt', 'link_usdt')
    
    construct_depth(exchange_list, symbol_prefix_list)
    funds_unit = [2 ** pow * 1000 for pow in range(0,16)]
    sample_generator('train', 128, 10, funds_unit)
    sample_generator('validation', 128, 10, funds_unit)
    sample_generator('test', 128, 10, funds_unit)


if __name__ == "__main__":
    setup_logger('sample_window_statistics', 'window_stats.log')
    setup_logger('depth_statistics', 'depth_stats.log')
    setup_logger('run', 'runtime.log')
    main()
