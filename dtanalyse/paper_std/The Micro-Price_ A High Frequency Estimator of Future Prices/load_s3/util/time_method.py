#encoding=utf-8

import time
import datetime
import pytz

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DATETIME_FORMAT1 = "%Y-%m-%d %H:%M"
DATETIME_FORMAT2 = "%Y-%m-%dT%H:%M"
DATETIME_FORMAT3 = "%Y/%m/%d %H:%M"
DATETIME_FORMAT4 = "%Y%m%d%H"
DATETIME_FORMAT5 = "%Y-%m-%d %H:%M:%S.%f"
DATETIME_FORMAT6 = "%Y%m%d%H%M%S"

# 东8区时区 Asia/Chongqing Asia/Shanghai
TZ_8 = pytz.timezone('Etc/GMT-8')
# 国际时间时区
TZ_0 = pytz.utc

def dt2unix(t, mill_type=1):
    # 直接使用 t.timestamp()得到时间戳
    # datetime.datetime()中可以设置时区，参数是：tzinfo，例如：
    # datetime.datetime(2022,10,10,8,46,59, tzinfo=pytz.utc)
    # if mill_type==1000:
    #     return time.mktime(t.timetuple()) * mill_type + t.microsecond/mill_type
    # else:
    #     return time.mktime(t.timetuple()) * mill_type
    return int(t.timestamp() * mill_type)

def unix2dt(t, mill_type=1, tz=None):
    '''
        tz: 表示时区
    '''
    return datetime.datetime.fromtimestamp(t / mill_type, tz=tz)

def dt2str(dt, dt_form='%Y-%m-%dT%H_%M'):
    return dt.strftime(dt_form)

def str2unix(time_str,dt_form=DATETIME_FORMAT, mill_type=1000, tz=None):
    dt = str2dt(time_str,dt_form)
    if tz is not None:
        dt = dt.replace(tzinfo=tz)
    return dt2unix(dt, mill_type)

def str2dt(time_str, dt_form="%Y-%m-%d %H:%M"):
    return datetime.datetime.strptime(time_str, dt_form)

def truncate_minute(d):
    '''
    将时间按分钟截取
    '''
    if isinstance(d, datetime.datetime):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute)
    if isinstance(d, datetime.date):
        return datetime.datetime(d.year, d.month, d.day)
    logging.fatal(d)
    raise ValueError("parament should be datetime or date")

def truncate_date(d):
    '''
    将时间按日期截取
    '''
    return datetime.datetime(d.year, d.month, d.day)

def truncate_hour(d):
    return datetime.datetime(d.year, d.month, d.day, d.hour)

def unix2seconds(t):
    '''
       将 t 时间戳，统一转化为单位为秒
    '''
    if len(str(int(t))) == 10:
        return t
    elif len(str(int(t))) == 13:
        return float(t) / 1000

def s3_time_adj(dt, time_type):
    '''
       S3 上使用到的时间调整
       time_type 用于描述是开始时间还是结束时间: 'begin_time' or 'end_time'
       开始时间分钟小于10进行调整，结束时间分钟大于50进行调整
    '''
    if time_type == 'end_time' and dt.minute > 50:
        return truncate_hour(dt) + datetime.timedelta(hours=1)
    elif time_type == 'begin_time' and dt.minute < 10:
        return truncate_hour(dt) - datetime.timedelta(hours=1)
    else:
        return dt
