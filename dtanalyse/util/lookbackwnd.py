LBW_TYPE_TIME = "time"
LBW_TYPE_VOLUME = "volume"

LBW_TIME_TICK = 1e9    #1秒对应的纳秒时间戳
LBW_VOLUME_TICK = 200000

LBW_TIME_STR = "_t"
LBW_VOLUME_STR = "_v"


class LookBackWindow(object):

    def __init__(self, 
                 increament,    # ts in ms for second for "time"; usdt for "volume"
                 max_scale, 
                 type=LBW_TYPE_TIME,    # or "volume"
                 ts_event=0
                ):
        self.type = type
        self.increament = increament
        self.max_scale = max_scale
        self.container = [[-1, -1, 0]]
        self.cur_scale = 0
        self.last_scale = 0
        self.init_value = 0
        if type == LBW_TYPE_TIME:
            self.init_value = ts_event
        self.last_value = 0
        self.is_rolling = False
        self.counter = 0
        self.rolling_wnd = 0

    # for every update——idx: index in the data queue; value_for_scale: value used for look back
    def update(self, idx, value_for_scale):
        if self.type == LBW_TYPE_TIME:
            self._update_for_time(idx, value_for_scale)
        else:
            self._update_for_volume(idx, value_for_scale)
        self._adjust()
    
    def _adjust(self):
        if self.last_scale - self.cur_scale > self.max_scale:
            _roll_wnd = self.last_scale - self.max_scale
            self.cur_scale += self.last_scale - self.max_scale
            self.init_value += _roll_wnd * self.increament

    def _update_for_time(self, idx, value_for_scale):
        # _value = value_for_scale - self.init_value
        # cur_value = self.container[-1][2]
        # new_scale = int(_value) // int(self.increament)
        new_scale = int(value_for_scale) // int(self.increament) - int(self.init_value) // int(self.increament)
        # print(new_scale, self.last_scale, self.cur_scale)

        new_value = value_for_scale
        self.is_rolling = False
        self.rolling_wnd = 0
        # if self.cur_scale + new_scale > self.last_scale + 1:
        # print(f"warning: time gap: {self.cur_scale} {self.last_scale} {new_scale} -- {value_for_scale} {self.init_value} {self.increament}")
        while self.last_scale < self.cur_scale + new_scale:
            self.container.append([-1, -1, 0])
            self.last_scale += 1
            self.rolling_wnd += 1
            self.is_rolling = True
            self.counter += 1
            # print(f"[{self.type}] data rolling: {self.cur_scale} + {new_scale}({value_for_scale}) >= {self.last_scale}")
                
        if self.container[-1][0] == -1:
            self.container[-1][0] = idx
        self.container[-1][1] = idx
        
        self.container[-1][2] = new_value
        self.last_value = self.container[-1][2]
        # print(f"[{self.type}] updated: {self.counter} {self.rolling_wnd}")
        
    def _update_for_volume(self, idx, value_for_scale):
        # todo: check for the condition of rolling
        _value = self.last_value + value_for_scale
        if 0 < _value < self.increament:
            _value = self.increament
        new_scale = int(_value) // int(self.increament)
        # print(new_scale, self.last_scale, self.cur_scale)

        cur_value = self.container[-1][2]
        new_value = value_for_scale
        while new_scale > 0 and self.last_scale - self.cur_scale <= self.max_scale:
            if cur_value > 0:
                self.container[-1][1] = idx
                self.container[-1][2] = self.increament
            new_scale -= 1
            new_value -= (self.increament - cur_value)
            self.container.append([idx, idx, 0])
            self.last_scale += 1
            self.rolling_wnd += 1
            self.is_rolling = True
            self.counter += 1
            _cur_value = 0
            # print(f"[{self.type}] data rolling: {self.cur_scale} + {new_scale}({value_for_scale}) >= {self.last_scale}; {new_value}")
                
        if self.container[-1][0] == -1:
            self.container[-1][0] = idx
        self.container[-1][1] = idx

        # todo：需确认：最后一条数值不应超过满仓？
        if new_value > self.increament:
            new_value = self.increament
        self.container[-1][2] += new_value
        self.last_value = self.container[-1][2]
        # print(f"[{self.type}] updated: {self.counter} {self.rolling_wnd}")
        
    # should be executed after process
    def rolling(self):
        if not self.is_rolling:
            return

        self.counter += 1
        if self.last_scale - self.cur_scale > self.max_scale:
            delta = self.last_scale - self.max_scale - self.cur_scale
            self.init_value += delta * self.increament
            self.cur_scale += delta
        if self.cur_scale > self.max_scale + 1:
            _roll_wnd = self.cur_scale - self.max_scale
            del self.container[:_roll_wnd]
            self.last_scale -= _roll_wnd
            self.cur_scale -= _roll_wnd
            if self.type == "time":
                # self.init_value -= _roll_wnd * self.increament
                # print(f'lbw rolling: {_roll_wnd} {self.init_value}')
                pass
            else:
                self.init_value = self.cur_scale * self.increament

    def size(self):
        # todo: check for if +1 is needed
        return self.last_scale - self.cur_scale + 1
        
    def get_scale_value(self):
        return (self.last_scale - self.cur_scale) * self.increament

    def get_attr(self):
        _attr = self.__dict__.copy()
        _attr.pop('container', None)
        return _attr

    def __repr__(self):
        return str(self.__dict__)


if __name__ == "__main__":    
    time_wnd = LookBackWindow(2000, 10, "time")

    for i, t in enumerate(range(0, 100000, 2500)):
        # if i % 2 == 0:
        #     continue
            
        last = time_wnd.last_scale
        time_wnd.update(i, t)
        print(f'{time_wnd.counter}: {i} {t}: {time_wnd.counter * time_wnd.increament} - {(time_wnd.counter+1) * time_wnd.increament  - 1}')
        size = len(time_wnd.container)
        print(f'{time_wnd.counter}: idx {i}, value {t}: {time_wnd.container}, value:{time_wnd.init_value}/{time_wnd.get_scale_value()} scale:{time_wnd.cur_scale}/{time_wnd.last_scale}(previous last scale: {last}, container size: {size})')

        time_wnd.rolling()



    