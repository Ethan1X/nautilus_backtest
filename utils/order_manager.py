import logging
from collections import defaultdict


OPENED = 'OPENED'
CANCELING = 'CANCELING'
EPSILON = 1e-8

class OrderManager:

    def __init__(self) -> None:
        # key: orderid
        # self.orders = defaultdict({'price': None, 'side': None, 'amount': None, 'status': None})
        self.orders = defaultdict(dict)
    
    def add_order(self, price, side, amount, order_id):
        self.orders[order_id] = {'price': price, 'side': side, 'amount': amount, 'status': OPENED, 'order_id': order_id}
        return
    
    def update_order_status(self, order_id, action):
        if action == 'cancel':
            self.orders[order_id]['status'] = CANCELING
        else:
            logging.error(f"遇到未知order 动作: {action}")
        return
    
    def del_order(self, order_id):
        if order_id in self.orders:
            del self.orders[order_id]
        return

    def get_not_stoped_order_ids(self, side):
        '''
            获取当前挂单中的订单ID
        '''
        return [order_info['order_id'] for order_info in self.orders.values()
                if order_info['side'] == side]

    def get_open_order_amount(self, side):
        '''
            获取挂单中的order amount
        '''
        return sum([order_info['amount'] for order_info in self.orders.values()
                    if order_info['side'] == side and order_info['status'] == OPENED])

    def get_open_order_info(self, side):
        '''
            获取挂单中的order info
        '''
        return [order_info for order_info in self.orders.values()
                    if order_info['side'] == side and order_info['status'] == OPENED]


    def get_cancel_order_ids(self, side, price):
        '''
            按照方向和价格获取需要取消的订单列表
        '''
        return [order_info['order_id'] for order_info in self.orders.values()
                    if order_info['side'] == side and order_info['status'] == OPENED and abs(order_info['price'] - price) > EPSILON]

