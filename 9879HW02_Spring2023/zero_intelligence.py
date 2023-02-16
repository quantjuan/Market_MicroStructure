import numpy as np
import matplotlib.pyplot as plt
import random

# the ZI class

class ZI:
    '''
    This defines the ZI (zero-intelligence) class
    Adapted from the R code by Jim Gatheral
    '''
    
    # Book setup
    L = 30 # default, number of price levels to be included in iterations
    LL = 1000 # default, total number of levels in buy and sell books
    # initial sizes on the bid/buy side 
    buy_size = [5 for x in range(LL - 8)] + [5, 4, 4, 3, 3, 2, 2, 1] + [0 for x in range(LL + 1)]
    buy_size = np.array(buy_size)
    # initial sizes on the ask/sell side
    sell_size = [0 for x in range(LL + 1)] + [1, 2, 2, 3, 3, 4, 4, 5] + [5 for x in range(LL - 8)]    
    sell_size = np.array(sell_size)
    
    def __init__(self, lmda, mu, nu, L=30, LL=1000, buy_size=buy_size, sell_size=sell_size):
        self.lmda = lmda
        self.mu = mu
        self.nu = nu
        self.L = L
        self.LL = LL
        self.price = np.array([x for x in range(-LL, LL + 1)])
        self.buy_size = buy_size.copy()
        self.sell_size = sell_size.copy()
        
        # dict for order book
        self.book = {'price': self.price,
                     'buy_size': self.buy_size,
                     'sell_size': self.sell_size
                    }
        # market activities
        self.activities = (self.limit_buy, self.limit_sell, self.cancel_buy, 
                  self.cancel_sell, self.market_buy, self.market_sell)
    
    def book_shape(self, band=10):
        buy = self.book['buy_size'][(self.mid_posn() - band):self.mid_posn()]
        sell = self.book['sell_size'][self.mid_posn():(self.mid_posn() + band + 1)]
        return np.concatenate([buy, sell])
    
    def book_shape_signed(self, band=10):
        buy = self.book['buy_size'][(self.mid_posn() - band):self.mid_posn()]
        sell = self.book['sell_size'][self.mid_posn():(self.mid_posn() + band + 1)]
        return np.concatenate([-buy, sell])
    
    def book_plot(self, band=10, signed=False):
        plt.figure(figsize=(8, 4))
        if signed is True:
            buy = self.book['buy_size'][(self.mid_posn() - band):self.mid_posn() + 1]
            sell = self.book['sell_size'][self.mid_posn() + 1:(self.mid_posn() + band + 1)]
            plt.plot([x for x in range(-band, 1)], -buy, 'ro:')
            plt.plot([0, 1], [0, sell[0]], 'b:')
            plt.plot([x for x in range(1, band + 1)], sell, 'bo:')
            plt.hlines(y = 0, xmin=-band, xmax=band, linestyle='dashed', linewidth=1)
            plt.title('Signed limit order book')
            plt.xlabel('Relative price')
            plt.ylabel('Quantity')
            plt.show()
            return None
        
        plt.plot([x for x in range(-band, band + 1)], self.book_shape(band), 'ro:')
        plt.xlabel('Relative price')
        plt.ylabel('Quantity')
        plt.title('Limit order book')
        plt.show()
        
    # functions for finding best quotes, mid, and spread
    def best_offer(self):
        return self.book['price'][self.book['sell_size'] > 0].min()
        
    def best_bid(self):
        return self.book['price'][self.book['buy_size'] > 0].max()
        
    def spread(self):
        return self.best_offer() - self.best_bid()
        
    def mid(self):
        return (self.best_offer() + self.best_bid())/2
    
    # functions for finding positions indices
    def bid_posn(self):
        return len(self.book['buy_size'][self.book['price'] <= self.best_bid()]) - 1
#         return len(self.book['buy_size'][self.book['price'] <= self.best_bid()]) - 1

    def ask_posn(self):
        return len(self.book['sell_size']) - len(self.book['sell_size'][self.book['price'] >= self.best_offer()])
#        return len(self.book['sell_size'][self.book['price'] >= self.best_offer()])
#        return len(self.book['sell_size'][self.book['price'] <= self.best_offer()])
    
    def mid_posn(self):
        return int((self.bid_posn() + self.ask_posn())/2)
        
    def pick(self, m):
        return random.randint(1, m)
    
    def go(self):
        pass
    
    # market activity functions
    def limit_buy(self, price=None):
        if price is None:
            prx = self.best_offer() - self.pick(self.L)
        else:
            prx = price
            if prx >= self.best_offer():
                print(f'Limit buy order at price {prx} greater than best offer at {self.best_offer()}')
                return None
        self.book['buy_size'][self.book['price']==prx] += 1
        return ['LB', prx]
    
    def limit_sell(self, price=None):
        if price is None:
            prx = self.best_bid() + self.pick(self.L)
        else:
            prx = price
            if prx <= self.best_bid():
                print(f'Limit sell order at price {prx} smaller than best bid at {self.best_bid()}')
                return None
        self.book['sell_size'][self.book['price']==prx] += 1
        return ['LS', prx]
        
    def market_buy(self):
        prx = self.best_offer()
        self.book['sell_size'][self.book['price']==prx] -= 1
        return ['MB', prx]
        
    def market_sell(self):
        prx = self.best_bid()
        self.book['buy_size'][self.book['price']==prx] -= 1
        return ['MS', prx]
    
    # function for finding the number of cancelable buy orders 
    def n_cb(self):
        return self.book['buy_size'][self.book['price'] >= self.best_offer() - self.L].sum()
    
    # function for finding the number of cancelable sell orders 
    def n_cs(self):
        return self.book['sell_size'][self.book['price'] <= self.best_bid() + self.L].sum()
        
    def cancel_buy(self, price=None):
        q = self.pick(self.n_cb())
        tmp = [x for x in self.book['buy_size']]
        tmp.reverse()
        tmp = np.array(tmp).cumsum()
        posn = len(tmp[tmp >= q]) - 1
        prx = self.book['price'][posn]
        if price is not None:
            prx = price
        self.book['buy_size'][posn] -= 1
        return ['CB', prx]
        
    def cancel_sell(self, price=None):
        q = self.pick(self.n_cs())
        tmp = np.cumsum(self.book['sell_size'])
        posn = len(tmp[tmp < q])
        prx = self.book['price'][posn]
        if price is not None:
            prx = price
        self.book['sell_size'][posn] -= 1
        return ['CS', prx]
    
    def generate_events(self, n_events=1, band=20, logging=False):
        lb_prob = self.lmda*self.L
        ls_prob = self.lmda*self.L
        mb_prob = self.mu/2
        ms_prob = self.mu/2
        
        # function that determines the event probabilities
        def probs():
            cb_prob = self.nu*self.n_cb()
            cs_prob = self.nu*self.n_cs()
            p = np.array([lb_prob, ls_prob, cb_prob, cs_prob, mb_prob, ms_prob])
            return p/p.sum()

        avg_book_shape = np.zeros(2*band + 1)

        if logging is True:
            event_log = []
            for i in range(n_events):
                p = probs()
                act = np.random.choice(self.activities, p=p)()
                event_log.append(act)
                avg_book_shape += self.book_shape(band)/n_events

            return event_log, avg_book_shape

        for i in range(n_events):
            p = probs()
            np.random.choice(self.activities, p=p)()
            avg_book_shape += self.book_shape(band)/n_events

        return avg_book_shape