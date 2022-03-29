
import pylibmc as mc


class mcManager(object):
    def __init__(self, host='127.0.0.1:11211'):
           self.mc = mc.Client([host], binary=True, behaviors={'tcp_nodelay':True, "ketama":True})

    def set(self, key, value):
        self.mc.set(key , value)

    def get(self, key):
        return self.mc.get(key)
    
    def __contains__(self, key):
 
        return key in self.mc