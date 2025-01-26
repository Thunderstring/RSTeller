import time
import random
from multiprocessing import Process, Event
from queue import PriorityQueue
from multiprocessing import Queue
from threading import Lock
import abc
import logging

class BaseDataProducer(Process, metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 queue: Queue, 
                 lock: Lock,
                 affair_queue:PriorityQueue, 
                 affair_lock: Lock,
                 producer_id=None,
                 tick_interval=0.01):
        
        super().__init__()
        self.queue = queue
        self.lock = lock
        self.affair_queue = affair_queue
        self.affair_lock = affair_lock
        self.producer_id = producer_id if producer_id else random.randint(1, 1000000)
        self.tick_interval = tick_interval
        self.stop_flag = Event()
        self.stop_flag.clear()
        self.logger = logging.getLogger('[DataProducer{}]'.format('' if producer_id is None else producer_id))
        self.daemon = True
        
    @abc.abstractmethod
    def init_database(self, **kwargs):
        
        raise NotImplementedError
    
    @abc.abstractmethod
    def deinit_database(self, **kwargs):
        
        raise NotImplementedError
    
    @abc.abstractmethod
    def prefetch_data(self, **kwargs):
        """implement this function to prefetch data from database, cache some data if necessary. \
            return a list of patch id for subsequent data generation.

        Raises:
            NotImplementedError
        """
        
        raise NotImplementedError
    
    @abc.abstractmethod
    def generate_task(self, data):
        
        raise NotImplementedError
    
    def check_buffer_prefetch(self, data_buffer):
        
        if len(data_buffer) < 1:
            data_buffer = self.prefetch_data()
            
        if len(data_buffer) < 1:
            self.logger.warning('No data to produce, sleeping for {} seconds...'.format(self.tick_interval))
            self.stop_flag.set()
            data_buffer = None
            time.sleep(self.tick_interval)
            
        return data_buffer

    # a wrapper function for run to ignore the KeyboardInterrupt exception
    def run(self):
        time.sleep(random.uniform(0, 120))
        try:
            self._run()
        except KeyboardInterrupt:
            # pass
            self.logger.warning('KeyboardInterrupt received, stopping...')
            self.stop()

    def _run(self):
        
        # connect to database
        self.logger.info('Connecting to database...')
        self.init_database()
        self.logger.info('Connected to database.')
        
        # start producing data
        self.logger.info('Starting to produce data...')
        self.data_buffer = []
        while not self.stop_flag.is_set():
            self.data_buffer = self.check_buffer_prefetch(self.data_buffer)
            if self.data_buffer is None:
                continue
            
            # get a task from data buffer
            try:
                task = self.generate_task(self.data_buffer.pop())
            except Exception as e:
                self.logger.error('Error occurred when generating task: {}'.format(e))
                continue
            if task is None:
                continue
            
            # put task into queue
            while not self.lock.acquire(blocking=False):
                if self.stop_flag.is_set():
                    self.logger.warning('Stop flag is set, stopping...')
                    self.deinit_database()
                    return
                time.sleep(self.tick_interval)
                
            while not self.queue.empty():
                if self.stop_flag.is_set():
                    self.logger.warning('Stop flag is set, stopping...')
                    self.deinit_database()
                    self.lock.release()
                    return
                time.sleep(self.tick_interval)
                
            self.queue.put(task)
            self.lock.release()
            
        self.logger.info('Data production stopped.')
        self.deinit_database()
                
    
    def stop(self):
        
        self.stop_flag.set()
        time.sleep(0.1)
        self.terminate()