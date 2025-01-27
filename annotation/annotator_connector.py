"""
TODO: update the backend from simple socket to fastapi with http protocol
"""

from queue import Queue
from threading import Thread, Lock
import socket
import logging
import time
import queue
from retry import retry
from utils.jsonsocket import easy_handshake, easy_disconnect, send_message, receive_message
class AnnotatorConnector(Thread):
    
    _SOCKET_TIMEOUT = 60
    
    def __init__(self, host:str, port:int, 
                       in_queue:Queue, 
                       in_queue_lock:Lock, 
                       out_queue:Queue, 
                       out_queue_lock:Lock,
                       affair_queue:queue.PriorityQueue,
                       affair_queue_lock:Lock,
                       tick_interval:float=0.01):
        super(AnnotatorConnector, self).__init__()
        # TODO: use a dict instead of a queue to store the tasks, so that we can easily remove a task by its ID
        self.host = host
        self.port = port
        self.in_queue = in_queue
        self.in_queue_lock = in_queue_lock
        self.out_queue = out_queue
        self.out_queue_lock = out_queue_lock
        self.affair_queue = affair_queue
        self.affair_queue_lock = affair_queue_lock
        self.server_lock = Lock()
        self.logger = logging.getLogger(f'[{self.host}:{self.port}]')
        self.tick_interval = tick_interval
        self.annotator_id = None
        
        self.exit_flag = False
        self.daemon = True
        self.busy = False
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
        except Exception as e:
            self.logger.error(f'Error in connecting to server: {e}')
            self.close()
            return
        
        response = self.handshake()
        if response is None:
            self.logger.error('Handshake failed')
            self.close()
            return
        
        self.annotator_id = response.get('annotator_id')
        self.busy = response.get('busy')
        
        self.logger.info(f'Annotator ID: {self.annotator_id}, busy status: {self.busy}')

    def get_annotator_id(self):
        if self.annotator_id is None:
            self.logger.error('Annotator ID is None')
            return -1
        return self.annotator_id

    def check_handshake_response(self, response:dict):
        
        if not isinstance(response, dict):
            self.logger.error('Handshake response is not a dictionary')
            return None
        
        if response.get('type') != 'handshake_response':
            self.logger.error('Handshake response type is not handshake_response')
            return None
        
        if response.get('annotator_id') is None:
            self.logger.error('Annotator ID is None in handshake response')
            return None
        
        if response.get('busy') is None:
            self.logger.error('Busy status is None in handshake response')
            return None
        
        return response
    
    def check_task_request(self, task:dict):
        
        if not isinstance(task, dict):
            self.logger.error('Task is not a dictionary')
            return None
        
        if task.get('type') != 'annotation_request':
            self.logger.error('Task type is not annotation_request')
            return None
        
        annotator_id =  task.get('annotator_id')
        if annotator_id is not None and annotator_id != self.annotator_id:
            self.logger.error('Annotator ID in task does not match the current annotator ID')
            return None
        
        if task.get('prompt_id') is None:
            self.logger.error('Prompt ID is None in task')
            return None
        
        if task.get('patch_id') is None:
            self.logger.error('Patch ID is None in task')
            return None
        
        if task.get('prompt') is None:
            self.logger.error('Prompt is None in task')
            return None
        
        return task
    
    
    @retry(tries=3, delay=1, backoff=2)
    def _handshake(self):
        response = easy_handshake(self.socket)
        return response
    
    def handshake(self):
        try:
            response = self._handshake()
        except Exception as e:
            self.logger.error('Error in handshake with server: ({}:{}), {}'.format(self.host, self.port, e))
            self.close()
            return
        
        response = self.check_handshake_response(response)
        if response is None:
            self.close()
            return
        
        self.logger.info(f'Handshake with server successful: ({self.host}:{self.port})')
        
        return response
    
    def post_task(self, task:dict):
        
        self.server_lock.acquire()
        try:
            send_message(self.socket, task)
        except Exception as e:
            self.logger.error(f'Error in sending task to server: {e}')
            self.server_lock.release()
            return None
        self.busy = True
        self.socket.settimeout(self._SOCKET_TIMEOUT)
        try:
            response = receive_message(self.socket)
        except Exception as e:
            self.logger.error(f'Error in receiving response from server: {e}')
            self.server_lock.release()
            self.socket.settimeout(None)
            # next time, we will first handshake with the server to check if it is busy
            return None
        
        self.socket.settimeout(None)   
        self.busy = False
        response_type = response.get('type')
        
        if response_type != 'annotation_response':
            self.logger.error(f'Response type is not annotation_response: {response_type}')
            return None
        send_message(self.socket, {'type': 'annotation_response_ack'})
        self.logger.debug('acknowledged the response from the server')
        self.server_lock.release()
   
        return response
            
    def run(self):
        while not self.exit_flag:
            # first check if the server is busy
            if self.busy:
                # delay a long period of time and try again to make sure 
                # the server has finished processing the previous task
                self.logger.warning('Server is busy, waiting for 20 seconds')
                response = self.handshake()
                self.busy = response['busy']
                time.sleep(20)
                continue
            
            # check if there is any task in the input queue
            if not self.in_queue_lock.acquire(blocking=False):
                time.sleep(self.tick_interval)
                continue
            
            # get the task from the input queue
            try:
                task = self.in_queue.get(block=False)
            except queue.Empty:
                self.in_queue_lock.release()
                time.sleep(self.tick_interval)
                continue
            self.in_queue_lock.release()
            
            # check if the task is a valid task
            task = self.check_task_request(task)
            if task is None:
                time.sleep(self.tick_interval)
                continue
                
            # send the task to the server
            response = self.post_task(task)
            if response is None:
                time.sleep(self.tick_interval)
                continue
            
            self.out_queue_lock.acquire()
            self.out_queue.put(response)
            self.out_queue_lock.release()
            
            self.logger.info(f'Received response from server: {response["type"]}')
            time.sleep(self.tick_interval)
            

            
    def close(self):
        """Disconnect from the server and release the resources. Send a message to the parent process to release the resources.
        """
        self.exit_flag = True
        self.server_lock.acquire()
        easy_disconnect(self.socket)
        self.affair_queue_lock.acquire()
        self.affair_queue.put((0, {'type': 'annotator_connector_closed', 
                               'host': self.host, 
                               'port': self.port}))
        self.affair_queue_lock.release()
        self.socket.close()
        self.server_lock.release()
        self.logger.info('Annotator connector closed')


# if __name__ == '__main__':
    
#     test_host = 'localhost'
#     test_port = 8000
    
#     task_cnt = 0
    
#     test_tasks = [
#         {'type': 'annotation_request', 'annotator_id': -1, 'prompt_id': 1, 'patch_id': 1, 'prompt': 'What is the name of this building?'},
#         {'type': 'annotation_request', 'annotator_id': -1, 'prompt_id': 2, 'patch_id': 2, 'prompt': 'What is the name of this building?'},
#         {'type': 'annotation_request', 'annotator_id': -1, 'prompt_id': 3, 'patch_id': 3, 'prompt': 'What is the name of this building?'}]
    
#     # create the input and output queues
#     in_queue = Queue()
#     in_queue_lock = Lock()
#     out_queue = Queue()
#     out_queue_lock = Lock()
#     affair_queue = Queue()
#     affair_queue_lock = Lock()
    
#     # create the annotator connector
#     annotator_connector = AnnotatorConnector(test_host, test_port, in_queue, in_queue_lock, out_queue, out_queue_lock, affair_queue, affair_queue_lock)
#     annotator_connector.start()
#     # send the tasks to the annotator connector
#     for task in test_tasks:
#         in_queue_lock.acquire()
#         in_queue.put(task)
#         in_queue_lock.release()
        
#     # wait for the responses
#     while True:
#         if task_cnt == len(test_tasks):
#             annotator_connector.close()
        
#         out_queue_lock.acquire()
#         if out_queue.empty():
#             out_queue_lock.release()
#             affair_queue_lock.acquire()
#             if affair_queue.empty():
#                 affair_queue_lock.release()
#                 time.sleep(0.1)
#                 continue
#             affair = affair_queue.get()
#             affair_queue_lock.release()
#             if affair['type'] == 'annotator_connector_closed':
#                 break
#         else:
#             response = out_queue.get()
#             out_queue_lock.release()
#             task_cnt += 1
#             print(response)
#             time.sleep(0.1)
#             continue
#     annotator_connector.join()
    
#     print('done')
    
    
