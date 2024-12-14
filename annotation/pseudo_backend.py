"""Server-side pseudo backend for testing purposes.
"""
import socket
import argparse
from utils.jsonsocket import send_message, receive_message
import logging
import threading
import queue
import time

log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
 

class PseudoBackend(threading.Thread):
    def __init__(self, host, port, queue: queue.Queue, socket_lock: threading.Lock, loggerHandler=None):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        
        self.busy = False
        self.queue = queue
        self.socket_lock = socket_lock
        
        self.exit_flag = False
        
        self.logger = logging.getLogger("[socket]")

        if loggerHandler is not None:
            self.logger.addHandler(loggerHandler)

        self.logger.info(f'PseudoBackend listening on {self.host}:{self.port}')

    
    def is_handshake(self, conn, data, response=True):
        # if the data from the client is a handshake request, send a response.
        # The handshake request is a JSON object with a "type" field set to "handshake".
        # The reponse is a JSON object with a "type" field set to "handshake_response", 
        # and a "annotator_id" field set to id of the llm backend inherit from the PseudoBackend class.
        # There is also a "busy" field set to False, indicating that the backend is not busy.
        if data.get('type') == 'handshake':
            if response:
                response_data = {
                    'type': 'handshake_response',
                    'annotator_id': -1,
                    'busy': self.busy
                }
                if hasattr(self, 'annotator_id'):
                    response_data['annotator_id'] = self.annotator_id
                send_message(conn, response_data)
            return True
        return False
    
    def is_disconnect(self, conn, data, response=True):
        
        if data == None:
            return True
        
        if data.get('type') == 'disconnect':
            if response:
                response_data = {
                    'type': 'disconnect_response',
                    'annotator_id': -1,
                    'busy': self.busy
                }
                if hasattr(self, 'annotator_id'):
                    response_data['annotator_id'] = self.annotator_id
                send_message(conn, response_data)
            return True
        return False
    
    def run(self):
        self.logger.info('PseudoBackend started')
        self.socket.settimeout(1)
        self.socket.listen()
        while True and not self.exit_flag:
            try:
                conn, addr = self.socket.accept()
            except socket.timeout:
                continue
            self.logger.info('Connected by {}'.format(addr))
            while True and not self.exit_flag:
                # self.logger.debug('exits: {}'.format(self.exit_flag))
                try:
                    if not self.socket_lock.acquire(blocking=False):
                        self.logger.debug('socket lock not acquired')
                        time.sleep(0.1)
                        continue
                    else:
                        self.logger.debug('socket lock acquired')
                    self.logger.debug('starting receive_message')
                    # FIXME: the conn is set to blocking mode after the annotator sends a response, 
                    # which causes the socket to block until the response is received. 
                    conn.settimeout(0.1)
                    in_data = receive_message(conn)
                    self.logger.debug('finished receive_message')
                    self.socket_lock.release()
                    self.logger.debug(f'Received message: {in_data}')
                except socket.timeout:
                    self.socket_lock.release()
                    time.sleep(0.1)
                    continue
                
                except Exception as e:
                    self.logger.error(f'Error: {e}')
                    self.socket_lock.release()
                    time.sleep(0.1)
                    continue
                
                if self.is_disconnect(conn, in_data):
                    self.logger.info('Disconnect received')
                    break
                
                if self.is_handshake(conn, in_data):
                    self.logger.info('Handshake received')
                    time.sleep(0.1)
                    continue
                
                self.queue.put((conn, in_data))
                time.sleep(0.1)
            
            self.logger.debug('Closing connection...')
            conn.close()
            self.logger.info('Connection closed')
        self.logger.debug('Closing socket...')
        self.socket.close()
        self.logger.info('PseudoBackend stopped')
    
    def stop(self):
        self.exit_flag = True
        self.join()


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', type=str, default='localhost', help='Host IP address')
    argparser.add_argument('--port', type=int, default=8000, help='Port number')
    argparser.add_argument('--buffer_size', type=int, default=1024, help='Buffer size')
    args = argparser.parse_args()
    
    backend = PseudoBackend(args.host, args.port)
    backend.run()