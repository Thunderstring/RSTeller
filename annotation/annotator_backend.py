from annotators import *
import annotators as anno
from pseudo_backend import PseudoBackend
from utils.jsonsocket import easy_send_message
from retry import retry
import argparse
import queue
import threading
from multiprocessing import Process, Queue, Event
import sys
import logging
import socket
import sqlite3
import os

log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)


class AnnotatorBackend(Process):
    def __init__(
        self, annotator, model_id, annotator_id, host, port, loggerHandler=None
    ):
        super().__init__()
        self.logger = logging.getLogger("[annotator]")
        self.annotator_name = annotator
        self.model_id = model_id
        self.annotator_id = annotator_id
        self.host = host
        self.port = port
        self.loggerHandler = loggerHandler
        self.exit_flag = Event()
        self.exit_flag.clear()
        if loggerHandler:
            self.logger.addHandler(loggerHandler)

    @retry(Exception, tries=3, delay=1, logger=logging.getLogger("[annotator]"))
    def send_response(self, connection, response):
        # this lock is needed to prevent multiple threads from sending responses at the same time
        # self.logger.debug(f"Sending response, try to unlock")
        # self.socket_lock.release()
        self.logger.debug(f"Sending response, lock acquiring")
        self.socket_lock.acquire()
        self.logger.debug(f"Sending response, lock acquired")
        try:
            easy_send_message(connection, response, "annotation_response_ack")
        except socket.timeout:
            raise Exception("Socket timeout")
        finally:
            self.socket_lock.release()
            self.logger.debug(f"Sending response, lock released")

    def run(self):

        try:
            self.annotator = globals()[self.annotator_name](
                model_id=self.model_id, annotator_id=self.annotator_id
            )
        except KeyError:
            self.logger.error(f"Annotator {self.annotator_name} not found")
            sys.exit(1)

        self.queue = Queue()
        self.socket_lock = threading.Lock()

        self.socket_backend = PseudoBackend(
            self.host, self.port, self.queue, self.socket_lock, self.loggerHandler
        )

        if hasattr(self.annotator, "annotator_id"):
            self.annotator_id = self.annotator.annotator_id
            self.socket_backend.annotator_id = self.annotator_id
        else:
            self.annotator_id = "-1"
            self.logger.warning(
                f"Annotator {self.annotator_name} does not have an annotator_id, using default value {self.annotator_id}"
            )

        self.socket_backend.start()
        self.logger.info(f"Annotator backend started on {self.host}:{self.port}")

        while not self.exit_flag.is_set():
            try:
                conn, message = self.queue.get(block=True, timeout=1)
                self.logger.debug(f"Received message: {message}")
                if message.get("type") == "annotation_request":
                    message.pop("type")
                    self.logger.debug("Received annotation request")
                    # TODO: verify if the annotator_id is valid
                    # TODO: verify if annotator is ready to work
                    prompt = message.pop("prompt")
                    self.socket_backend.busy = True
                    annotator_out = self.annotator.inference(prompt)
                    response = {
                        "type": "annotation_response",
                        "annotation": annotator_out,
                        "annotator_id": self.annotator_id,
                    }
                    response.update(message)
                    self.logger.debug(f"Sending annotation response: {response}")
                    try:
                        self.send_response(conn, response)
                    except Exception as e:
                        self.logger.error(f"Error while sending response: {e}")
                    self.socket_backend.busy = False

            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error while processing message: {e}")

            # time.sleep(0.01)
        self.socket_backend.stop()

    def stop(self):
        self.exit_flag.set()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--annotation-db", type=str, default="database/annotation.db",
        help="Path to the annotation database, default is \
                database/annotation.db"
    )
    argparser.add_argument(
        "--annotator",
        type=str,
        choices=anno.__ALL__,
        default="PseudoAnnotator",
        help="Name of the annotator to use, default is \
                PseudoAnnotator. Currently supported annotators are {}".format(
            anno.__ALL__
        ),
    )
    argparser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID from huggingface.co/models, default is None.",
    )
    argparser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help='Host to run the server on, default is localhost. Type \
        "network" to run on the ip from the network interface. Also \
        supports arbitrary IPv4 addresses.',
    )
    argparser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )

    args = argparser.parse_args()

    # check if the database exists
    if not os.path.exists(args.annotation_db):
        print(f"Annotation database {args.annotation_db} does not exist.")
        sys.exit(1)

    # setup the database
    conn = sqlite3.connect(args.annotation_db)
    c = conn.cursor()
    # check if model_id exists in the annotator table where ANNOTATOR=model_id
    c.execute("SELECT ID FROM annotator WHERE ANNOTATOR=?", (args.model_id,))
    result = c.fetchone()
    if result is None:
        print(f"Model ID {args.model_id} not found in the annotator table.")
        # add the model_id to the annotator table and get the auto-generated ID
        c.execute("INSERT INTO annotator (ANNOTATOR) VALUES (?)", (args.model_id,))
        args.annotator_id = c.lastrowid
        conn.commit()
        print(f"Model ID {args.model_id} added to the annotator table.")
    else:
        args.annotator_id = result[0]
        print(f"Model ID {args.model_id} found in the annotator table.")

    annotator_backend = AnnotatorBackend(
        args.annotator, args.model_id, args.annotator_id, args.host, args.port
    )
    annotator_backend.start()

    print("Annotator backend started. Press Ctrl+C to stop.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        annotator_backend.stop()
