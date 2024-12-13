from .base_annotator import BaseAnnotator
import time

class PseudoAnnotator(BaseAnnotator):
    def __init__(self):
        self.annotator_id = -1

    def inference(self, data):
        # simulate the time cost of inference
        time.sleep(3)
        return data