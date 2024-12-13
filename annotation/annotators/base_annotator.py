import abc

class BaseAnnotator(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self, **kwargs):
        super().__init__()
    
    @abc.abstractmethod
    def inference(self, text):
        raise NotImplementedError("inference method not implemented")