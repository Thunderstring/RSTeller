import abc

class BaseAnnotator(metaclass=abc.ABCMeta):
    
    _VALID_MODELS = []
    
    @abc.abstractmethod
    def __init__(self, **kwargs):
        super().__init__()
        self.model_id = kwargs.get("model_id", None)
        # check if model is valid
        if self.model_id not in self._VALID_MODELS:
            raise ValueError(f"Invalid model: {self.model_id} for class {self.__class__.__name__}. Valid models are: {self._VALID_MODELS}")
    
        self.annotator_id = kwargs.get("annotator_id", None)
    
    @abc.abstractmethod
    def inference(self, text):
        raise NotImplementedError("inference method not implemented")