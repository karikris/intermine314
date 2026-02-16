from intermine314.util import ReadableException


class ModelError(ReadableException):
    pass


class PathParseError(ModelError):
    pass


class ModelParseError(ModelError):
    def __init__(self, message, source, cause=None):
        self.source = source
        super(ModelParseError, self).__init__(message, cause)

    def __str__(self):
        base = repr(self.message) + ":" + repr(self.source)
        if self.cause is None:
            return base
        else:
            return base + repr(self.cause)
