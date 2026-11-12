import logging

LINES = []


class _H(logging.Handler):
    def emit(self, record):
        LINES.append(record.getMessage())


def init_logger(name):
    lg = logging.getLogger(name)
    if not any(isinstance(h, _H) for h in lg.handlers):
        lg.addHandler(_H())
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg
