import os

def get_data_dir():
    """Returns the data directory
    """
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_path = os.path.dirname(os.path.abspath(filename))
    DATA = os.path.join(this_path, "../data")
    if not os.path.exists(DATA):
        raise ValueError(DATA + " folder doesn't exist")
    return DATA