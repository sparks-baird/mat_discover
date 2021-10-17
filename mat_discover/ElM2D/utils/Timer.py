"""Timer class."""
from time import time


class Timer(object):
    """
    Simple timer class.

    https://stackoverflow.com/a/5849861/13697228
    Usage
    -----
    with Timer("description"):
        # do stuff
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Enter the timer."""
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        """Exit the timer."""
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print(("Elapsed: {}\n").format(round((time() - self.tstart), 5)))


class NoTimer(object):
    """Use in place of Timer without actually printing output."""

    def __init__(self, name):
        """Take name as argument and do nothing."""
        pass
