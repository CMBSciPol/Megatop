import time

from .logger import logger


class Timer:
    """
    Basic timer class to time different
    parts of pipeline stages.
    """

    def __init__(self):
        """
        Initialize the timers with an empty dict
        """
        self.timers = {}

    def start(self, timer_label):
        """
        Start the timer with a given label. It allows
        to time multiple nested loops using different labels.

        Parameters
        ----------
        timer_label : str
            Label of the timer.
        """
        if timer_label in self.timers:
            msg = f"Timer {timer_label} already exists."
            raise ValueError(msg)
        self.timers[timer_label] = time.time()

    def stop(self, timer_label, text_to_output=None):
        """
        Stop the timer with a given label.
        Allows to output a custom text different
        from the label.

        Parameters
        ----------
        timer_label : str
            Label of the timer.
        text_to_output : str, optional
            Text to output instead of the timer label.
            Defaults to None.
        verbose : bool, optional
            Print the output text.
            Defaults to True.
        """
        if timer_label not in self.timers:
            msg = f"Timer {timer_label} does not exist."
            raise ValueError(msg)

        dt = time.time() - self.timers[timer_label]
        self.timers.pop(timer_label)
        prefix = f"[{text_to_output}]" if text_to_output else f"[{timer_label}]"
        logger.info(f"{prefix} took {dt:.02f}s to process.")
