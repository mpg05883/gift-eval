from datetime import datetime
from zoneinfo import ZoneInfo
import time
import torch.distributed as dist


def get_timestamp() -> str:
    """
    Gets the current time in Pacific Time formatted as
    `MM DD, YYYY HH:MM:SSAM/PM`.
    """
    time_zone = "America/Los_Angeles"
    timestamp = datetime.now(ZoneInfo(time_zone))

    format = "%b %d, %Y %I:%M:%S%p"
    return timestamp.strftime(format)

def is_rank_zero() -> bool:
    """
    Returns True if the process is the rank zero process. Always returns True
    for non-distributed setups.

    Returns:
        bool: True if the process is the rank zero process or the setup is
        non-distributed. Else, False.
    """
    return True if not dist.is_initialized() else dist.get_rank() == 0

def format_elapsed_time(start: float, end: float) -> str:
    """
    Formats the elapsed time between two timestamps.
    
    The elapsed time is automatically converted to seconds, minutes, hours, or
    days, depending on the duration.

    Args:
        start (float): Start time in seconds.
        end (float): End time in seconds.

    Returns:
        str: Formatted elapsed time as a string with appropriate units.
    """
    seconds = abs(end - start)
    
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86_400:
        return f"{seconds / 3600:.2f} hours"
    else:
        return f"{seconds / 86400:.2f} days"