"""Message Notification Router package.

Production routing intentionally never loads sample_messages.csv.  The sample
file is used only by code/evaluation for development regression checks.
"""

from .engine import Router

__all__ = ["Router"]
