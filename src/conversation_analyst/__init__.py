"""Conversation Analyst -- multimodal measurement of dyadic conversation.

The package turns per-person video recordings of a two-person conversation
into a documented table of behavioral measures: who spoke when, how quickly
each replied, what they looked at, when they nodded, smiled and laughed,
how their speech and movement tracked one another, and how all of that
changed over the course of the conversation.

Built at the Niedenthal Emotions Lab, University of Wisconsin-Madison.
The engine's historical name is convlab; user-facing surfaces say
Conversation Analyst.
"""

from __future__ import annotations

import os

# MediaPipe pulls in TensorFlow's logging on some installs; silence the
# banner before any transitive import can emit it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

__version__ = "1.1.0"
APP_NAME = "Conversation Analyst"

from conversation_analyst.config import Config
from conversation_analyst.session import PERSONS, Session, discover_sessions, iter_sessions

__all__ = [
    "__version__",
    "Config",
    "Session",
    "PERSONS",
    "discover_sessions",
    "iter_sessions",
]
