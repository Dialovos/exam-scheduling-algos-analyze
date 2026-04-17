"""Atomic JSON checkpoint for resumable tuning runs.

Writes to ``<path>.tmp`` then ``os.replace`` — so a SIGKILL mid-write
leaves the previous checkpoint intact rather than a half-written file.
"""
from __future__ import annotations

import json
import os


class Checkpoint:
    def __init__(self, path):
        self.path = path

    def save(self, state):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def load(self):
        if not os.path.isfile(self.path):
            return None
        with open(self.path) as f:
            return json.load(f)
