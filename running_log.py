# -*- coding: utf-8 -*-
import atexit
import datetime
import json
import os
import sys
from typing import Any, Union


def save_json(obj: Any, path: str, indent: Union[int, None] = 2) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


class RunningLog:
    def __init__(self, log_path: str):
        self.running = {
            'start': str(datetime.datetime.now()),
            'end': None,
            'argv': sys.argv,
            'parameters': {},
            'state': 'failed'
        }

        def save_running_log():
            print('saving running log to running-log.json')
            self.running['end'] = str(datetime.datetime.now())
            filename = os.path.join(log_path, 'running-log.json')
            all_running = []
            if os.path.isfile(filename):
                all_running = load_json(filename)
            all_running.append(self.running)
            save_json(all_running, filename)
        atexit.register(save_running_log)

    def set(self, key, value):
        self.running[key] = value
