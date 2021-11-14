from .vec_env import VecEnvWrapper
# PLATOON: just put resultswriter code here since it's from an earlier version of stable baselines
# from stable_baselines3.common.monitor import ResultsWriter
# For ResultsWriter:
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time
from collections import deque

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def reset_agent(self):
        obs = self.venv.reset_agent()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def reset_random(self):
        obs = self.venv.reset_random()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info

        return obs, rews, dones, newinfos

class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class

    :param filename: the location to save a log file, can be None for no log
    :param header: the header dictionary object of the saved csv
    :param reset_keywords: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[Dict[str, Union[float, str]]] = None,
        extra_keys: Tuple[str, ...] = (),
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        self.file_handler = open(filename, "wt")
        self.file_handler.write("#%s\n" % json.dumps(header))
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + extra_keys)
        self.logger.writeheader()
        self.file_handler.flush()


    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:
        """
        Close the file handler

        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()


    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()
