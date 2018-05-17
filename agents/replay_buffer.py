import numpy as np
from collections import namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=100000):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
        """
        self._buffer_size = buffer_size
        self._memory = list()
        self._td_err = np.empty([0])
        self._epsilon = 1e-7

    def add(self, exp, td_err):
        """Add a new experience to memory."""
        self._memory.append(exp)
        self._td_err = np.append(self._td_err, td_err)
        # Remove elements with low td_err
        if len(self) > self._buffer_size:
            idx = np.argmin(self._td_err)
            del self._memory[idx]
            self._td_err = np.delete(self._td_err, idx)
            assert len(self._memory) == len(self._td_err)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        p = np.absolute(self._td_err) + self._epsilon
        p = p / np.sum(p)
        idx = np.random.choice(len(self), batch_size, replace=False, p=p)
        return [self._memory[i] for i in idx], idx

    def update_td_err(self, experience_indexes, td_errs):
        assert not np.any(np.isnan(td_errs))
        self._td_err[experience_indexes] = td_errs

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self._memory)

    def scrape_stats(self, stat):
        td_err = np.absolute(self._td_err)
        stat.scalar('td_err_mean', np.mean(td_err))
        stat.scalar('td_err_deviation', np.std(td_err))
