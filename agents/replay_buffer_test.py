import unittest
import numpy as np

from replay_buffer import ReplayBuffer, Experience

class ReplayBufferTest(unittest.TestCase):

    def test_add(self):
        buf = ReplayBuffer(2)
        buf.add(Experience(0, 0, 0, 0, False), 0.5)
        buf.add(Experience(1, 0, 0, 0, False), 0.2)
        buf.add(Experience(3, 0, 0, 0, False), 0.1)

        self.assertEqual(len(buf), 2)
        # Check that the state with td_err 0.1 wasn't added to the ReplayBuffer
        exps, _ = buf.sample(2)
        for exp in exps:
            self.assertTrue(exp.state != 3)

    def test_sample(self):
        buf = ReplayBuffer(3)
        buf.add(Experience(0, 0, 0, 0, False), 9.0)
        buf.add(Experience(1, 0, 0, 0, False), 1.0)
        buf.add(Experience(2, 0, 0, 0, False), 0.0)

        # Calculate count ratio of sampled elements
        counter = {
                0: 0.0,
                1: 0.0,
                2: 0.0,
                }
        for _ in range(10000):
            exps, _ = buf.sample(1)
            exp = exps[0]
            counter[exp.state] += 1
        for key, counts in counter.items():
            counter[key] = round(counts / 10000, 1)

        np.testing.assert_approx_equal(counter[0], 0.9, significant=1)
        np.testing.assert_approx_equal(counter[1], 0.1, significant=2)
        np.testing.assert_approx_equal(counter[2], 0.0, significant=2)

    def test_update_td_err(self):
        buf = ReplayBuffer(3)
        buf.add(Experience(0, 0, 0, 0, False), 9.99)
        buf.add(Experience(1, 0, 0, 0, False), 0.01)
        buf.add(Experience(2, 0, 0, 0, False), 0.00)

        # Wait till appropriate combination will be sampled
        done = False
        max_retries = 100
        retries = 0
        while not done and retries < max_retries:
            exp, idx = buf.sample(2)
            done = (idx[0] == 0) and (idx[1] == 1)

        self.assertTrue(done) # Have we managed to sample appropriate combination?

        buf.update_td_err(idx, [0, 9.0])

        for _ in range(10):
            exp, idx = buf.sample(1)
            self.assertEqual(idx[0], 1)


if __name__ == '__main__':
    unittest.main()
