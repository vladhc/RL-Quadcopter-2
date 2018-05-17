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
        for exp in buf.sample(2):
            self.assertTrue(exp.state != 3)

    def test_sample(self):
        buf = ReplayBuffer(3)
        buf.add(Experience(0, 0, 0, 0, False), 9.0)
        buf.add(Experience(1, 0, 0, 0, False), 1.0)
        buf.add(Experience(2, 0, 0, 0, False), 0.0)

        counter = {}
        for _ in range(10000):
            for exp in buf.sample(2):
                if exp.state not in counter:
                    counter[exp.state] = 0
                counter[exp.state] += 1

        for key, counts in counter.items():
            counter[key] = counts / 10000

        print(counter)
        np.testing.assert_approx_equal(counter[0], 0.9, significant=2)
        np.testing.assert_approx_equal(counter[1], 0.1, significant=2)
        np.testing.assert_approx_equal(counter[2], 0.0, significant=2)

if __name__ == '__main__':
    unittest.main()
