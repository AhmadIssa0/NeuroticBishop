import numpy as np


class BinPredictor:

    def __init__(self, max_mate_range=40):
        self.max_mate_range = max_mate_range
        # Create symmetric bins around 0
        negative_bins = np.array([
            -10000.0, -5948.0, -2502.0, -1780.0, -1504.0, -1324.0, -1191.0, -1089.0, -1008.0, -938.0,
            -877.0, -822.0, -779.0, -741.0, -711.0, -688.0, -666.0, -644.0, -622.0, -594.0, -563.0, -533.0, -506.0,
            -481.0, -457.0, -434.0, -414.0, -394.0, -375.0, -356.0, -338.0, -320.0, -303.0, -287.0, -271.0, -257.0,
            -243.0, -229.0, -216.0, -204.0, -192.0, -181.0, -170.0, -160.0, -151.0, -142.0, -134.0, -127.0, -120.0,
            -113.0, -106.0, -101.0, -95.0, -89.0, -84.0, -79.0, -75.0, -70.0, -65.0, -61.0, -57.0, -53.0, -50.0,
            -46.0, -42.0, -39.0, -36.0, -33.0, -30.0, -27.0, -25.0, -22.0, -19.0, -17.0, -14.0, -12.0, -10.0, -8.0,
            -6.0, -3.0
        ])
        # Mirror the negative bins to create positive bins (excluding the extremes and 0)
        positive_bins = -negative_bins[::-1][1:]  # Reverse and negate, skip the most negative
        self.bin_ticks = np.concatenate([negative_bins, [0.0], positive_bins])
        self.total_num_bins = len(self.bin_ticks) + 2 + 2 * max_mate_range + 1

    def to_bin_index(self, cp_eval: int, mate: int) -> int:
        """Converts a cp_eval and a mate to a bin index. If mate is non-zero ignores cp_eval. The order of the bins
        is as follows:

        #-1, #-2, ..., #-max_mate_range, #<-max_mate_range, cp_eval < min, [bin_ticks[0], bin_ticks[1]), ...,
        [bin_ticks[1], bin_ticks[2]), ..., [bin_ticks[-2], bin_ticks[-1]), cp_eval >= max, #>max_mate_range,
        #max_mate_range, ..., #2, #1.
        """
        if mate != 0:
            if mate < -self.max_mate_range:
                return self.max_mate_range
            elif mate > self.max_mate_range:
                return self.total_num_bins - self.max_mate_range - 1
            elif mate < 0:
                return -mate - 1
            else:
                return self.total_num_bins - mate
        else:
            # Not a mate!
            idx = np.searchsorted(self.bin_ticks, cp_eval, 'right')  # 'right' means [a, b)
            return idx + self.max_mate_range + 1

    def bin_index_to_description(self, idx):
        assert 0 <= idx < self.total_num_bins
        if idx < self.max_mate_range:
            # Mates in fewer moves, negative side
            return "Mate", -(idx + 1), -(idx + 1)
        elif idx == self.max_mate_range:
            # Mate in less than -max_mate_range
            return "Mate", -np.inf, -self.max_mate_range - 1
        elif idx == self.max_mate_range + 1:
            # Eval less than the first bin edge
            return "Eval", -np.inf, self.bin_ticks[0]
        elif idx < self.total_num_bins - self.max_mate_range - 2:
            # Regular eval bins
            bin_start_idx = idx - (self.max_mate_range + 2)
            print('bin_start_idx', bin_start_idx)
            bin_start = self.bin_ticks[bin_start_idx]
            bin_end = self.bin_ticks[bin_start_idx + 1]
            return "Eval", bin_start, bin_end
        elif idx == self.total_num_bins - self.max_mate_range - 1:
            return "Mate", self.max_mate_range + 1, np.inf
        elif idx == self.total_num_bins - self.max_mate_range - 2:
            return "Eval", self.bin_ticks[-1], np.inf
        else:
            return "Mate", self.total_num_bins - idx, self.total_num_bins - idx
