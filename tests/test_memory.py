"""Testes do MemoryMonitor."""

import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.memory import MemoryMonitor


class TestMemoryMonitor(unittest.TestCase):

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            MemoryMonitor(budget_mb=0)
        with self.assertRaises(ValueError):
            MemoryMonitor(budget_mb=-1)

    def test_used_mb_returns_positive(self):
        m = MemoryMonitor(budget_mb=1024)
        used = m.used_mb()
        # Qualquer processo Python tem RSS positivo
        self.assertGreater(used, 0)

    def test_used_ratio_consistent(self):
        m = MemoryMonitor(budget_mb=1024)
        used = m.used_mb()
        ratio = m.used_ratio()
        # ratio = used / budget (com tolerancia para chamadas concorrentes)
        self.assertAlmostEqual(used / 1024, ratio, places=2)

    def test_peak_tracking(self):
        m = MemoryMonitor(budget_mb=1024)
        _ = m.used_mb()
        peak1 = m.peak_mb()
        # Aloca algum lixo
        garbage = [list(range(10000)) for _ in range(100)]
        _ = m.used_mb()
        peak2 = m.peak_mb()
        # peak2 >= peak1 (pico nao diminui)
        self.assertGreaterEqual(peak2, peak1)
        del garbage

    def test_reset_peak(self):
        m = MemoryMonitor(budget_mb=1024)
        _ = m.used_mb()
        m.reset_peak()
        # Apos reset, peak = uso atual
        self.assertAlmostEqual(m.peak_mb(), m.used_mb(), delta=10)

    def test_should_flush_with_small_budget(self):
        # Com budget muito pequeno, used_ratio > threshold
        m = MemoryMonitor(budget_mb=1)  # 1 MB so
        self.assertTrue(m.should_flush())

    def test_should_not_flush_with_huge_budget(self):
        # Com budget enorme, used_ratio << threshold
        m = MemoryMonitor(budget_mb=999_999)
        self.assertFalse(m.should_flush())


if __name__ == "__main__":
    unittest.main()
