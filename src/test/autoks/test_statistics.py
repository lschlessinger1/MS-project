from unittest import TestCase
from unittest.mock import MagicMock

from src.autoks.statistics import Statistic, MultiStat
import numpy as np

class TestStatistic(TestCase):

    def test_record(self):
        stat = Statistic(name='test', function=lambda x: x + 1)
        self.assertEqual(2, stat.function(1))
        stat.record(5)
        self.assertListEqual(stat.data, [6])

    def test_clear(self):
        stat = Statistic(name='test')
        stat.record(5)
        stat.clear()
        self.assertListEqual(stat.data, [])


class TestMultiStat(TestCase):

    def test_add_statistic(self):
        ms = MultiStat('test_ms')
        stat = Statistic(name='test_stat')
        ms.add_statistic(stat)
        self.assertIn(stat.name, ms.stats)
        self.assertEqual(ms.stats.get(stat.name), stat)

    def test_add_raw_value_stat(self):
        ms = MultiStat('test_ms')
        func = lambda x: x**2
        ms.add_raw_value_stat(func)
        self.assertIn(ms.RAW_VALUE_STAT_NAME, ms.stats)
        self.assertEqual(ms.stats.get(ms.RAW_VALUE_STAT_NAME).name, ms.RAW_VALUE_STAT_NAME)
        self.assertEqual(ms.stats.get(ms.RAW_VALUE_STAT_NAME).function, func)

    def test_get_raw_values(self):
        ms = MultiStat('test_ms')
        result = ms.get_raw_values()
        self.assertIsNone(result)

        ms.add_raw_value_stat(lambda x: x**2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        self.assertListEqual(ms.get_raw_values(), [1., 4., 9.])

    def test_stat_names(self):
        ms = MultiStat('test_ms')
        stat_1 = Statistic('name1')
        stat_2 = Statistic('name2')
        stat_3 = Statistic('name3')
        ms.add_statistic(stat_1)
        ms.add_statistic(stat_2)
        ms.add_statistic(stat_3)
        self.assertListEqual([stat_1.name, stat_2.name, stat_3.name], ms.stat_names())

    def test_stats_list(self):
        ms = MultiStat('test_ms')
        stat_1 = Statistic('name1')
        stat_2 = Statistic('name2')
        stat_3 = Statistic('name3')
        ms.add_statistic(stat_1)
        ms.add_statistic(stat_2)
        ms.add_statistic(stat_3)
        self.assertListEqual([stat_1, stat_2, stat_3], ms.stats_list())

    def test_mean(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7, 9])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([0, 1])
        self.assertListEqual(ms.mean(), [1., 2., 3., 7., 0.5])

    def test_median(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7, 9])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([0, 1, 1])
        self.assertListEqual(ms.median(), [1., 2., 3., 7., 1.])

    def test_maximum(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7, 9])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([0, 1, 1])
        self.assertListEqual(ms.maximum(), [1., 2., 3., 9., 1.])

    def test_std(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7])
        self.assertListEqual(ms.std(), [0, 0, 0, 1.])

    def test_var(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([4, 1, 1])
        self.assertListEqual(ms.var(), [0, 0, 0, 1., 2.])

    def test_running_max(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([4, 1, 1])
        self.assertListEqual(ms.running_max(), [1, 2, 3, 7, 7])

    def test_running_mean(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7, 0])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([40, 0, 5, 5, 5, 5])
        self.assertListEqual(ms.running_mean(), [1., 1.5, 2., 2.5, 4.])

    def test_running_std(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        self.assertListEqual(ms.running_std(), [0, 0.5])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(4)
        self.assertTrue(np.allclose([0, 0.5, 1.247219128924647], ms.running_std()))

    def test_clear_all_values(self):
        ms = MultiStat('test_ms')
        stat_1 = Statistic('name1')
        stat_1.clear = MagicMock()
        stat_2 = Statistic('name2')
        stat_2.clear = MagicMock()
        stat_3 = Statistic('name3')
        stat_3.clear = MagicMock()
        ms.add_statistic(stat_1)
        ms.add_statistic(stat_2)
        ms.add_statistic(stat_3)

        ms.clear_all_values()
        stat_1.clear.assert_called_once()
        stat_2.clear.assert_called_once()
        stat_3.clear.assert_called_once()