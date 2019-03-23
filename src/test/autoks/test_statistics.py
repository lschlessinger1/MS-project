from unittest import TestCase

from src.autoks.statistics import Statistic, MultiStat


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