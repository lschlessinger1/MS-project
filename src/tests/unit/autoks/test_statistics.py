import os
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from src.autoks.statistics import Statistic, MultiStat, StatBook, StatBookCollection


class TestStatistic(TestCase):

    def test_record(self):
        stat = Statistic(name='tests', function=lambda x: x + 1)
        self.assertEqual(2, stat.function(1))
        stat.record(5)
        self.assertListEqual(stat.data, [6])

    def test_clear(self):
        stat = Statistic(name='tests')
        stat.record(5)
        stat.clear()
        self.assertListEqual(stat.data, [])

    def test_to_dict(self):
        stat = Statistic(name='tests')
        stat.record(5)
        output_dict = stat.to_dict()
        self.assertEqual(stat.name, output_dict["name"])
        self.assertEqual(stat.data, output_dict["data"])
        self.assertEqual(stat.function.__name__, output_dict["function"])

    def test_from_dict(self):
        stat = Statistic(name='tests')
        stat.record(5)
        output_dict = stat.to_dict()

        actual = Statistic.from_dict(output_dict)
        self.assertEqual(stat.name, actual.name)
        self.assertEqual(stat.data, actual.data)
        self.assertEqual(stat.function.__name__, actual.function.__name__)

        def square(x):
            return x ** 2

        stat = Statistic(name='tests', function=square)
        stat.record(2)
        output_dict = stat.to_dict()

        actual = Statistic.from_dict(output_dict, fn_map={"square": square})
        self.assertEqual(stat.name, actual.name)
        self.assertEqual(stat.data, actual.data)
        self.assertEqual(stat.function.__name__, actual.function.__name__)
        self.assertEqual(stat.function(2), actual.function(2))


class TestMultiStat(TestCase):

    def test_add_statistic(self):
        ms = MultiStat('test_ms')
        stat = Statistic(name='test_stat')
        ms.add_statistic(stat)
        self.assertIn(stat.name, ms.stats)
        self.assertEqual(ms.stats.get(stat.name), stat)

    def test_add_raw_value_stat(self):
        ms = MultiStat('test_ms')
        func = lambda x: x ** 2
        ms.add_raw_value_stat(func)
        self.assertIn(ms.RAW_VALUE_STAT_NAME, ms.stats)
        self.assertEqual(ms.stats.get(ms.RAW_VALUE_STAT_NAME).name, ms.RAW_VALUE_STAT_NAME)
        self.assertEqual(ms.stats.get(ms.RAW_VALUE_STAT_NAME).function, func)

    def test_get_raw_values(self):
        ms = MultiStat('test_ms')
        result = ms.get_raw_values()
        self.assertIsNone(result)

        ms.add_raw_value_stat(lambda x: x ** 2)
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

    def test_sum(self):
        ms = MultiStat('test_ms')
        ms.add_raw_value_stat(lambda x: x)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(1)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(2)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record(3)
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([5, 7])
        ms.stats.get(ms.RAW_VALUE_STAT_NAME).record([4, 1, 1])
        self.assertListEqual(ms.sum(), [1, 2, 3, 12, 6])

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

    def test_to_dict(self):
        ms = MultiStat(name='test_ms')
        stat_1 = Statistic('name1')
        stat_1.record(6)
        ms.add_statistic(stat_1)
        output_dict = ms.to_dict()
        self.assertEqual(ms.name, output_dict["name"])
        self.assertEqual([stat.to_dict() for stat in ms.stats_list()], output_dict["stats"])


class TestStatBook(TestCase):

    def test_add_multistat(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        sb.add_multi_stat(ms)
        self.assertIn(ms.name, sb.multi_stats)
        self.assertEqual(sb.multi_stats.get(ms.name), ms)

    def test_add_raw_value_stat(self):
        sb = StatBook('test_sb')
        func = lambda x: x + 1
        sb.add_raw_value_stat('test_ms', func)
        self.assertIn('test_ms', sb.multi_stats)
        self.assertIn(MultiStat.RAW_VALUE_STAT_NAME, sb.multi_stats['test_ms'].stats)
        self.assertEqual(sb.multi_stats['test_ms'].stats.get(MultiStat.RAW_VALUE_STAT_NAME).function, func)

    def test_get_raw_values(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.get_raw_values = MagicMock()
        ms.get_raw_values.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.get_raw_values(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.get_raw_values.assert_called_once()

    def test_mean(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.mean = MagicMock()
        ms.mean.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.mean(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.mean.assert_called_once()

    def test_median(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.median = MagicMock()
        ms.median.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.median(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.median.assert_called_once()

    def test_maximum(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.maximum = MagicMock()
        ms.maximum.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.maximum(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.maximum.assert_called_once()

    def test_std(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.std = MagicMock()
        ms.std.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.std(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.std.assert_called_once()

    def test_var(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.var = MagicMock()
        ms.var.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.var(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.var.assert_called_once()

    def test_running_max(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.running_max = MagicMock()
        ms.running_max.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.running_max(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.running_max.assert_called_once()

    def test_running_mean(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.running_mean = MagicMock()
        ms.running_mean.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.running_mean(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.running_mean.assert_called_once()

    def test_running_std(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_stat')
        ms.running_std = MagicMock()
        ms.running_std.return_value = [3., 4., 5.]
        sb.add_multi_stat(ms)
        result = sb.running_std(ms.name)
        self.assertListEqual(result, [3., 4., 5.])
        ms.running_std.assert_called_once()

    def test_multi_stats_names(self):
        sb = StatBook('test_sb')
        ms1 = MultiStat(name='test_stat1')
        ms2 = MultiStat(name='test_stat2')
        sb.add_multi_stat(ms1)
        sb.add_multi_stat(ms2)
        result = sb.multi_stats_names()
        self.assertListEqual(result, [ms1.name, ms2.name])

    def test_multi_stats(self):
        sb = StatBook('test_sb')
        ms1 = MultiStat(name='test_stat1')
        ms2 = MultiStat(name='test_stat2')
        sb.add_multi_stat(ms1)
        sb.add_multi_stat(ms2)
        result = sb.multi_stats_list()
        self.assertListEqual(result, [ms1, ms2])

    def test_update_stat_book(self):
        sb = StatBook('test_sb')
        sb.add_raw_value_stat('test_stat1')
        sb.add_raw_value_stat('test_stat2')
        sb.update_stat_book(data=[1, 2, 3])

        result = sb.multi_stats['test_stat1'].get_raw_values()
        self.assertListEqual(result, [[1, 2, 3]])

        result = sb.multi_stats['test_stat2'].get_raw_values()
        self.assertListEqual(result, [[1, 2, 3]])

    def test_clear_all_values(self):
        sb = StatBook('test_sb')

        ms1 = MultiStat(name='test_stat1')
        ms1.clear_all_values = MagicMock()

        ms2 = MultiStat(name='test_stat2')
        ms2.clear_all_values = MagicMock()

        sb.add_multi_stat(ms1)
        sb.add_multi_stat(ms2)

        sb.clear_all_values()

        ms1.clear_all_values.assert_called_once()
        ms2.clear_all_values.assert_called_once()

    def test_to_dict(self):
        sb = StatBook('test_sb')
        ms = MultiStat(name='test_ms')
        sb.add_multi_stat(ms)
        output_dict = sb.to_dict()
        self.assertEqual(sb.name, output_dict["name"])
        self.assertEqual([ms.to_dict() for ms in sb.multi_stats_list()], output_dict["multi_stats"])


class TestStatBookCollection(TestCase):

    def setUp(self) -> None:
        self.sb_names = ["sb_name_1", "sb_name_2"]
        self.ms_names = ["mean", "variance"]

    def test_add_stat_book(self):
        sbc = StatBookCollection()
        sbc.create_shared_stat_books(self.sb_names, self.ms_names)

        sb = StatBook('test_sb')
        sbc.add_stat_book(sb)
        self.assertIn(sb.name, sbc.stat_book_names())
        self.assertIn(sb, sbc.stat_book_list())

    def test_to_dict(self):
        sbc = StatBookCollection()
        sbc.create_shared_stat_books(self.sb_names, self.ms_names)

        output_dict = sbc.to_dict()
        self.assertEqual([sb.to_dict() for sb in sbc.stat_book_list()], output_dict["stat_books"])

    def test_from_dict(self):
        sbc = StatBookCollection()
        sbc.create_shared_stat_books(self.sb_names, self.ms_names)

        actual = StatBookCollection.from_dict(sbc.to_dict())
        self.assertEqual(sbc.stat_book_names(), actual.stat_book_names())

        expected_ms_names = [sb.multi_stats_names() for sb in sbc.stat_book_list()]
        actual_ms_names = [sb.multi_stats_names() for sb in actual.stat_book_list()]
        self.assertListEqual(expected_ms_names, actual_ms_names)

    def test_save(self):
        sbc = StatBookCollection()
        sbc.create_shared_stat_books(self.sb_names, self.ms_names)

        fname = "test_save"
        out_fname = sbc.save(fname)

        self.addCleanup(os.remove, out_fname)

    def test_load(self):
        sbc = StatBookCollection()
        sbc.create_shared_stat_books(self.sb_names, self.ms_names)

        fname = "test_save"
        out_fname = sbc.save(fname)

        new_sbc = StatBookCollection.load(out_fname)

        self.assertEqual(sbc.stat_book_names(), new_sbc.stat_book_names())

        expected_ms_names = [sb.multi_stats_names() for sb in sbc.stat_book_list()]
        actual_ms_names = [sb.multi_stats_names() for sb in new_sbc.stat_book_list()]
        self.assertListEqual(expected_ms_names, actual_ms_names)

        self.addCleanup(os.remove, out_fname)
