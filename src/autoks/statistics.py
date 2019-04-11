from typing import Callable, Optional, List, Dict, Iterable, Union, Any

import numpy as np

DataType = Union[float, List[float]]
DataList = List[DataType]
Function = Callable[..., Union[float, np.ndarray, List[float]]]


class Statistic:
    data: DataList

    def __init__(self, name: str,
                 function: Optional[Function] = None):
        self.name = name  # the name of the statistic e.g. arithmetic mean
        self.data = []

        if function is None:
            self.function = lambda x: x  # identity function
        else:
            self.function = function

    def record(self, data: Any, *args, **kwargs) -> None:
        value = self.function(data, *args, **kwargs)
        self.data.append(value)

    def plot(self):
        raise NotImplementedError()

    def clear(self) -> None:
        """Reset statistic."""
        self.data = []

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name!r}, n_data={len(self.data)}, ' \
            f'function={self.function.__name__!r})'


class MultiStat:
    """Track multiple statistics."""
    stats: Dict[str, Statistic]
    RAW_VALUE_STAT_NAME = 'raw_values'

    def __init__(self, name: str,
                 stats: Optional[Iterable[Statistic]] = None):
        """

        :param name: The name of the collection of random variables, statistical population, data set, or probability
        distribution .
        :param stats:
        """
        self.name = name  # the name of the random variable to track
        self.stats = dict()
        if stats is not None:
            for stat in stats:
                self.add_statistic(stat)

    def add_statistic(self, statistic: Statistic):
        self.stats[statistic.name] = statistic

    def add_raw_value_stat(self, function: Optional[Function] = None):
        raw_value_stat = Statistic(self.RAW_VALUE_STAT_NAME, function)
        self.add_statistic(raw_value_stat)

    def get_raw_values(self) -> Optional[DataList]:
        if self.RAW_VALUE_STAT_NAME in self.stats:
            return self.stats[self.RAW_VALUE_STAT_NAME].data

    def stat_names(self) -> List[str]:
        return list(self.stats.keys())

    def stats_list(self) -> List[Statistic]:
        return list(self.stats.values())

    # convenience methods for raw values
    def mean(self) -> DataList:
        return [float(np.mean(value)) for value in self.get_raw_values()]

    def median(self) -> DataList:
        return [float(np.median(value)) for value in self.get_raw_values()]

    def maximum(self) -> DataList:
        return [float(np.max(value)) for value in self.get_raw_values()]

    def std(self) -> DataList:
        return [float(np.std(value)) for value in self.get_raw_values()]

    def var(self) -> DataList:
        return [float(np.var(value)) for value in self.get_raw_values()]

    def sum(self) -> DataList:
        return [float(np.sum(value)) for value in self.get_raw_values()]

    def running_max(self) -> DataList:
        max_so_far = []
        for data_point in self.get_raw_values():
            max_data_point = np.max(data_point)
            if max_so_far:
                max_data_point = max(max_data_point, np.max(max_so_far))
            max_so_far.append(max_data_point)
        return max_so_far

    def running_mean(self) -> DataList:
        means = [np.mean(value) for value in self.get_raw_values()]
        mean_so_far = np.cumsum(means) / np.arange(1, len(means) + 1)
        return list(mean_so_far.tolist())

    def running_std(self) -> DataList:
        values = np.array(self.get_raw_values())
        rollling_std = np.zeros(values.size)
        for i in range(rollling_std.size):
            if i > 0:
                rollling_std[i] = np.std(values[:i + 1])
        return list(rollling_std.tolist())

    def plot(self):
        raise NotImplementedError()

    def clear_all_values(self) -> None:
        for statistic in self.stats.values():
            statistic.clear()

    def __repr__(self):
        return f'{self.__class__.__name__}(stats={self.stats!r})'


class StatBook:
    """Store many multi-statistics."""
    multi_stats: Dict[str, MultiStat]

    def __init__(self, name: str,
                 multi_stats: Optional[Iterable[MultiStat]] = None):
        self.name = name
        self.multi_stats = dict()
        if multi_stats is not None:
            for multi_stat in multi_stats:
                self.add_multi_stat(multi_stat)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name
        self.label = ' '.join(name.split('_')).capitalize()

    def add_multi_stat(self, multi_stat: MultiStat):
        self.multi_stats[multi_stat.name] = multi_stat

    def add_raw_value_stat(self, multi_stat_name: str,
                           function: Optional[Function] = None):
        self.add_multi_stat(MultiStat(multi_stat_name))
        multi_stat = self.multi_stats[multi_stat_name]
        multi_stat.add_raw_value_stat(function=function)

    def get_raw_values(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].get_raw_values()

    # wrappers for convenience methods
    def mean(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].mean()

    def median(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].median()

    def maximum(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].maximum()

    def std(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].std()

    def var(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].var()

    def sum(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].sum()

    def running_max(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].running_max()

    def running_mean(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].running_mean()

    def running_std(self, multi_stat_name: str) -> DataList:
        return self.multi_stats[multi_stat_name].running_std()

    def multi_stats_names(self) -> List[str]:
        return list(self.multi_stats.keys())

    def multi_stats_list(self) -> List[MultiStat]:
        return list(self.multi_stats.values())

    def update_stat_book(self, data: Any, *args, **kwargs):
        for multi_stat in self.multi_stats_list():
            for stat in multi_stat.stats_list():
                stat.record(data, *args, **kwargs)

    def clear_all_values(self) -> None:
        for multi_stat in self.multi_stats.values():
            multi_stat.clear_all_values()

    def __repr__(self):
        multi_stat_names = [multi_stat.name for multi_stat in self.multi_stats_list()]
        return f'{self.__class__.__name__}(name={self.name!r}, multi_stats={multi_stat_names!r})'


class StatBookCollection:
    """Store many stat books."""
    stat_books: Dict[str, StatBook]

    def __init__(self, stat_book_names: Iterable[str],
                 multi_stat_names: Iterable[str],
                 raw_value_functions: Optional[Iterable[Function]] = None):
        # TODO:  make raw value functions map to mutli-stats
        self.stat_books = dict()
        for sb_name in stat_book_names:
            # shared multi-stats
            multi_stats = [MultiStat(ms_name) for ms_name in multi_stat_names]
            functions = [None] * len(multi_stats) if raw_value_functions is None else raw_value_functions
            for ms, fxn in zip(multi_stats, functions):
                ms.add_raw_value_stat(function=fxn)
            self.add_stat_book(StatBook(sb_name, multi_stats))

    def add_stat_book(self, stat_book: StatBook):
        self.stat_books[stat_book.name] = stat_book

    def stat_book_names(self) -> List[str]:
        return list(self.stat_books.keys())

    def stat_book_list(self) -> List[StatBook]:
        return list(self.stat_books.values())

    def clear_all_values(self) -> None:
        for stat_book in self.stat_books.values():
            stat_book.clear_all_values()

    def __repr__(self):
        stat_book_names = [book.name for book in self.stat_book_list()]
        return f'{self.__class__.__name__}(stat_books={stat_book_names!r})'
