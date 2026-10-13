"""Private format marker and bounded views for built-in analytical exports."""
from sys import getsizeof


class _ColumnFormat(str):
    pass


COLUMN_FORMAT = _ColumnFormat('dict')


class ColumnBatch:
    def __init__(self, source, columns):
        self.source = source
        self.columns = tuple(columns)
        self.conversion_bytes = 0

    def __len__(self):
        return len(self.source)

    def column_values(self):
        for row in self.source:
            if not isinstance(row, (list, tuple)) or len(row) != len(self.columns):
                raise ValueError('Parquet rows must contain exactly the selected columns')
        values = tuple(zip(*self.source)) if len(self) else tuple(() for _ in self.columns)
        self.conversion_bytes = getsizeof(values) + sum(getsizeof(column) for column in values)
        self.observe_native(self.conversion_bytes)
        return values

    def dict_rows(self):
        return [dict(zip(self.columns, row, strict=True)) for row in self.source]

    def observe_native(self, size_bytes):
        observe = getattr(self.source, 'observe_native', None)
        if observe is not None:
            observe(size_bytes)
