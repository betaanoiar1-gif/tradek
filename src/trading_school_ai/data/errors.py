class DataError(Exception):
    """Base class for data contract violations."""


class MissingDatasetError(DataError):
    """Canonical dataset is not available at the configured path."""


class SchemaError(DataError):
    """Dataset schema does not match the canonical contract."""


class TemporalContractError(DataError):
    """Timestamp ordering / uniqueness / timezone contract violated."""


class OHLCValidityError(DataError):
    """OHLC relationships or volume are invalid."""
