"""optionsDX adapter tests, driven by a miniature file in the real format."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data.adapters.optionsdx import (CANONICAL_COLUMNS, clean_columns,
                                     convert_directory, read_optionsdx,
                                     to_quote_dicts)

HEADER = ("[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], "
          "[UNDERLYING_LAST], [EXPIRE_DATE], [EXPIRE_UNIX], [DTE], [C_DELTA], "
          "[C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], [C_IV], [C_VOLUME], [C_LAST], "
          "[C_SIZE], [C_BID], [C_ASK], [STRIKE], [P_BID], [P_ASK], [P_SIZE], "
          "[P_LAST], [P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV], "
          "[P_VOLUME], [STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]")


def row(strike, dte, p_bid, p_ask, p_delta, expiry="2024-02-16",
        quote="2024-01-02", volume="120.0"):
    return (f"1704225600, {quote} 16:00, {quote}, 16.000000, 450.000000, "
            f"{expiry}, 1708119000, {dte}.000000, 0.8, 0.01, 0.05, -0.02, 0.007, "
            f"0.15, 900.0, 20.10, 1 x 1, 20.00, 20.20, {strike}.000000, "
            f"{p_bid}, {p_ask}, 12 x 340, 2.55, {p_delta}, 0.010000, 0.050000, "
            f"-0.020000, -0.004, 0.180000, {volume}, 20.0, 0.044")


def write_file(path, rows):
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def sample(tmp_path):
    return write_file(tmp_path / "spy_eod_202401.txt", [
        row(400, 45, "2.000000", "2.400000", "-0.200000"),
        row(395, 45, "0.800000", "1.200000", "-0.140000"),
        row(300, 400, "0.050000", "0.100000", "-0.010000"),   # far LEAP, filtered
    ])


def test_clean_columns_strips_brackets_and_padding():
    df = pd.DataFrame(columns=[" [QUOTE_DATE] ", "[P_BID]"])
    assert list(clean_columns(df).columns) == ["QUOTE_DATE", "P_BID"]


def test_read_produces_canonical_columns(sample):
    assert list(read_optionsdx(sample).columns) == CANONICAL_COLUMNS


def test_wide_rows_are_unpivoted_to_one_put_per_row(sample):
    df = read_optionsdx(sample, option_types=("P",))
    assert (df.option_type == "P").all()
    assert set(df.strike) == {400.0, 395.0}       # the 400-DTE row is filtered out


def test_put_fields_come_from_the_p_prefixed_columns(sample):
    df = read_optionsdx(sample).set_index("strike")
    assert df.loc[400.0, "bid"] == 2.0            # not the call's 20.00
    assert df.loc[400.0, "ask"] == 2.4
    assert df.loc[400.0, "delta"] == -0.2
    assert df.loc[400.0, "mid"] == pytest.approx(2.2)


def test_calls_can_be_requested_separately(sample):
    df = read_optionsdx(sample, option_types=("C",))
    assert (df.option_type == "C").all()
    assert df.iloc[0]["bid"] == 20.0


def test_dte_window_filters_during_the_read(sample):
    assert len(read_optionsdx(sample, min_dte=0, max_dte=60)) == 2
    assert len(read_optionsdx(sample, min_dte=0, max_dte=500)) == 3
    assert len(read_optionsdx(sample, min_dte=100, max_dte=500)) == 1


def test_quote_size_is_split_into_bid_and_ask_size(sample):
    df = read_optionsdx(sample)
    assert df.iloc[0]["bid_size"] == 12
    assert df.iloc[0]["ask_size"] == 340


def test_timestamp_is_stamped_at_the_close(sample):
    """optionsDX is a 16:00 snapshot, not the spec's 15:45 decision time."""
    ts = read_optionsdx(sample).iloc[0]["timestamp"]
    assert (ts.hour, ts.minute) == (16, 0)


def test_open_interest_is_zero_because_the_source_has_none(sample):
    """P_SIZE is bid/ask size, not open interest. min_oi must be disabled."""
    assert (read_optionsdx(sample).open_interest == 0).all()


def test_missing_volume_becomes_nan_not_zero(tmp_path):
    f = write_file(tmp_path / "spy_eod_202402.txt",
                   [row(400, 45, "2.000000", "2.400000", "-0.200000", volume="")])
    assert pd.isna(read_optionsdx(f).iloc[0]["volume"])


def test_empty_result_still_has_canonical_columns(sample):
    out = read_optionsdx(sample, min_dte=900, max_dte=999)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.empty


def test_convert_directory_writes_one_parquet(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_file(src / "spy_eod_202401.txt", [row(400, 45, "2.0", "2.4", "-0.20")])
    write_file(src / "spy_eod_202402.txt",
               [row(410, 40, "2.5", "2.9", "-0.25", quote="2024-02-01")])
    info = convert_directory(src, tmp_path / "out.parquet", progress=None)
    assert info["files"] == 2 and info["rows"] == 2
    df = pd.read_parquet(tmp_path / "out.parquet")
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df.timestamp.is_monotonic_increasing


def test_convert_directory_rejects_an_empty_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .txt files"):
        convert_directory(tmp_path, tmp_path / "x.parquet", progress=None)


def test_to_quote_dicts_feeds_the_generic_normalizer(sample):
    from data.adapters.sample import normalize_rows
    res = normalize_rows(list(to_quote_dicts(read_optionsdx(sample))))
    assert len(res.quotes) == 2
    q = next(q for q in res.quotes if q.strike == 400.0)
    assert q.option_type == "P" and q.bid == 2.0 and q.delta == -0.2
