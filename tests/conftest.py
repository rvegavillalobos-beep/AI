import warnings

import pytest

from helpers import make_export, to_xlsx
from rickhard import analysis, demo, loading
from rickhard.config import LoadOptions, Scope, Settings

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="session")
def demo_bytes() -> bytes:
    return demo.demo_file("xlsx")


@pytest.fixture(scope="session")
def demo_ds(demo_bytes):
    return loading.build_dataset([("demo.xlsx", demo_bytes)])


@pytest.fixture(scope="session")
def demo_analysis(demo_ds):
    return analysis.analyze(demo_ds.runs, Settings(), Scope(), 3, demo_ds.has_z, demo_ds.limits)


@pytest.fixture(scope="session")
def synth_df():
    return make_export()


@pytest.fixture(scope="session")
def synth_xlsx(synth_df) -> bytes:
    return to_xlsx(synth_df)


@pytest.fixture(scope="session")
def original_opts() -> LoadOptions:
    """Reading options that reproduce the original app's rules exactly."""
    return LoadOptions(run_gap_min=0, zero_as_missing=False)
