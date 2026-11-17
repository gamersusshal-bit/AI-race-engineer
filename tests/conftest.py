import pytest

from race_engineer.data.schema import validate_laps
from race_engineer.data.synthetic import generate_dataset


@pytest.fixture(scope="session")
def raw_laps():
    return generate_dataset(seasons=(2022, 2023, 2024), seed=3, with_telemetry=True)


@pytest.fixture(scope="session")
def laps(raw_laps):
    return validate_laps(raw_laps)[0]


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, raw_laps):
    from race_engineer import pipeline
    out = tmp_path_factory.mktemp("artifacts")
    return out, pipeline.run(out, seed=3)
