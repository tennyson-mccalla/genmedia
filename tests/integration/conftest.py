import os
import pytest


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("GENMEDIA_TEST_LIVE"):
        skip = pytest.mark.skip(reason="Set GENMEDIA_TEST_LIVE=1 to run integration tests")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip)
