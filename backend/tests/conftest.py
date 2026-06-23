"""Shared pytest fixtures for the API tests.

The `client` fixture is session-scoped: entering TestClient as a context
manager runs the app's real lifespan (model + graph warm-up), which takes
~30s because it loads the trained model and runs it over the full test set.
Scoping it to the session means every test in the suite pays that cost once,
not once per test.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
