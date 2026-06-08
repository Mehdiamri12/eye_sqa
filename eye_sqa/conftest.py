"""
conftest.py — Root-level pytest fixtures.

Provides the API server and Playwright browser fixtures for E2E tests.
The server starts automatically when E2E tests run — no manual startup needed.

uvicorn and playwright are only imported inside fixtures so that running
unit/API tests without these packages installed does not fail.
"""

import pytest
import threading
import time


@pytest.fixture(scope="session", autouse=False)
def api_server():
    """Start the API server once for the full E2E session."""
    import uvicorn

    class _Server(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.config = uvicorn.Config(
                "api.main:app", host="127.0.0.1", port=8000, log_level="error"
            )
            self.server = uvicorn.Server(self.config)
        def run(self):
            self.server.run()
        def stop(self):
            self.server.should_exit = True

    srv = _Server()
    srv.start()
    time.sleep(1.5)
    print("\n[conftest] API server started on http://localhost:8000")
    yield
    srv.stop()
    print("\n[conftest] API server stopped")


@pytest.fixture(scope="session")
def browser_session(api_server):
    """Single Chromium instance for the full E2E session."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        print("[conftest] Browser launched")
        yield browser
        browser.close()
        print("[conftest] Browser closed")


@pytest.fixture
def page(browser_session):
    """
    Fresh browser context + page for each test.
    Stays open (page.pause()) after the test so you can interact
    with the result. Press Resume in the Playwright Inspector to continue.
    """
    ctx  = browser_session.new_context(viewport={"width": 1100, "height": 800})
    page = ctx.new_page()
    yield page
    page.pause()   # keeps browser open — press Resume to move to next test
    ctx.close()
