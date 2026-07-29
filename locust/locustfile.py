"""Locust load test for RurangaSort AI.

Usage:
    locust -f locust/locustfile.py --host http://localhost

Point --host at the Nginx load balancer (http://localhost with docker-compose,
or the ALB/public URL in production), not directly at a single API container --
otherwise you are not exercising the load balancer you're trying to measure.

Scenarios referenced in the README (1/2/4 API containers x 10/50/100 users) are
run from the command line / web UI, e.g.:

    locust -f locust/locustfile.py --host http://localhost \\
        --users 50 --spawn-rate 5 --run-time 3m --headless \\
        --csv reports/load_tests/2c_50u
"""
from __future__ import annotations

from pathlib import Path

from locust import HttpUser, between, task

TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"
IMAGE_PATHS = sorted(TEST_IMAGES_DIR.glob("*.jpg"))


class RurangaSortUser(HttpUser):
    """Simulates a user repeatedly uploading an image for prediction, with
    occasional health/metrics checks (the same traffic mix the dashboard and
    prediction page would generate)."""

    wait_time = between(0.5, 2.0)

    def on_start(self):
        if not IMAGE_PATHS:
            raise RuntimeError(
                f"No sample images found in {TEST_IMAGES_DIR}. "
                "Run scripts/generate_synthetic_dataset.py or add a few .jpg files there."
            )
        self._images = [path.read_bytes() for path in IMAGE_PATHS]
        self._cursor = 0

    def _next_image(self) -> bytes:
        image = self._images[self._cursor % len(self._images)]
        self._cursor += 1
        return image

    @task(10)
    def predict(self):
        files = {"file": ("sample.jpg", self._next_image(), "image/jpeg")}
        self.client.post("/predict", files=files, name="/predict")

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")
