import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

class OptionalLangfuse:
    """Optional, non-blocking Langfuse observability."""

    def __init__(self):
        self.client = None
        self.enabled = False

        requested = (
            os.getenv("LANGFUSE_ENABLED", "false")
            .strip()
            .lower() == "true"
        )

        if not requested:
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            logger.warning(
                "Langfuse disabled: missing API keys."
            )
            return

        try:
            from langfuse import get_client

            self.client = get_client()
            self.enabled = True

        except Exception as error:
            logger.warning(
                "Langfuse initialization failed: %s",
                error,
            )

    @contextmanager
    def observation(
        self,
        name: str,
        as_type: str = "span",
        **kwargs,
    ):
        """Create an observation without blocking RAG."""

        if not self.enabled:
            yield None
            return

        try:
            manager = self.client.start_as_current_observation(
                name=name,
                as_type=as_type,
                **kwargs,
            )

            current = manager.__enter__()

        except Exception as error:
            logger.warning(
                "Langfuse observation unavailable: %s",
                error,
            )
            yield None
            return

        try:
            yield current

        except BaseException:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception as error:
                logger.warning(
                    "Langfuse observation close failed: %s",
                    error,
                )
            raise

        else:
            try:
                manager.__exit__(None, None, None)
            except Exception as error:
                logger.warning(
                    "Langfuse observation close failed: %s",
                    error,
                )

    def update(self, observation, **kwargs):
        """Safely update an existing observation."""

        if observation is None:
            return

        try:
            observation.update(**kwargs)
        except Exception as error:
            logger.warning(
                "Langfuse update failed: %s",
                error,
            )

    def flush(self):
        """Send pending telemetry in short-lived scripts."""

        if not self.enabled:
            return

        try:
            self.client.flush()
        except Exception as error:
            logger.warning(
                "Langfuse flush failed: %s",
                error,
            )