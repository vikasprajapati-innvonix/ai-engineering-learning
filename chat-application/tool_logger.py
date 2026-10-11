import logging
import time

from langchain_core.callbacks import BaseCallbackHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tools")


class ToolLogger(BaseCallbackHandler):
    def __init__(self):
        self._start = {}

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._start[run_id] = time.perf_counter()
        name = (serialized or {}).get("name") or kwargs.get("name")
        log.info("TOOL START  %s | input=%s", name, input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        took = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        log.info("TOOL END    %.2fs | output=%s", took, str(getattr(output, "content", output))[:300])

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._start.pop(run_id, None)
        log.error("TOOL ERROR %r", error)