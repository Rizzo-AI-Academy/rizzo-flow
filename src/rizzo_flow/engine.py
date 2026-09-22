"""Thread-safe long-lived model service with deterministic postprocessing."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .calibration import Calibration
from .decisions import decode
from .prompts import compile_request
from .protocols import ScoringBackend
from .responses import Response
from .schema import Request


class Engine:
    def __init__(
        self,
        backend: ScoringBackend,
        ctx: int = 8192,
        calibration: Calibration | None = None,
    ) -> None:
        if ctx < 1:
            raise ValueError("ctx must be positive")
        self.backend = backend
        self.ctx = ctx
        self.calibration = calibration
        if calibration and calibration.fingerprint != backend.metadata.fingerprint:
            raise ValueError(
                "Calibration was fitted for a different model/runtime/prompt configuration"
            )
        self._lock = Lock()
        # All model work runs on one thread that lives as long as the process. Web servers call
        # decide() from short-lived pool threads, and MLX's CUDA backend aborts the process when
        # a thread that ran computations exits.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rizzo-inference")

    def decide(self, request: Request | dict) -> dict:
        # Re-validate a serialized snapshot, also protecting mutable Pydantic objects.
        request = Request.model_validate(
            request.model_dump() if isinstance(request, Request) else request
        )
        started = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            prefix, jobs = compile_request(self.backend.tokenizer, request, self.ctx)
            encoded = time.perf_counter()
            logits, timing = self._worker.submit(
                self.backend.score, prefix, jobs, request.mode
            ).result()
            answers = {}
            for job in jobs:
                question = request.questions[job.id]
                temperature = (
                    self.calibration.temperatures.get(question.type, 1.0)
                    if self.calibration
                    else 1.0
                )
                answers[job.id] = decode(question, logits[job.id], temperature)
                answers[job.id]["prompt_sha256"] = job.prompt_sha256
                answers[job.id]["input_tokens"] = len(job.tokens)
        response = {
            "model": self.backend.metadata.as_dict(),
            "mode": request.mode,
            "answers": answers,
            "calibration": self.calibration.model_dump() if self.calibration else None,
            "timing": timing.model_copy(
                update={
                    "queue_seconds": acquired - started,
                    "compile_seconds": encoded - acquired,
                    "total_seconds": time.perf_counter() - started,
                }
            ),
        }
        return Response.model_validate(response).model_dump()
