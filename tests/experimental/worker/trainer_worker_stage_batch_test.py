# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `TrainerWorker.stage_batch`, the RPC server's executor hook and the
orchestrator's `stage_train_batch` (TUNIX_STAGE_BATCH=stage)."""

import asyncio
import contextlib
import os
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker


class FakeTrainer(abstract_trainer.AbstractTrainer):
  """The parts of trainer_worker_test's FakeTrainer these tests touch."""

  def __init__(self):
    self.fwd_bwd_calls = []
    self.policy_version = 3
    self.step_count = 10

  @property
  def checkpoint_dir(self) -> str | None:
    return None

  def compile(self, dummy_data=None):
    pass

  def with_loss_fn(self, loss_fn, has_aux=False):
    pass

  def with_gen_model_input_fn(self, gen_model_input_fn):
    pass

  def fwd_bwd(self, payload, **kwargs):
    self.fwd_bwd_calls.append((payload, kwargs))

  def update(self, **kwargs):
    self.step_count += 1
    return self.step_count

  def eval_step(self, payload, **kwargs):
    pass

  @contextlib.contextmanager
  def model_scope(self, *args, **kwargs):
    yield None, args, kwargs

  def save_checkpoint(self, metadata, **kwargs):
    pass

  def restore_checkpoint(self, **kwargs):
    return {}

  def get_metrics(self):
    return {}

  def prepare_weight_sync(self, **kwargs):
    pass

  def close(self):
    pass


class StagingFakeTrainer(FakeTrainer):
  """FakeTrainer with the engine's `stage_batch(payload, key=...)` hook."""

  def __init__(self, staged: bool = True):
    super().__init__()
    self.stage_calls = []
    self.staged = staged

  def stage_batch(self, payload, key=None):
    self.stage_calls.append((payload, key, threading.current_thread().name))
    return self.staged


def _payload():
  return datatypes.RLTrainerPayload(
      prompt_ids=np.array([[1, 2]], dtype=np.int32),
      prompt_mask=np.ones((1, 2), dtype=np.float32),
      completion_ids=np.array([[3, 4]], dtype=np.int32),
      completion_mask=np.ones((1, 2), dtype=np.float32),
      advantages=np.array([1.0], dtype=np.float32),
  )


class TrainerWorkerStageBatchTest(absltest.TestCase):

  def _worker(self, trainer):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: trainer, worker_id="trainer_0"
    )
    worker.initialize()
    return worker

  def test_stage_batch_forwards_payload_and_request_id_as_key(self):
    trainer = StagingFakeTrainer()
    worker = self._worker(trainer)
    payload = _payload()
    req = datatypes.TrainRequest(request_id="mb-7", payload=payload)
    resp = worker.stage_batch(request=req)
    self.assertTrue(resp.metadata["staged"])
    self.assertEqual(resp.request_id, "mb-7")
    self.assertLen(trainer.stage_calls, 1)
    self.assertIs(trainer.stage_calls[0][0], payload)
    self.assertEqual(trainer.stage_calls[0][1], "mb-7")
    # Not a step: no fwd_bwd, no state change, worker still ready.
    self.assertEmpty(trainer.fwd_bwd_calls)
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

  def test_stage_batch_without_trainer_support_says_not_staged(self):
    worker = self._worker(FakeTrainer())
    resp = worker.stage_batch(
        request=datatypes.TrainRequest(request_id="x", payload=_payload())
    )
    self.assertFalse(resp.metadata["staged"])
    self.assertIn("reason", resp.metadata)

  def test_stage_batch_trainer_declines(self):
    worker = self._worker(StagingFakeTrainer(staged=False))
    resp = worker.stage_batch(
        request=datatypes.TrainRequest(request_id="x", payload=_payload())
    )
    self.assertFalse(resp.metadata["staged"])

  def test_fwd_bwd_passes_stage_key_and_no_payload(self):
    trainer = StagingFakeTrainer()
    worker = self._worker(trainer)
    req = datatypes.TrainRequest(request_id="mb-1", payload=None)
    resp = worker.fwd_bwd(request=req, stage_key="mb-1")
    self.assertTrue(resp.metadata["queued"])
    self.assertLen(trainer.fwd_bwd_calls, 1)
    payload, kwargs = trainer.fwd_bwd_calls[0]
    self.assertIsNone(payload)
    self.assertEqual(kwargs["stage_key"], "mb-1")
    self.assertFalse(kwargs["apply_optimizer"])

  def test_executor_methods_default_and_with_env(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(trainer_worker.STAGE_BATCH_ENV, None)
      worker = self._worker(StagingFakeTrainer())
      self.assertEqual(worker.remote_executor_methods, frozenset({"stage_batch"}))
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      worker = self._worker(StagingFakeTrainer())
      self.assertEqual(
          worker.remote_executor_methods, frozenset({"stage_batch", "fwd_bwd"})
      )
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "serialize"}, clear=False
    ):
      self.assertFalse(trainer_worker.stage_batch_env_on())


class RemoteExecutionExecutorHookTest(absltest.TestCase):

  def test_listed_method_runs_off_the_loop_thread(self):
    trainer = StagingFakeTrainer()
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      worker = trainer_worker.TrainerWorker(
          trainer_factory=lambda: trainer, worker_id="trainer_0"
      )
      worker.initialize()
    server = remote_execution.InProcessRemoteExecutionServer(worker)
    loop_thread = threading.current_thread().name

    async def run():
      stage = remote_execution.ExecutionRequest(
          method_name="stage_batch",
          args=(),
          kwargs={
              "request": datatypes.TrainRequest(
                  request_id="s1", payload=_payload()
              )
          },
          request_id="r-stage",
      )
      info = remote_execution.ExecutionRequest(
          method_name="info", args=(), kwargs={}, request_id="r-info"
      )
      return await server.execute_request(stage), await server.execute_request(
          info
      )

    stage_resp, info_resp = asyncio.run(run())
    self.assertIsNone(stage_resp.error_message, stage_resp.traceback)
    self.assertTrue(stage_resp.result.metadata["staged"])
    self.assertLen(trainer.stage_calls, 1)
    # stage_batch ran on the rpc-method pool, not on the loop's thread; `info`
    # (not listed) ran inline as before.
    self.assertTrue(
        trainer.stage_calls[0][2].startswith("rpc-method"),
        trainer.stage_calls[0][2],
    )
    self.assertNotEqual(trainer.stage_calls[0][2], loop_thread)
    self.assertIsNone(info_resp.error_message)
    self.assertEqual(info_resp.result.worker_id, "trainer_0")


class _FakeHandle:
  """A trainer handle whose `asubmit` records the RPCs and answers like TrainerWorker."""

  def __init__(self, staged=True, has_stage=True):
    self.calls = []
    self.staged = staged
    self.has_stage = has_stage

  async def asubmit(self, method_name, **kwargs):
    self.calls.append((method_name, kwargs))
    if method_name == "stage_batch":
      if not self.has_stage:
        raise AttributeError("Method 'stage_batch' not found")
      return datatypes.Response(metadata={"staged": self.staged})
    if method_name == "fwd_bwd":
      return datatypes.Response(metadata={"queued": True})
    raise AssertionError(method_name)


class EngineStageTrainBatchTest(absltest.TestCase):

  def _engine(self, handle):
    engine = distributed_rl_engine.DistributedRLEngine.__new__(
        distributed_rl_engine.DistributedRLEngine
    )
    engine._trainer_workers = {datatypes.Role.ACTOR: handle}
    return engine

  def test_stage_returns_key_and_train_step_ships_no_payload(self):
    handle = _FakeHandle()
    engine = self._engine(handle)
    payload = _payload()
    key = asyncio.run(engine.stage_train_batch(payload))
    self.assertIsNotNone(key)
    self.assertEqual(handle.calls[0][0], "stage_batch")
    req = handle.calls[0][1]["request"]
    self.assertEqual(req.request_id, key)
    self.assertIs(req.payload, payload)
    # The train_step for it carries the key and no payload.
    engine._actor_busy_guard = mock.MagicMock()
    engine._actor_busy_guard.return_value.__aenter__ = mock.AsyncMock()
    engine._actor_busy_guard.return_value.__aexit__ = mock.AsyncMock(
        return_value=False
    )
    engine._actor_weights_dirty = False
    asyncio.run(
        engine.train_step(
            None, accumulate_gradients=True, apply_optimizer=False, stage_key=key
        )
    )
    name, kwargs = handle.calls[1]
    self.assertEqual(name, "fwd_bwd")
    self.assertIsNone(kwargs["request"].payload)
    self.assertEqual(kwargs["stage_key"], key)

  def test_not_staged_or_missing_method_returns_none(self):
    for handle in (_FakeHandle(staged=False), _FakeHandle(has_stage=False)):
      engine = self._engine(handle)
      self.assertIsNone(asyncio.run(engine.stage_train_batch(_payload())))


if __name__ == "__main__":
  absltest.main()
