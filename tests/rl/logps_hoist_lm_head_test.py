# Copyright 2025 Google LLC
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

"""compute_per_token_logps(hoist_lm_head=True) matches the chunked path, values and gradients.

Runs on 8 host CPU devices with the head weight and the hidden states actually sharded, so a
missing or doubled reduction of the head gradient shows up as a factor-of-axis-size error.
"""

import os

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
)

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from tunix.rl import common  # pylint: disable=g-import-not-at-top
from tunix.tests import test_common as tc  # pylint: disable=g-import-not-at-top

jax.config.update("jax_default_matmul_precision", "highest")

P = jax.sharding.PartitionSpec


def _inputs(batch, seq, vocab, seed=0):
  rng = np.random.default_rng(seed)
  completion = rng.integers(1, vocab, size=(batch, seq)).astype(np.int32)
  seg = np.zeros((batch, seq), np.int32)
  pos = np.zeros((batch, seq), np.int32)
  for b in range(batch):
    cut = seq // 2 + b  # two packed sequences, then padding at the end
    end = seq - 2
    seg[b, :cut] = 1
    seg[b, cut:end] = 2
    pos[b, :cut] = np.arange(cut)
    pos[b, cut:end] = np.arange(end - cut)
    completion[b, end:] = 0
  return completion, seg, pos


class HoistLmHeadTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # mesh (4, 2): batch 4 takes fsdp, context shards the sequence.
      dict(testcase_name="b4_fsdp4_ctx2", mesh_shape=(4, 2), batch=4, seq=37, chunk=8),
      # batch 2: fsdp cannot shard it, so fsdp shards the sequence too.
      dict(testcase_name="b2_fsdp4_ctx2", mesh_shape=(4, 2), batch=2, seq=50, chunk=16),
      # batch 3, 12 positions: context divides neither, so the hoist falls back to the unhoisted path.
      dict(testcase_name="b3_replicated_axis", mesh_shape=(4, 2), batch=3, seq=13, chunk=4),
  )
  def test_matches_chunked(
      self, mesh_shape, batch, seq, chunk, axis_type=jax.sharding.AxisType.Auto
  ):
    mesh = jax.make_mesh(
        mesh_shape, ("fsdp", "context"), axis_types=(axis_type, axis_type)
    )
    config = tc.ModelConfig()
    model = tc.ToyTransformer(config=config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    # Shard the head weight over fsdp on the hidden dim, the way FSDP shards it.
    def place(path, x):
      names = "/".join(str(getattr(p, "key", p)) for p in path)
      if "lm_head" in names and "kernel" in names:
        return jax.device_put(x, jax.sharding.NamedSharding(mesh, P("fsdp", None)))
      return jax.device_put(x, jax.sharding.NamedSharding(mesh, P()))

    state = jax.tree_util.tree_map_with_path(place, state)
    completion, seg, pos = _inputs(batch, seq, config.vocab_size)
    prompt = np.zeros((batch, 0), np.int32)
    w_logp = jnp.asarray(np.random.default_rng(1).normal(size=(batch, seq)), jnp.float32)
    w_ent = jnp.asarray(np.random.default_rng(2).normal(size=(batch, seq)), jnp.float32)

    def loss(st, hoist):
      logps, ent = common.compute_per_token_logps(
          graphdef,
          st,
          prompt,
          completion,
          pad_id=0,
          eos_id=-1,
          stop_gradient=False,
          return_entropy=True,
          segment_ids=seg,
          segment_positions=pos,
          temperature=0.7,
          chunk_size=chunk,
          hoist_lm_head=hoist,
      )
      return jnp.sum(logps * w_logp) + 0.1 * jnp.sum(ent * w_ent), (logps, ent)

    with mesh:  # legacy mesh context, as MaxText traces under
      (l0, (lp0, en0)), g0 = jax.jit(
          jax.value_and_grad(lambda s: loss(s, False), has_aux=True)
      )(state)
      (l1, (lp1, en1)), g1 = jax.jit(
          jax.value_and_grad(lambda s: loss(s, True), has_aux=True)
      )(state)

    diffs = {
        "logps": float(np.max(np.abs(np.asarray(lp1) - np.asarray(lp0)))),
        "entropy": float(np.max(np.abs(np.asarray(en1) - np.asarray(en0)))),
        "loss": float(abs(l1 - l0)),
    }
    for (path, a), b in zip(
        jax.tree_util.tree_leaves_with_path(g0), jax.tree_util.tree_leaves(g1)
    ):
      denom = float(np.max(np.abs(np.asarray(a)))) or 1.0
      diffs["grad" + jax.tree_util.keystr(path)] = (
          float(np.max(np.abs(np.asarray(b) - np.asarray(a)))) / denom
      )
    worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:4]
    print(f"HOIST_DIFFS {self._testMethodName} loss={float(l0):.6f} worst={worst}")
    np.testing.assert_allclose(lp1, lp0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(en1, en0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(l1, l0, rtol=1e-5, atol=1e-4)
    flat0 = jax.tree_util.tree_leaves_with_path(g0)
    flat1 = jax.tree_util.tree_leaves(g1)
    names = []
    for (path, a), b in zip(flat0, flat1):
      name = jax.tree_util.keystr(path)
      names.append(name)
      scale = float(np.max(np.abs(a))) or 1.0
      np.testing.assert_allclose(
          np.asarray(b), np.asarray(a), rtol=1e-4, atol=1e-5 * scale, err_msg=name
      )
    # The comparison covers the head (kernel, bias) and an upstream layer.
    self.assertTrue(any("lm_head" in n and "kernel" in n for n in names))
    self.assertTrue(any("layers" in n for n in names))
    head_grad = [a for (p, a) in flat0 if "lm_head" in jax.tree_util.keystr(p)]
    self.assertGreater(float(np.max(np.abs(head_grad[0]))), 0.0)

  def test_matches_chunked_reduce_scatter(self):
    # TUNIX_LOGPS_HOIST_LM_HEAD_RS: the head enters sharded and is gathered
    # inside the region (gradient reduce-scattered once); same numerics.
    os.environ[common.LOGPS_HOIST_LM_HEAD_RS_ENV] = "1"
    try:
      self.assertTrue(common.logps_hoist_lm_head_rs_default())
      self.test_matches_chunked_b4_fsdp4_ctx2()  # pytype: disable=attribute-error
      self.test_matches_chunked_b2_fsdp4_ctx2()  # pytype: disable=attribute-error
    finally:
      os.environ.pop(common.LOGPS_HOIST_LM_HEAD_RS_ENV, None)
    self.assertFalse(common.logps_hoist_lm_head_rs_default())

  def test_head_leaf_detection(self):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state)
    leaves = [x for _, x in leaves_with_path]
    used = common._lm_head_leaf_mask(  # pylint: disable=protected-access
        graphdef, leaves, treedef, 16, jnp.float32
    )
    picked = sorted(
        jax.tree_util.keystr(p) for (p, _), u in zip(leaves_with_path, used) if u
    )
    self.assertLen(picked, 2)
    self.assertTrue(all("lm_head" in p for p in picked), picked)

  def test_env_default(self):
    os.environ.pop(common.LOGPS_HOIST_LM_HEAD_ENV, None)
    self.assertFalse(common.logps_hoist_lm_head_default())
    os.environ[common.LOGPS_HOIST_LM_HEAD_ENV] = "1"
    try:
      self.assertTrue(common.logps_hoist_lm_head_default())
    finally:
      os.environ.pop(common.LOGPS_HOIST_LM_HEAD_ENV, None)


if __name__ == "__main__":
  absltest.main()
