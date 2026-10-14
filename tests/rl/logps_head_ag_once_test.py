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

"""compute_chunked_logps with a model that gathers its output head once.

The model's head kernel is sharded on its embed dim over an 8-device mesh;
`gather_output_head` returns a copy whose head is constrained to replicated.
Log-probs, entropy and the gradients with respect to the head and the hidden
states must match the per-chunk path. Run with
XLA_FLAGS=--xla_force_host_platform_device_count=8 (the mesh tests skip on
fewer devices). This checks numerics only; where the collectives land is a
property of the compiled program, not of these values.
"""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.rl import common

_EMB = 16
_VOCAB = 40


class _Head(nnx.Module):
  """Final RMS norm and an [emb, vocab] projection, like MaxText's output head."""

  def __init__(self, mesh, hoist, rngs):
    self.mesh = mesh
    self.hoist = hoist
    self.scale = nnx.Param(
        1.0 + 0.1 * jax.random.normal(rngs.params(), (_EMB,), jnp.float32)
    )
    self.kernel = nnx.Param(
        jax.random.normal(rngs.params(), (_EMB, _VOCAB), jnp.float32) * 0.3
    )

  def compute_final_logits(self, hs):
    hs = hs * jax.lax.rsqrt(jnp.mean(hs * hs, axis=-1, keepdims=True) + 1e-6)
    return jnp.einsum("bte,ev->btv", hs * self.scale[...], self.kernel[...])

  def gather_output_head(self):
    if not self.hoist:
      return self
    graphdef, params, rest = nnx.split(self, nnx.Param, ...)
    replicated = NamedSharding(self.mesh, P())
    params = jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(x, replicated), params
    )
    return nnx.merge(graphdef, params, rest)


class LogpsHeadAgOnceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 8:
      self.skipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:8]), ("fsdp",))

  def _run(self, hoist, seq_len, chunk_size, return_entropy, temperature):
    model = _Head(self.mesh, hoist, nnx.Rngs(0))
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    params = jax.device_put(
        params,
        jax.tree.map(
            lambda x: NamedSharding(
                self.mesh, P("fsdp") if x.ndim == 2 else P()
            ),
            params,
        ),
    )
    rng = np.random.default_rng(1)
    batch = 8
    hs = jax.device_put(
        jnp.asarray(rng.normal(size=(batch, seq_len, _EMB)), jnp.float32),
        NamedSharding(self.mesh, P("fsdp")),
    )
    ids = jnp.asarray(rng.integers(0, _VOCAB, size=(batch, seq_len)), jnp.int32)
    w1 = jnp.asarray(rng.normal(size=(batch, seq_len)), jnp.float32)
    w2 = jnp.asarray(rng.normal(size=(batch, seq_len)), jnp.float32)

    def loss(p, h):
      m = nnx.merge(graphdef, p, rest)
      out = common.compute_chunked_logps(
          m, h, ids, temperature, chunk_size, return_entropy
      )
      if return_entropy:
        lp, en = out
        return jnp.sum(lp * w1) + 0.1 * jnp.sum(en * w2), (lp, en)
      return jnp.sum(out * w1), (out, None)

    (l, aux), grads = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    )(params, hs)
    return l, aux, grads

  @parameterized.product(
      seq_len_and_chunk=[(24, 8), (23, 8), (24, 5)],
      return_entropy=[True, False],
      temperature=[1.0, 0.7],
  )
  def test_matches_per_chunk_head(
      self, seq_len_and_chunk, return_entropy, temperature
  ):
    seq_len, chunk = seq_len_and_chunk
    ref = self._run(False, seq_len, chunk, return_entropy, temperature)
    got = self._run(True, seq_len, chunk, return_entropy, temperature)
    (l0, (lp0, en0), (gp0, gh0)) = ref
    (l1, (lp1, en1), (gp1, gh1)) = got
    tol = dict(rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lp1, lp0, **tol)
    if return_entropy:
      np.testing.assert_allclose(en1, en0, **tol)
    np.testing.assert_allclose(l1, l0, **tol)
    np.testing.assert_allclose(gh1, gh0, **tol)
    for a, b in zip(jax.tree.leaves(gp0), jax.tree.leaves(gp1)):
      self.assertGreater(float(jnp.max(jnp.abs(a))), 0.0)
      np.testing.assert_allclose(b, a, **tol)

  def test_hook_off_returns_same_model(self):
    model = _Head(self.mesh, False, nnx.Rngs(0))
    self.assertIs(model.gather_output_head(), model)


if __name__ == "__main__":
  absltest.main()
