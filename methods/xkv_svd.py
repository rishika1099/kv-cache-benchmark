import torch
from .base import MethodWrapper


class XKVMethod(MethodWrapper):
    """
    xKV: Per-layer SVD low-rank approximation of KV cache.

    Inspired by: xKV (Chang et al., 2025) —
    "Cross-Layer SVD for KV-Cache Compression"
    Simplified to: per-layer SVD (group_size=1), no cross-layer grouping.

    CORE IDEA:
    KV tensors lie in a low-rank subspace. Store U, S, Vh instead of full K/V.
    Reconstruct approximate K/V at attention time: K_approx = U @ diag(S) @ Vh.
    Periodically recompute SVD as new tokens accumulate.

    NOTE ON LIMITATION:
    SVD recomputation at every recompute_interval steps adds latency overhead.
    This method trades throughput for memory. The SVD is applied to whatever key
    representation HuggingFace exposes in past_key_values (post-RoPE), since
    pre-RoPE states are not directly accessible through the HF interface.

    KNOWN TRADEOFF:
    Applying SVD post-RoPE introduces positional information into the decomposition,
    which may slightly reduce approximation quality compared to pre-RoPE SVD.
    """

    def __init__(self, rank_k=128, recompute_interval=50):
        self.rank_k = rank_k
        # Values need higher rank (less compressible than keys)
        self.rank_v = int(rank_k * 1.5)
        self.recompute_interval = recompute_interval
        self.cache = {}  # layer_idx -> dict with SVD state

    def _svd_compress(self, tensor, rank):
        """
        Compress tensor via truncated SVD, performed **per head**.
        tensor shape: (batch, heads, seq_len, head_dim)
        Returns: U_list, S_list, Vh_list (one per head, truncated to `rank`),
                 and original_shape tuple.

        Per-head SVD operates on (seq_len, head_dim) so that
        effective_rank = min(rank, seq_len, head_dim), giving real
        compression whenever rank < min(seq_len, head_dim).
        """
        batch, heads, seq_len, head_dim = tensor.shape
        t = tensor.squeeze(0).float()  # (heads, seq_len, head_dim)

        U_list, S_list, Vh_list = [], [], []

        for h in range(heads):
            mat = t[h]  # (seq_len, head_dim)
            effective_rank = min(rank, seq_len, head_dim)

            try:
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                U = U[:, :effective_rank]    # (seq_len, effective_rank)
                S = S[:effective_rank]       # (effective_rank,)
                Vh = Vh[:effective_rank, :]  # (effective_rank, head_dim)
            except Exception:
                # SVD can fail on degenerate matrices; keep full rank as fallback
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

            U_list.append(U.to(torch.float16))
            S_list.append(S.to(torch.float16))
            Vh_list.append(Vh.to(torch.float16))

        return U_list, S_list, Vh_list, (batch, heads, seq_len, head_dim)

    def _svd_reconstruct(self, U_list, S_list, Vh_list, original_shape):
        """Reconstruct approximated tensor from per-head SVD components."""
        batch, heads, seq_len, head_dim = original_shape
        device = U_list[0].device

        pieces = []
        for h in range(heads):
            # U: (seq_len, r), S: (r,), Vh: (r, head_dim)
            recon = (U_list[h].float() * S_list[h].float().unsqueeze(0)) @ Vh_list[h].float()
            pieces.append(recon.to(torch.float16))

        # Stack heads -> (heads, seq_len, head_dim), then unsqueeze batch
        return torch.stack(pieces, dim=0).unsqueeze(0)  # (1, heads, seq_len, head_dim)

    def process_prefill(self, past_key_values, attention_weights=None):
        self.cache = {}
        result = []

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0], layer_kv[1]
            seq_len = k.shape[2]
            head_dim = k.shape[3]

            # Only compress if we have enough tokens to make SVD meaningful
            min_tokens_for_svd = max(self.rank_k, 16)

            if seq_len < min_tokens_for_svd:
                # Too short — store raw, no SVD
                self.cache[layer_idx] = {
                    'U_k': None, 'S_k': None, 'Vh_k': None, 'shape_k': None,
                    'U_v': None, 'S_v': None, 'Vh_v': None, 'shape_v': None,
                    'residual_k': k.to(torch.float16),
                    'residual_v': v.to(torch.float16),
                    'use_svd': False,
                }
                result.append((k.to(torch.float16), v.to(torch.float16)))
                continue

            U_k, S_k, Vh_k, shape_k = self._svd_compress(k, self.rank_k)
            U_v, S_v, Vh_v, shape_v = self._svd_compress(v, self.rank_v)

            self.cache[layer_idx] = {
                'U_k': U_k, 'S_k': S_k, 'Vh_k': Vh_k, 'shape_k': shape_k,
                'U_v': U_v, 'S_v': S_v, 'Vh_v': Vh_v, 'shape_v': shape_v,
                'residual_k': torch.empty(
                    (1, k.shape[1], 0, k.shape[3]), dtype=torch.float16, device=k.device
                ),
                'residual_v': torch.empty(
                    (1, v.shape[1], 0, v.shape[3]), dtype=torch.float16, device=v.device
                ),
                'use_svd': True,
            }

            recon_k = self._svd_reconstruct(U_k, S_k, Vh_k, shape_k)
            recon_v = self._svd_reconstruct(U_v, S_v, Vh_v, shape_v)
            result.append((recon_k, recon_v))

        return tuple(result)

    def process_step(self, past_key_values, step, attention_weights=None):
        """
        Append new token to residual buffer.
        Periodically recompute SVD on combined (SVD reconstruction + residual).

        NOTE: SVD recomputation adds latency spikes at recompute_interval steps.
        This is the documented throughput-memory tradeoff for this method.
        """
        result = []

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0], layer_kv[1]

            if layer_idx not in self.cache or not self.cache[layer_idx]['use_svd']:
                result.append((k.to(torch.float16), v.to(torch.float16)))
                continue

            state = self.cache[layer_idx]

            # New token is last position in k/v
            new_k = k[:, :, -1:, :].to(torch.float16)
            new_v = v[:, :, -1:, :].to(torch.float16)

            # Append to residual
            state['residual_k'] = torch.cat([state['residual_k'], new_k], dim=2)
            state['residual_v'] = torch.cat([state['residual_v'], new_v], dim=2)

            # Periodic SVD recompute
            if (self.recompute_interval > 0 and
                    step > 0 and
                    step % self.recompute_interval == 0 and
                    state['residual_k'].shape[2] > 0):

                # Reconstruct full K from SVD + residual
                recon_k = self._svd_reconstruct(
                    state['U_k'], state['S_k'], state['Vh_k'], state['shape_k']
                )
                recon_v = self._svd_reconstruct(
                    state['U_v'], state['S_v'], state['Vh_v'], state['shape_v']
                )
                combined_k = torch.cat([recon_k, state['residual_k']], dim=2)
                combined_v = torch.cat([recon_v, state['residual_v']], dim=2)

                # Recompute SVD on combined
                U_k, S_k, Vh_k, shape_k = self._svd_compress(combined_k, self.rank_k)
                U_v, S_v, Vh_v, shape_v = self._svd_compress(combined_v, self.rank_v)

                state['U_k'] = U_k
                state['S_k'] = S_k
                state['Vh_k'] = Vh_k
                state['shape_k'] = shape_k
                state['U_v'] = U_v
                state['S_v'] = S_v
                state['Vh_v'] = Vh_v
                state['shape_v'] = shape_v

                # Clear residual after recompute
                state['residual_k'] = torch.empty(
                    (1, k.shape[1], 0, k.shape[3]), dtype=torch.float16, device=k.device
                )
                state['residual_v'] = torch.empty(
                    (1, v.shape[1], 0, v.shape[3]), dtype=torch.float16, device=v.device
                )

                recon_k_new = self._svd_reconstruct(U_k, S_k, Vh_k, shape_k)
                recon_v_new = self._svd_reconstruct(U_v, S_v, Vh_v, shape_v)
                result.append((recon_k_new, recon_v_new))
            else:
                # Return SVD reconstruction + residual buffer
                recon_k = self._svd_reconstruct(
                    state['U_k'], state['S_k'], state['Vh_k'], state['shape_k']
                )
                recon_v = self._svd_reconstruct(
                    state['U_v'], state['S_v'], state['Vh_v'], state['shape_v']
                )
                full_k = torch.cat([recon_k, state['residual_k']], dim=2)
                full_v = torch.cat([recon_v, state['residual_v']], dim=2)
                result.append((full_k, full_v))

        return tuple(result)

    def reset(self):
        self.cache = {}

    def get_kv_size_bytes(self, past_key_values):
        """Count: per-head U + S + Vh tensor bytes + residual bytes."""
        total = 0
        for layer_idx, state in self.cache.items():
            if state['use_svd']:
                # SVD components are lists of tensors (one per head)
                for list_key in ('U_k', 'S_k', 'Vh_k', 'U_v', 'S_v', 'Vh_v'):
                    tensor_list = state[list_key]
                    if tensor_list is not None:
                        for t in tensor_list:
                            total += t.numel() * t.element_size()
                for key in ('residual_k', 'residual_v'):
                    t = state[key]
                    total += t.numel() * t.element_size()
            else:
                for key in ('residual_k', 'residual_v'):
                    t = state[key]
                    total += t.numel() * t.element_size()

        # Fallback
        if not self.cache:
            for layer in past_key_values:
                k, v = layer[0], layer[1]
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark.runner import generate_with_method

    model_name = "facebook/opt-125m"
    print(f"Loading {model_name} for xKV SVD smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = XKVMethod(rank_k=16, recompute_interval=5)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print("xKV SVD smoke test PASSED")
