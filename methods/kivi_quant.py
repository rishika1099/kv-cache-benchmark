import torch
from .base import MethodWrapper


def quantize_per_channel(tensor, bits):
    """
    Quantize along head_dim (channel) dimension.
    tensor shape: (batch, heads, seq_len, head_dim)
    """
    min_val = tensor.min(dim=-1, keepdim=True).values
    max_val = tensor.max(dim=-1, keepdim=True).values
    scale = (max_val - min_val) / (2 ** bits - 1)
    scale = scale.clamp(min=1e-8)
    zero = min_val
    quantized = ((tensor - zero) / scale).round().clamp(0, 2 ** bits - 1)
    return (
        quantized.to(torch.uint8),
        scale.to(torch.float16),
        zero.to(torch.float16),
    )


def quantize_per_token(tensor, bits):
    """
    Quantize along seq_len (token) dimension.
    tensor shape: (batch, heads, seq_len, head_dim)
    """
    min_val = tensor.min(dim=-2, keepdim=True).values
    max_val = tensor.max(dim=-2, keepdim=True).values
    scale = (max_val - min_val) / (2 ** bits - 1)
    scale = scale.clamp(min=1e-8)
    zero = min_val
    quantized = ((tensor - zero) / scale).round().clamp(0, 2 ** bits - 1)
    return (
        quantized.to(torch.uint8),
        scale.to(torch.float16),
        zero.to(torch.float16),
    )


def dequantize(quantized, scale, zero):
    return quantized.to(torch.float16) * scale + zero


class KIVIMethod(MethodWrapper):
    """
    KIVI: Asymmetric KV cache quantization.

    Inspired by: KIVI (Liu et al., ICML 2024) —
    "A Tuning-Free Asymmetric 2bit Quantization for KV Cache"

    Keys have outliers along channel dimension → quantize per-channel.
    Values have outliers along token dimension → quantize per-token.
    Keep most recent `residual_length` tokens in FP16 (sliding window).
    Pure PyTorch — no Triton/CUDA kernels.
    """

    def __init__(self, bits=4, residual_length=128):
        self.bits = bits
        self.residual_length = residual_length
        self.cache = {}  # layer_idx -> dict with quantized state

    def process_prefill(self, past_key_values, attention_weights=None):
        self.cache = {}
        result = []

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0].half(), layer_kv[1].half()
            # k/v shape: (batch, heads, seq_len, head_dim)

            seq_len = k.shape[2]

            if seq_len <= self.residual_length:
                # Entire sequence fits in residual — keep as FP16, nothing to quantize
                res_k = k.to(torch.float16)
                res_v = v.to(torch.float16)
                self.cache[layer_idx] = {
                    'quantized_k': None, 'scale_k': None, 'zero_k': None,
                    'quantized_v': None, 'scale_v': None, 'zero_v': None,
                    'residual_k': res_k,
                    'residual_v': res_v,
                }
                result.append((res_k, res_v))
                continue

            # Split into historical and residual
            hist_k = k[:, :, :-self.residual_length, :]
            hist_v = v[:, :, :-self.residual_length, :]
            res_k = k[:, :, -self.residual_length:, :].to(torch.float16)
            res_v = v[:, :, -self.residual_length:, :].to(torch.float16)

            # Quantize historical: keys per-channel, values per-token
            q_k, s_k, z_k = quantize_per_channel(hist_k, self.bits)
            q_v, s_v, z_v = quantize_per_token(hist_v, self.bits)

            self.cache[layer_idx] = {
                'quantized_k': q_k, 'scale_k': s_k, 'zero_k': z_k,
                'quantized_v': q_v, 'scale_v': s_v, 'zero_v': z_v,
                'residual_k': res_k,
                'residual_v': res_v,
            }

            # Reconstruct for the model: dequantized historical + residual
            deq_k = dequantize(q_k, s_k, z_k)
            deq_v = dequantize(q_v, s_v, z_v)
            reconstructed_k = torch.cat([deq_k, res_k], dim=2)
            reconstructed_v = torch.cat([deq_v, res_v], dim=2)
            result.append((reconstructed_k, reconstructed_v))

        return tuple(result)

    def process_step(self, past_key_values, step, attention_weights=None):
        result = []

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0], layer_kv[1]
            # HF appends the new token to past_key_values — last position is new token

            if layer_idx not in self.cache:
                # Fallback: no cache state, return as-is
                result.append((k, v))
                continue

            state = self.cache[layer_idx]
            # The new token is the last position in k/v
            new_k = k[:, :, -1:, :].to(torch.float16)
            new_v = v[:, :, -1:, :].to(torch.float16)

            # Append new token to residual
            res_k = torch.cat([state['residual_k'], new_k], dim=2)
            res_v = torch.cat([state['residual_v'], new_v], dim=2)

            # If residual exceeds residual_length, move oldest token to quantized cache
            if res_k.shape[2] > self.residual_length:
                overflow_k = res_k[:, :, :1, :].half()
                overflow_v = res_v[:, :, :1, :].half()
                res_k = res_k[:, :, 1:, :]
                res_v = res_v[:, :, 1:, :]

                if state['quantized_k'] is None:
                    # Initialize quantized cache with the overflow token
                    q_k, s_k, z_k = quantize_per_channel(overflow_k, self.bits)
                    q_v, s_v, z_v = quantize_per_token(overflow_v, self.bits)
                else:
                    # Dequantize existing, append overflow, re-quantize
                    existing_k = dequantize(
                        state['quantized_k'], state['scale_k'], state['zero_k']
                    ).half()
                    existing_v = dequantize(
                        state['quantized_v'], state['scale_v'], state['zero_v']
                    ).half()
                    combined_k = torch.cat([existing_k, overflow_k], dim=2)
                    combined_v = torch.cat([existing_v, overflow_v], dim=2)
                    q_k, s_k, z_k = quantize_per_channel(combined_k, self.bits)
                    q_v, s_v, z_v = quantize_per_token(combined_v, self.bits)

                state['quantized_k'] = q_k
                state['scale_k'] = s_k
                state['zero_k'] = z_k
                state['quantized_v'] = q_v
                state['scale_v'] = s_v
                state['zero_v'] = z_v

            state['residual_k'] = res_k
            state['residual_v'] = res_v

            # Reconstruct full KV for model
            if state['quantized_k'] is not None:
                deq_k = dequantize(state['quantized_k'], state['scale_k'], state['zero_k'])
                deq_v = dequantize(state['quantized_v'], state['scale_v'], state['zero_v'])
                full_k = torch.cat([deq_k, res_k], dim=2)
                full_v = torch.cat([deq_v, res_v], dim=2)
            else:
                full_k = res_k
                full_v = res_v

            result.append((full_k, full_v))

        return tuple(result)

    def reset(self):
        self.cache = {}

    def get_kv_size_bytes(self, past_key_values):
        """
        Count: quantized tensor bytes (uint8) + scale/zero bytes (fp16) + residual (fp16).
        """
        total = 0
        for layer_idx, state in self.cache.items():
            if state['quantized_k'] is not None:
                for key in ('quantized_k', 'quantized_v'):
                    t = state[key]
                    total += int(t.numel() * self.bits / 8)
                for key in ('scale_k', 'zero_k', 'scale_v', 'zero_v'):
                    t = state[key]
                    total += t.numel() * t.element_size()
            for key in ('residual_k', 'residual_v'):
                t = state[key]
                total += t.numel() * t.element_size()

        # If cache is empty (fallback), count from past_key_values directly
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
    print(f"Loading {model_name} for KIVI smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = KIVIMethod(bits=4, residual_length=16)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print("KIVI smoke test PASSED")
