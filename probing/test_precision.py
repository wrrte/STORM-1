"""Offline regression checks for the TITAN RTX precision exception."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sub_models.attention_blocks import ScaledDotProductAttention
from sub_models.precision import get_amp_dtype


class PrecisionTests(unittest.TestCase):
    def test_only_titan_rtx_selects_fp16(self):
        names = {
            "NVIDIA TITAN RTX": torch.float16,
            "TITAN RTX": torch.float16,
            "  nvidia titan rtx  ": torch.float16,
            "NVIDIA GeForce RTX 3090": torch.bfloat16,
            "NVIDIA RTX A6000": torch.bfloat16,
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": torch.bfloat16,
            "NVIDIA TITAN V": torch.bfloat16,
            "NVIDIA TITAN Xp": torch.bfloat16,
            "Tesla V100-SXM2-32GB": torch.bfloat16,
            "Unknown GPU": torch.bfloat16,
        }
        with patch("torch.cuda.is_available", return_value=True):
            for name, expected in names.items():
                with self.subTest(gpu=name), patch("torch.cuda.get_device_name", return_value=name) as get_name:
                    self.assertEqual(get_amp_dtype(), expected)
                    # Do not hard-code physical GPU 0 or cache another device's policy.
                    get_name.assert_called_once_with()

    def test_without_cuda_keeps_original_dtype(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.get_device_name") as get_name:
            self.assertEqual(get_amp_dtype(), torch.bfloat16)
            get_name.assert_not_called()

    def test_other_gpus_match_original_attention_and_gradients_exactly(self):
        for name in ("NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000"):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(gpu=name, dtype=dtype), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value=name):
                    module = ScaledDotProductAttention(temperature=2.0, attn_dropout=0.1)
                inputs = [torch.randn(2, 2, 4, 4, dtype=dtype, requires_grad=True) for _ in range(3)]
                reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
                mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
                rng_state = torch.get_rng_state()
                output, attention = module(*inputs, mask=mask)
                output.float().square().sum().backward()

                # The pre-change attention computation, including training dropout.
                torch.set_rng_state(rng_state)
                q, k, v = reference_inputs
                reference_attention = torch.matmul(q / 2.0, k.transpose(2, 3))
                reference_attention = reference_attention.masked_fill(mask == 0, -1e9)
                reference_attention = F.dropout(F.softmax(reference_attention, dim=-1), p=0.1, training=True)
                reference_output = torch.matmul(reference_attention, v)
                reference_output.float().square().sum().backward()

                self.assertTrue(torch.equal(output, reference_output))
                self.assertTrue(torch.equal(attention, reference_attention))
                for actual, reference in zip(inputs, reference_inputs):
                    self.assertTrue(torch.equal(actual.grad, reference.grad))

    def test_titan_fp16_mask_has_finite_outputs_and_gradients(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value="NVIDIA TITAN RTX"):
            module = ScaledDotProductAttention(temperature=2.0, attn_dropout=0.0)
        inputs = [torch.randn(2, 2, 4, 4, dtype=torch.float16, requires_grad=True) for _ in range(3)]
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
        output, attention = module(*inputs, mask=mask)
        output.float().square().sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(attention).all())
        self.assertTrue((attention.masked_select(~mask) == 0).all())
        for tensor in inputs:
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
