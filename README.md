# mlx-benchmark
TODO: rename

This is a small tool to benchmark inference performance with different models/settings
See --help for help

# Usage example
```bash
python mlx_benchmark.py \
--model mlx-community/Qwen3.5-9B-MLX-4bit \
--prompts 512 2048 4096 \
--generation 128 \
--kv-bits 4 none \
--prefill 512 1024 \
--repeats 3
```

```bash
python mlx_benchmark.py `
  --model "lmstudio-community/granite-4.2-8b-MLX-4bit" `
  --model "mlx-community/LFM2.5-8B-A1B-MLX-4bit" `
  --model "mlx-community/gpt-oss-20b-MXFP4" `
  --model "mlx-community/mistralai_Ministral-3-14B-Instruct-2512-MLX-MXFP4" `
  --model "mlx-community/Qwen3-14B-4bit" `
  --model "ibm-granite/granite-4.2-8b-q4-mlx" `
  --prompts 512 2048 `
  --generation 128 `
  --repeats 1 `
  --kv-bits none 4 `
  --prefill 512 1024 `
  --mtp none `
  --timeout 900 `
  --csv mlx_comparison_results.csv `
  --logs mlx_comparison_logs
```