"""Fast Concurrent Downloader for Qwen 3.8 27B-FP8 Remaining Layers (16..63).

Downloads all 48 missing layer safetensors files to complete the 64-layer backbone.
Supports resume and multi-threaded throughput.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download


def download_single_layer(layer_idx: int) -> Tuple_Layer:
    filename = f"layers-{layer_idx}.safetensors"
    repo_id = "Qwen/Qwen3.8-27B-FP8"
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        elapsed = time.time() - t0
        return layer_idx, True, size_mb, elapsed, None
    except Exception as e:
        return layer_idx, False, 0.0, time.time() - t0, str(e)


Tuple_Layer = tuple[int, bool, float, float, str | None]


def download_all_remaining():
    layers_to_download = list(range(16, 64))
    total_files = len(layers_to_download)
    print("=" * 95, flush=True)
    print(f"Iniciando download das {total_files} camadas restantes (layers-16 a layers-63)...", flush=True)
    print("Volume total esperado: ~18.29 GB | Repositorio: Qwen/Qwen3.8-27B-FP8", flush=True)
    print("=" * 95, flush=True)

    t_start = time.time()
    completed_count = 0
    total_downloaded_mb = 0.0

    # 3 concurrent downloads for optimal throughput without hitting thread contention
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(download_single_layer, idx): idx for idx in layers_to_download}

        for future in as_completed(future_map):
            layer_idx, success, size_mb, elapsed, err = future.result()
            completed_count += 1
            if success:
                total_downloaded_mb += size_mb
                speed_mb_s = size_mb / max(elapsed, 0.001)
                percent = (completed_count / total_files) * 100
                print(f"[{completed_count:>2}/{total_files}] layers-{layer_idx}.safetensors baixado ({size_mb:.1f} MB em {elapsed:.1f}s, {speed_mb_s:.1f} MB/s) - Progresso: {percent:.1f}%", flush=True)
            else:
                print(f"[{completed_count:>2}/{total_files}] [ERRO] layers-{layer_idx}.safetensors: {err}", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 95, flush=True)
    print(f"[CONCLUIDO] {completed_count}/{total_files} camadas baixadas com sucesso!", flush=True)
    print(f"Volume Total: {total_downloaded_mb / 1024:.2f} GB | Tempo Total: {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)
    print(f"Velocidade Media Global: {total_downloaded_mb / max(total_time, 0.001):.1f} MB/s", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    download_all_remaining()
