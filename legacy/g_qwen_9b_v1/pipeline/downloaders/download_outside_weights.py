"""Background Downloader for Qwen 3.8 27B-FP8 Outside Weights & Remaining Layers.

Downloads:
1. outside.safetensors (6.0 GB: embed_tokens, lm_head, norm)
2. Remaining layer files (layers-16 to layers-63)
With resume support and progress logging.
"""

import os
import sys
import time
from huggingface_hub import hf_hub_download


def download_outside():
    repo_id = "Qwen/Qwen3.8-27B-FP8"
    filename = "outside.safetensors"
    print("=" * 90, flush=True)
    print(f"Iniciando download de {filename} do repositorio {repo_id}...", flush=True)
    print("Tamanho esperado: ~6.0 GB (contem embed_tokens, lm_head, norm)", flush=True)
    print("=" * 90, flush=True)

    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
        )
        t_total = time.time() - t0
        print("\n" + "=" * 90, flush=True)
        print(f"[SUCESSO] Download concluido em {t_total:.2f} segundos ({t_total/60:.2f} minutos)!", flush=True)
        print(f"Arquivo salvo em: {path}", flush=True)
        print(f"Tamanho no disco: {os.path.getsize(path) / (1024**3):.2f} GB", flush=True)
        print("=" * 90, flush=True)
    except Exception as e:
        print(f"\n[ERRO no download]: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    download_outside()
