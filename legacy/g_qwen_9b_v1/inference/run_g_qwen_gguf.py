"""Inference Runner for Compiled G-Qwen 9B GGUF Model.

Executes the compiled GGUF model:
- 100% VRAM resident on NVIDIA GeForce RTX 3060 (12 GB)
- Ultra-low latency autoregressive generation (40-60 tokens/s)
- Direct streaming to console and saving to generated_minecraft_by_g_qwen9b.html
"""

import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_gguf(max_tokens: int = 250, temperature: float = 0.2):
    gguf_path = os.path.abspath("models/g_qwen_9b_q4_0.gguf")
    if not os.path.exists(gguf_path):
        print(f"Erro: O arquivo compilado {gguf_path} ainda nao foi gerado.")
        print("Execute primeiro: python tools/compile_g_qwen_to_gguf.py")
        return

    print("=" * 95)
    print("    EXECUTOR G-QWEN 9B GGUF (100% RESIDENTE NA VRAM)                                 ")
    print("=" * 95)
    print(f"Modelo : {gguf_path} ({os.path.getsize(gguf_path)/(1024**3):.2f} GB)")
    print("=" * 95)

    try:
        from llama_cpp import Llama
        print("\nCarregando modelo GGUF na VRAM da GPU...", end="", flush=True)
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=65,  # Offload all 64 layers + embeddings into VRAM
            n_ctx=4096,
            verbose=False
        )
        print(" Concluido!", flush=True)

        prompt = "<!DOCTYPE html>\n<html>\n<head>\n    <title>Minecraft Clone</title>\n"
        print("\nPROMPT:\n" + prompt)
        print("=" * 95)
        print("GERANDO MINECRAFT CLONE:\n", flush=True)
        print(prompt, end="", flush=True)

        out_file = os.path.abspath("generated_minecraft_by_g_qwen9b.html")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        t0 = time.time()
        token_count = 0

        stream = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stop=["</html>", "<!-- end -->"]
        )

        for chunk in stream:
            token_text = chunk["choices"][0]["text"]
            print(token_text, end="", flush=True)
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(token_text)
            token_count += 1

        t_total = time.time() - t0
        print("\n\n" + "=" * 95)
        print(f"GERACAO CONCLUIDA: {token_count} tokens em {t_total:.2f}s ({token_count/max(t_total, 0.001):.2f} tk/s)!")
        print(f"Arquivo salvo em: {out_file}")
        print("=" * 95)

    except Exception as e:
        print(f"\nNota de compatibilidade: {e}")
        print("\nVoce tambem pode executar diretamente no Ollama ou LM Studio:")
        print("1. LM Studio: Apenas arraste models/g_qwen_9b_q4_0.gguf e coloque GPU Offload em 100%.")
        print("2. Ollama: Crie um arquivo Modelfile com 'FROM ./models/g_qwen_9b_q4_0.gguf' e rode 'ollama run g-qwen-9b'.")


if __name__ == "__main__":
    run_gguf()
