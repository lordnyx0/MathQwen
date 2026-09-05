"""Test GGUFWriter for Qwen 3.5 architecture."""
import os
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

def test_writer():
    out_path = "models/test_qwen.gguf"
    writer = GGUFWriter(out_path, "qwen35")

    # Add metadata
    writer.add_block_count(2)
    writer.add_context_length(4096)
    writer.add_embedding_length(5120)
    writer.add_feed_forward_length(17408)
    writer.add_head_count(24)
    writer.add_head_count_kv(4)
    writer.add_layer_norm_rms_eps(1e-6)

    # Add dummy tensor
    w = np.random.randn(5120, 5120).astype(np.float32)
    writer.add_tensor("token_embd.weight", w, raw_dtype=GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully wrote test GGUF file: {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")
    if os.path.exists(out_path):
        os.remove(out_path)

if __name__ == "__main__":
    test_writer()
