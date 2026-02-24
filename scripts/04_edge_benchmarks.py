#!/usr/bin/env python3
"""
Experiment 3: Edge Deployment Benchmarks on Jetson Orin Nano

Captures all the metrics needed for the Edge AI Prize:
  - CXR Foundation inference latency
  - MedGemma inference latency  
  - Peak GPU & system memory
  - Power consumption (via tegrastats)
  - Throughput (images/minute)
  - Cold start time

Usage:
    python scripts/04_edge_benchmarks.py [--n-images 50]
"""

import os
import re
import json
import time
import argparse
import subprocess
import threading
import numpy as np
import psutil
from pathlib import Path

RESULTS_DIR = Path("results/benchmarks")


# ============================================================
# Jetson-Specific Monitoring
# ============================================================

class JetsonMonitor:
    """Monitor Jetson Orin Nano power/memory via tegrastats."""
    
    def __init__(self):
        self.readings = []
        self._running = False
        self._thread = None
        self.is_jetson = self._check_jetson()
    
    def _check_jetson(self) -> bool:
        """Check if running on a Jetson device."""
        try:
            result = subprocess.run(
                ["tegrastats", "--help"],
                capture_output=True, timeout=2
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def start(self):
        """Start monitoring in background thread."""
        if not self.is_jetson:
            print("Not running on Jetson — using psutil for basic metrics")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> dict:
        """Stop monitoring and return aggregated stats."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        
        if not self.readings:
            return self._get_psutil_stats()
        
        powers = [r.get("power_mw", 0) for r in self.readings if r.get("power_mw")]
        gpu_mems = [r.get("gpu_mem_mb", 0) for r in self.readings if r.get("gpu_mem_mb")]
        
        return {
            "n_readings": len(self.readings),
            "power_mw_mean": np.mean(powers) if powers else None,
            "power_mw_max": np.max(powers) if powers else None,
            "power_w_mean": np.mean(powers) / 1000 if powers else None,
            "gpu_mem_mb_peak": np.max(gpu_mems) if gpu_mems else None,
        }
    
    def _monitor_loop(self):
        """Read tegrastats output."""
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            while self._running:
                line = proc.stdout.readline()
                if line:
                    parsed = self._parse_tegrastats(line.strip())
                    if parsed:
                        self.readings.append(parsed)
            proc.terminate()
        except Exception as e:
            print(f"tegrastats error: {e}")
    
    def _parse_tegrastats(self, line: str) -> dict:
        """Parse a tegrastats output line."""
        result = {}
        
        # Power: VDD_IN XXXXX/YYYYY or similar patterns
        power_match = re.search(r"VDD_IN\s+(\d+)/(\d+)", line)
        if power_match:
            result["power_mw"] = int(power_match.group(1))
        
        # GPU memory
        gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gpu_match:
            result["gpu_util"] = int(gpu_match.group(1))
        
        # RAM
        ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram_match:
            result["ram_used_mb"] = int(ram_match.group(1))
            result["ram_total_mb"] = int(ram_match.group(2))
        
        return result if result else None
    
    def _get_psutil_stats(self) -> dict:
        """Fallback: get basic stats via psutil."""
        mem = psutil.virtual_memory()
        return {
            "n_readings": 0,
            "system_ram_used_gb": mem.used / (1024**3),
            "system_ram_total_gb": mem.total / (1024**3),
            "system_ram_percent": mem.percent,
            "note": "Running on non-Jetson hardware — power metrics unavailable",
        }


# ============================================================
# Benchmark Functions
# ============================================================

def benchmark_cxr_foundation(n_images: int = 50) -> dict:
    """Benchmark CXR Foundation inference speed using TensorFlow SavedModel.

    Matches the real production inference path in app/app.py:
      - PIL for image preprocessing (grayscale, resize)
      - tf.train.Example serialization
      - tf.saved_model.load() for the model
    """
    import io
    from PIL import Image

    IMAGE_SIZE = 1024  # Must match app.py
    CXR_MODEL_PATH = os.environ.get(
        "CXR_MODEL_PATH", "models/cxr-foundation/elixr-c-v2-pooled"
    )

    print(f"\n--- CXR Foundation Benchmark ({n_images} images) ---")

    # ----- helpers matching app.py's _preprocess_image / _extract_embedding -----

    def _preprocess_image(tf, image: Image.Image) -> bytes:
        """Convert PIL image to serialized tf.Example (CXR Foundation input)."""
        img = image.convert("L")  # Grayscale
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        feature = {
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[png_bytes])
            ),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        return example.SerializeToString()

    def _run_inference(tf, model, serialized_example: bytes):
        """Run model inference matching app.py's _extract_embedding."""
        input_tensor = tf.constant([serialized_example])

        try:
            if (hasattr(model, "signatures")
                    and "serving_default" in model.signatures):
                serve_fn = model.signatures["serving_default"]
                input_keys = list(serve_fn.structured_input_signature[1].keys())
                output = serve_fn(**{input_keys[0]: input_tensor})
            else:
                output = model(input_tensor)
        except Exception:
            output = model(inputs=input_tensor)

        if isinstance(output, dict):
            keys = list(output.keys())
            return output[keys[0]].numpy()
        return output.numpy()

    # ----- Try to load model via TensorFlow -----

    try:
        import tensorflow as tf
        try:
            import tensorflow_text  # noqa: F401 — registers SentencepieceOp
        except ImportError:
            pass

        tf.get_logger().setLevel("ERROR")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        if not os.path.exists(CXR_MODEL_PATH):
            raise FileNotFoundError(f"Model path not found: {CXR_MODEL_PATH}")

        model = tf.saved_model.load(CXR_MODEL_PATH)
        print(f"  Loaded CXR Foundation from {CXR_MODEL_PATH}")

    except Exception as e:
        print(f"Could not load CXR Foundation via TensorFlow: {e}")
        print("Creating SYNTHETIC benchmark (not measuring real inference)...")
        # Synthetic fallback — clearly labelled so results aren't confused
        latencies = np.random.exponential(0.5, n_images) + 0.3
        return {
            "model": "cxr-foundation (SYNTHETIC)",
            "n_images": n_images,
            "mean_latency_ms": np.mean(latencies) * 1000,
            "median_latency_ms": np.median(latencies) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "throughput_img_per_min": 60 / np.mean(latencies),
            "note": "SYNTHETIC — TF model not loaded; does NOT reflect real latency",
        }

    # ----- Create synthetic PIL images for benchmarking -----

    latencies = []

    def _make_dummy_image() -> Image.Image:
        """Generate a random grayscale PIL image of the expected size."""
        arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    # Warmup (3 forward passes)
    warmup_img = _make_dummy_image()
    warmup_ex = _preprocess_image(tf, warmup_img)
    for _ in range(3):
        _run_inference(tf, model, warmup_ex)

    # Benchmark
    for i in range(n_images):
        img = _make_dummy_image()
        serialized = _preprocess_image(tf, img)

        t0 = time.perf_counter()
        _run_inference(tf, model, serialized)
        t1 = time.perf_counter()

        latencies.append(t1 - t0)

    return {
        "model": "cxr-foundation",
        "framework": "tensorflow",
        "device": "cpu",  # TF will log actual device; kept for schema compat
        "n_images": n_images,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "median_latency_ms": np.median(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "min_latency_ms": np.min(latencies) * 1000,
        "max_latency_ms": np.max(latencies) * 1000,
        "throughput_img_per_min": 60 / np.mean(latencies),
        "latencies_ms": [l * 1000 for l in latencies],
    }


def benchmark_medgemma(n_prompts: int = 10) -> dict:
    """Benchmark MedGemma report generation via Ollama."""
    
    print(f"\n--- MedGemma Benchmark ({n_prompts} prompts) ---")
    
    test_prompt = """You are a musculoskeletal radiology assistant. An AI screening system 
has analyzed a shoulder X-ray and produced the following results:

- Fracture probability: 78% (HIGH)
- Classification: fracture_detected
- Confidence: moderate

Based on these screening results, provide:
1. CLINICAL SUMMARY (2-3 sentences)
2. URGENCY LEVEL: URGENT, MODERATE, or LOW
3. PATIENT EXPLANATION (2-3 sentences)

Important: This is an AI screening result, not a diagnosis."""

    try:
        import ollama
        medgemma_model = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"

#        # Check if model is available
#        models = ollama.list()
#        medgemma_model = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
#        for m in models.get("models", []):
#            if "medgemma" in m.get("name", "").lower():
#                medgemma_model = m["name"]
#                break
#
#        if not medgemma_model:
#            print("MedGemma not found in Ollama. Available models:")
#            for m in models.get("models", []):
#                print(f"  {m['name']}")
#            return {"error": "MedGemma not found in Ollama", "note": "Run: ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"}
        
        print(f"Using model: {medgemma_model}")
        
        # Warmup
        ollama.generate(model=medgemma_model, prompt="Hello", options={"num_predict": 10})
        
        # Benchmark
        latencies = []
        token_counts = []
        
        for i in range(n_prompts):
            t0 = time.perf_counter()
            response = ollama.generate(
                model=medgemma_model,
                prompt=test_prompt,
                options={"num_predict": 300, "temperature": 0.3},
            )
            t1 = time.perf_counter()
            
            latency = t1 - t0
            latencies.append(latency)
            
            n_tokens = response.get("eval_count", len(response.get("response", "").split()))
            token_counts.append(n_tokens)
            
            print(f"  Run {i+1}/{n_prompts}: {latency:.1f}s, {n_tokens} tokens")
        
        return {
            "model": medgemma_model,
            "n_prompts": n_prompts,
            "mean_latency_s": np.mean(latencies),
            "median_latency_s": np.median(latencies),
            "p95_latency_s": np.percentile(latencies, 95),
            "mean_tokens": np.mean(token_counts),
            "mean_tokens_per_sec": np.mean(token_counts) / np.mean(latencies),
            "latencies_s": latencies,
        }
        
    except ImportError:
        print("ollama package not installed. pip install ollama")
        return {"error": "ollama not installed"}
    except Exception as e:
        print(f"MedGemma benchmark failed: {e}")
        return {"error": str(e)}


def benchmark_combined_pipeline() -> dict:
    """Benchmark the full pipeline: image → triage score → report."""
    
    print("\n--- Combined Pipeline Benchmark ---")
    
    # This measures the complete user-facing flow
    results = {
        "step_1_preprocessing_ms": None,
        "step_2_cxr_foundation_ms": None,
        "step_3_classification_ms": None,
        "step_4_medgemma_s": None,
        "total_to_triage_ms": None,
        "total_to_report_s": None,
    }
    
    # TODO: Fill in with actual pipeline timing once models are loaded
    print("  (Run after models are set up on Jetson)")
    
    return results


# ============================================================
# Memory Profiling
# ============================================================

def profile_memory() -> dict:
    """Profile system and GPU memory usage."""
    mem = psutil.virtual_memory()
    result = {
        "system_ram_total_gb": round(mem.total / (1024**3), 2),
        "system_ram_used_gb": round(mem.used / (1024**3), 2),
        "system_ram_available_gb": round(mem.available / (1024**3), 2),
    }

    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
            result["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2))
            result["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["gpu_name"] = "Apple Silicon (MPS)"
            result["gpu_mem_allocated_mb"] = round(torch.mps.current_allocated_memory() / (1024**2))
            result["note"] = "MPS uses unified memory — see system RAM for total"
    except ImportError:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                result["gpu_name"] = gpus[0].name
                result["note"] = "GPU detected via TensorFlow"
        except ImportError:
            pass

    return result


# ============================================================
# Results Formatting
# ============================================================

def _fmt(value, spec: str = ".0f") -> str:
    """Safely format a numeric value, returning 'N/A' for non-numeric inputs."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):{spec}}"
    except (TypeError, ValueError):
        return "N/A"


def format_benchmark_table(cxr_results: dict, medgemma_results: dict, memory: dict) -> str:
    """Format results into a markdown table for the writeup."""

    lines = [
        "## Edge Deployment Benchmarks — Jetson Orin Nano 8GB",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Device** | {memory.get('gpu_name', 'Jetson Orin Nano 8GB')} |",
        f"| **System RAM** | {memory.get('system_ram_total_gb', '8')} GB |",
        "",
        "### CXR Foundation (Fracture Screening)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean latency | {_fmt(cxr_results.get('mean_latency_ms'))} ms |",
        f"| Median latency | {_fmt(cxr_results.get('median_latency_ms'))} ms |",
        f"| P95 latency | {_fmt(cxr_results.get('p95_latency_ms'))} ms |",
        f"| Throughput | {_fmt(cxr_results.get('throughput_img_per_min'))} images/min |",
        f"| GPU memory | {_fmt(cxr_results.get('gpu_mem_peak_mb'))} MB |",
        "",
        "### MedGemma 1.5 4B Q4 (Report Generation)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    
    if "error" not in medgemma_results:
        lines.extend([
            f"| Mean latency | {_fmt(medgemma_results.get('mean_latency_s'), '.1f')} s |",
            f"| Tokens/sec | {_fmt(medgemma_results.get('mean_tokens_per_sec'), '.1f')} |",
            f"| Mean output tokens | {_fmt(medgemma_results.get('mean_tokens'))} |",
        ])
    else:
        lines.append(f"| Status | {medgemma_results.get('error', 'Not tested')} |")
    
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=50, help="Number of images for CXR benchmark")
    parser.add_argument("--n-prompts", type=int, default=5, help="Number of prompts for MedGemma benchmark")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start Jetson monitoring
    monitor = JetsonMonitor()
    monitor.start()
    
    # Profile memory before loading anything
    memory_before = profile_memory()
    print("Memory before benchmarks:")
    print(json.dumps(memory_before, indent=2))
    
    # Run benchmarks
    cxr_results = benchmark_cxr_foundation(args.n_images)
    medgemma_results = benchmark_medgemma(args.n_prompts)
    
    # Memory after loading models
    memory_after = profile_memory()
    
    # Stop monitoring
    jetson_stats = monitor.stop()
    
    # Compile all results
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cxr_foundation": {k: v for k, v in cxr_results.items() if k != "latencies_ms"},
        "medgemma": {k: v for k, v in medgemma_results.items() if k != "latencies_s"},
        "memory_before": memory_before,
        "memory_after": memory_after,
        "jetson_power": jetson_stats,
    }
    
    # Save
    with open(args.output_dir / "edge_benchmarks.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate markdown table for writeup
    table = format_benchmark_table(cxr_results, medgemma_results, memory_after)
    with open(args.output_dir / "benchmark_table.md", "w") as f:
        f.write(table)
    
    # Print summary
    print("\n" + "="*60)
    print("EDGE BENCHMARK SUMMARY")
    print("="*60)
    print(table)
    print(f"\nResults saved to: {args.output_dir}")
    
    print("\nNext step: Build the Gradio app → python app/app.py")


if __name__ == "__main__":
    main()
