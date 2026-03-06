# Project Narrative: kernel-builder

## Summary
We're building a multi-backend kernel generation and management system that turns templates into optimized operator kernels across Metal, CUDA, SYCL, XPU, Neuron, and NKI. The goal is to make it trivial to spawn new kernel repos with automatic build tooling, run benchmarks, and upload validated kernels to the Hub.

## Current Foci
- **Metal GPU Hardened & CI-Validated**: Encoder lifecycle fixed (using PyTorch's stream interface), Metal standard targeting corrected (metal3.1 for macOS 14 best-effort, metal3.2 for macOS 15+, metal4.0 for macOS 26). GitHub Actions CI matrix deployed—validates Metal kernels across macOS 14 (limited), 15, and 26 on every PR. MPS encoder pattern now consistent across all examples.
- **Infrastructure-as-Code Ready**: Terraform scripts deployed for provisioning self-hosted runners with Nix environment isolation. CI matrix scales horizontally without manual configuration. Self-hosted runners enable private hardware testing for vLLM and commercial backends.
- **Backend Consolidation Complete**: All six backends (Metal, CUDA, SYCL, XPU, Neuron, NKI) now have Python bindings, CMake variants, and template examples. Neuron and NKI recently integrated with working Python support; focus now on encouraging real-world hardware validation from users.
- **Build Metadata & Arch Handling**: GPU architectures captured in metadata.json and flow through build variants automatically. No-arch kernel support allows backends without architecture constraints to build without Nix. Build system seamlessly handles both arch-specific and arch-agnostic kernels.

## How It Works
**Four-layer architecture:**

1. **Template Layer** (`builder/templates/`): Per-backend templates (Metal, CUDA, SYCL, etc.) with kernel stubs, CMake preambles, and torch-extension setup. `init` pulls a modern template repo and scaffolds a new kernel project locally.

2. **Build System** (`build2cmake/`): Rust-based transpiler that reads `build.toml` (kernel descriptor) and emits backend-specific CMake. Handles kernel variants (different dtypes, architectures), Python bindings, benchmarks. Uses Nix for reproducible builds. GPU architectures captured in metadata.json and flow through build variants automatically. Supports no-arch kernels for backends without GPU constraints (decouples build from Nix for simpler CI).

3. **CLI & Hub Integration** (`kernels/`): Python CLI for creating kernels, benchmarking against reference implementations, uploading to Hub with metadata cards. Manages lockfiles and kernel repos. Supports local kernel repo redirection via `kernels link` for development workflows.

4. **CI Infrastructure** (`terraform/` + `.github/workflows/`): Terraform scripts for provisioning self-hosted runners with Nix environment isolation. GitHub Actions matrix validates kernels across multiple macOS versions (14, 15, 26) on every PR. Infrastructure-as-code for reproducible CI setup.

**Flow**: User runs `kernels init` → scaffolds repo with template → edits kernel code → runs `kernels benchmark` → uploads with `kernels skills add`. Hub integration uses HF API to publish both the kernel package and model cards. CI automatically validates across target platforms.

## The Story So Far
We started with a single-backend (probably CUDA-centric) approach and have been progressively:

1. **Generalized the build system** (2024-2025): Extracted backend-specific logic into templates, built the CMake transpiler to generate correct configs for each target without hand-written backend files.

2. **Expanded backends steadily**: CUDA → SYCL (Intel GPUs, CPUs via DPC++) → XPU (Intel oneAPI) → Metal (Apple Silicon, with the recent encoder/version fixes) → Neuron, NKI (newest, still hardening).

3. **Professionalized the CLI**: Started basic, now it's feature-complete with redirection support (linking local kernel repos into projects), named conventions enforced in `init`, benchmarking with graphics output, comprehensive docs.

4. **Hardened Metal specifically** (March 2026): Discovered that direct encoder creation (`[cmdBuf computeCommandEncoder]`) breaks PyTorch's kernel coalescing and causes crashes on sequential calls. Solution: use `stream→commandEncoder()` and commit with `SyncType::COMMIT_AND_CONTINUE`. Fixed Metal standard version targeting (was trying to use metal4.0 which requires nonexistent macOS 26). Deployed GitHub Actions matrix to validate kernels across macOS 14 (Metal 3.1, best-effort), macOS 15 (Metal 3.2), and macOS 26 (Metal 4.0)—now running on every PR. Pattern consolidated across all MPS encoder examples.

5. **Structured the build metadata** (recent): GPU architectures captured in metadata.json, flow through build variants automatically. No-arch kernel support decouples build from Nix dependency, allowing simpler CI for backends without GPU variants.

## Dragons & Gotchas
- **Metal encoder lifecycle**: Don't create encoders directly from the command buffer. Use PyTorch's stream interface (via `stream→commandEncoder()`) or kernels will crash on second call. Commit with `SyncType::COMMIT_AND_CONTINUE`. This pattern is now standardized across all examples.
- **macOS 14 support is best-effort**: Metal 3.1 works on macOS 14 but with known limitations and less coverage testing. Prefer macOS 15 (Metal 3.2) for production kernels. macOS 14 CI runs but some edge cases may not be caught.
- **Metal standard version vs. macOS version**: `-std metal4.0` produces AIR v28, requiring macOS 26. Use `metal3.2` (AIR v27, macOS 15+) as default unless benchmarking a specific compiler feature. The `-mmacosx-version-min` flag doesn't control this; the Metal standard does.
- **Dtype support per backend**: Metal doesn't have bf16 in all kernel types, CUDA has native int8 but Metal doesn't. Document early or face surprises during porting.
- **GitHub Actions label context**: In `pull_request` triggers, use `github.event.pull_request.labels.*.name`, not `github.labels.*.name`. The latter doesn't exist—silent failures in conditional job logic.
- **vLLM kernel API differences**: If porting kernels for vLLM integration, the API surface (e.g., `swap_blocks` now requires block_size_in_bytes, paged_attention variants differ) is fragile. Check compatibility early.
- **Nix on self-hosted runners**: Wrap Python calls in `nix develop` to ensure proper environment isolation; without it, build environments bleed between runs.

## vLLM Metal Integration (March 2026)
We're actively integrating Hub Metal kernels (paged-attention, rotary-embedding, fused-rms-norm) into vLLM's MPS platform backend for Apple Silicon inference. This work validates that kernel-builder's ecosystem works end-to-end.

**Current status:**
- ✅ MPS platform backend created (device detection, memory queries, dtype support)
- ✅ Metal attention backend working (paged gather + SDPA, native PyTorch ops)
- ✅ MPS worker and model runner implemented (handles unified memory, lazy evaluation)
- ✅ Smoke test passing (distilgpt2 with dummy weights)
- 🔄 **E2E validation in progress**: Running Qwen2-7B inference with BF16/FP16, benchmarking throughput/latency
- 🚧 Benchmarking harness created (vLLM vs llama.cpp comparison)

**Key findings:**
- MPS lazy evaluation requires careful sync handling (torch.mps.synchronize() after warmup, but not per-layer)
- Unified memory model differs from CUDA—GPU→CPU transfers need special handling (cached CPU tensors, Event-based sync)
- Pre-allocation of buffers (seq_lens, query_start_loc) as CPU copies avoids per-sequence GPU→CPU sync

## Open Questions
- **Neuron & NKI field validation**: These backends are now merged and have Python bindings + CMake support, but lack real-world testing on actual hardware. Worth prioritizing once users start onboarding.
- **Paged vs. contiguous KV for Metal**: Some kernels (metal-flash-sdpa) work great for prefill but struggle with paged KV cache (needed for vLLM decode). Should we design two-path kernels (contiguous prefill + paged decode) or commit to one architecture?
- **vLLM Metal performance baseline**: What is competitive throughput vs llama.cpp Metal? Early results show MPS backend is stable but need benchmarks to confirm performance is acceptable for production.
- **Hub schema stability**: Model cards now include repo_id and proper formatting, but are the schemas stable enough for reliable indexing and serving by the Hub?
