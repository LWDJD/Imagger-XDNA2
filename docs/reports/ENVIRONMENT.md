# Environment

Exact paths and versions this was built and run against.  All of them were
found on the machine; none were downloaded except the `mlir_aie` wheel.

## Machine

    AMD Ryzen AI 9 H 365 w/ Radeon 880M, 20 cores, 32 GB
    Windows 11 Home China (build 26200.9457)
    NPU: Strix (XDNA2 / AIE2P),  BDF 0064:00:01.1

## Toolchain

| what | version | where |
|---|---|---|
| mlir_aie | 1.4.4 (`mlir_aie-1.4.4-cp312-cp312-win_amd64.whl`) | `_pylibs\mlir_aie\` |
| llvm-aie / Peano | 1.8.0 (bundled) | `_pylibs\mlir_aie\bin\` |
| aiecc / aie-opt / aie-translate / bootgen | — | `_pylibs\mlir_aie\bin\*.exe` |
| XRT | 2.21.0 | see below |
| NPU driver | 32.0.20102.3930 | — |
| NPU firmware | 1.1.2.64 | — |

## Python interpreters (two are needed on purpose)

| role | version | path |
|---|---|---|
| **compile** (mlir_aie, cp312) | 3.12 | `<python312>\python.exe` |
| **run** (pyxrt) | **3.10.9** | `<python310>\python.exe` |

`pyxrt.pyd` links `python310.dll`, so the run side MUST be a 3.10 interpreter.
Two complete 3.10 installs exist on this machine:

    <python310>\python.exe
    <user>\AppData\Roaming\an IDE\ModularData\ai-agent\vm\tools\python\python.exe

Neither path contains "310"/"3.10" text, which is why ordinary searches miss
them.

## XRT / pyxrt

    C:\Windows\System32\xrt_coreutil.dll
    C:\Windows\System32\AMD\pyxrt.pyd
    C:\Program Files\RyzenAI\1.6.0\xrt\xrt_coreutil.dll
    <xrt>\
        pyxrt.pyd   xrt_core.dll   xrt_coreutil.dll   xrt-smi.exe

The last folder is the one that works as a unit.  Put it on `sys.path` and
`os.add_dll_directory()` before importing `pyxrt`.

Verify:

    xrt-smi.exe examine
    -> |[0064:00:01.1]|NPU Strix|

## mlir_aie install

Not on PyPI.  Downloaded from the GitHub release and unzipped (pip's unpack
step is blocked by the sandbox on this machine):

    https://github.com/Xilinx/mlir-aie/releases/download/v1.4.4/mlir_aie-1.4.4-cp312-cp312-win_amd64.whl

Unzip into a directory and put BOTH the directory itself and
`<dir>\mlir_aie\python` on `sys.path` (the wheel ships an `aie.pth` pointing at
the latter).

## Kernel sources (shipped in the wheel)

    _pylibs\mlir_aie\include\aie_kernels\linalg\mm_bfp.cc          BFP16 GEMM
    _pylibs\mlir_aie\include\aie_api\detail\aie2p\mmul_bfp16_bfp16.hpp
    _pylibs\mlir_aie\include\aie_kernels\activation\{gelu,tanh,softmax,silu,sigmoid}.cc
    _pylibs\mlir_aie\python\aie\utils\bfp.py                       numpy codec

## Reference implementations worth knowing about

    _pylibs\quark\onnx\operators\custom_ops\src\bfp\cpu\bfp_kernel.cc
        AMD's own BFP16 codec -- confirms the exponent and rounding rules
    _pylibs\quark\onnx\operators\custom_ops\src\custom_op_library.cc
        the Ort::CustomOpBase registration pattern, domain "com.amd.quark"
    onnxruntime_vitis_ai_custom_ops.dll
        AMD's NPU custom ops (custom_op_mm.cpp, custom_op_rope.cpp, ...);
        PE strings show they "run AIE kernel" through xrt_coreutil.dll

## Gotchas

* The DSH sandbox temp dir `<temp>\<temp>` is
  cleaned up by something else on this machine; every command fails with
  `windows-acl-run: --temp is not an existing directory` until it is recreated.
* ConstrainedLanguage mode blocks `[Console]::OutputEncoding`, `.ToString()`,
  `[System.IO.*]`, `Get-CimInstance`, and `.Substring` -- do that work in
  Python instead.
* `safetensors.get_tensor()` raises an access violation here; read tensors by
  parsing the header offsets and using `np.frombuffer`.
* `wgt_formatting` refuses BFP16:
  `Unsupported OP type for weight formatting: MatMul_bfp16xbfp16xbfp16`.
