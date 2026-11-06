# SPDX-License-Identifier: MIT
"""The Z-Image-Turbo XDNA2 engine.

Nothing is imported here on purpose: the modules below are device-side and
import-heavy, and several of them are launched as scripts rather than imported.
Keeping this file empty is what makes ``python -m imaggers.paths`` work without
pulling torch into a process that only wants to print a path.
"""
