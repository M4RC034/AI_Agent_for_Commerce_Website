"""Retrieval components.

Import-order note (macOS / Apple Silicon)
    ``torch`` MUST be imported before ``faiss``. Both link their own copy of
    libomp, and when faiss wins the race the OpenMP runtime is initialised in a
    state that segfaults the first MPS tensor op — no exception, just SIGSEGV.
    Importing torch here, before any submodule pulls in faiss, pins the order
    for the whole package regardless of which module is imported first.
"""

import torch as _torch  # noqa: F401  (import for side effect: must precede faiss)
