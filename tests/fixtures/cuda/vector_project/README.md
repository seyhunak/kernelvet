Test fixture: a small CUDA project used by KernelVet's migration tests.
It intentionally contains constructs HIPIFY cannot translate (inline PTX,
32-lane warp assumptions) so the analyzer and auditor have something to find.
