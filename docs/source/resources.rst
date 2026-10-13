Resource profiles
=================

``intermine314.export.resource_profile`` defines advisory profiles ``cpu16``
(the default), ``cpu8``, ``cpu6``, ``cpu4`` and ``cpu2``. ``tor_low_mem`` remains
available. Automatic CPU allocations respect process affinity and discoverable
Linux container quotas. Explicit HTTP worker overrides remain separate from
native CPU allocations.

Stage defaults
--------------

======================= ===== ===== ==== ==== ====
Stage                   cpu16 cpu8  cpu6 cpu4 cpu2
======================= ===== ===== ==== ==== ====
Public HTTP             4     4     4    4    2
Low-latency HTTP         2     2     2    2    2
High-latency HTTP        8     8     6    4    2
Arrow CPU               4     4     3    2    1
Arrow I/O               2     2     2    2    1
Ordered COPY            4     4     3    2    1
Unordered COPY          8     8     6    4    2
DuckDB SQL              16    8     6    4    2
Polars                  12    8     6    4    2
Python processes        4     4     3    2    1
Metadata                4     4     4    4    2
Mutable resource owner  1     1     1    1    1
======================= ===== ===== ==== ==== ====

These are stage allocations, not simultaneously allocated pools. The memory
planning envelope defaults to the smaller of 4 GiB and 25% of available memory,
including discoverable container limits; the portable fallback is 512 MiB.
The shared queue allowance is the smaller of 64 MiB and one eighth of the
envelope. Explicit ``memory_budget_bytes`` and queue settings override defaults.
Neither Arrow buffer bytes nor a database memory limit guarantees process RSS.

Set ``INTERMINE314_RESOURCE_PROFILE`` to select a named profile, or construct a
``ResourceProfile`` explicitly. Explicit profiles take precedence over that
environment variable. Existing ``INTERMINE314_TEMP_DIR`` behavior is preserved.
Counts must be positive integers; booleans and fractional values are rejected.

Integration status
------------------

Task 2.1 introduces and validates profile definitions. Production entry-point
integration and shared engine admission follow in tasks 2.2 and 2.3.
