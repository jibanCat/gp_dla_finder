# Performance: BLAS threads

More BLAS threads are not always faster for this likelihood. Its matrix
operations are only about **2 Mflop**, so the overhead of managing threads can
be as important as the matrix calculation itself. We therefore measured the
scaling on one laptop before choosing a recommendation.

Two sweeps on the same 10-core machine (OpenBLAS 0.3.23, order-balanced):

| BLAS threads | live pool, one process | fresh process per count |
|---|---|---|
| 1 | 0.2438 ms | 0.2341 ms |
| 2 | 0.2196 (−9.9 %) | 0.1882 (**−19.6 %**) |
| 4 | 0.1693 (−30.6 %) | 0.1547 (**−33.9 %**) |
| 8 | 0.4272 (+75 %) | 0.2040 (**−12.9 %**) |
| 10 | 1.7508 (+618 %) | 1.6164 (+591 %) |

The fresh-process column is closer to normal use because thread variables are
usually set before Python starts. In that test, eight threads were 13% faster
than one. Resizing a live pool made the same setting look 75% slower, so keep the
two measurements separate.

Two and four threads improved the measured runtime. Performance collapsed when
the pool covered the **whole machine** (10 threads on 10 cores).

## Which advice applies depends on how you parallelise

**One spectrum at a time in one process.** A small BLAS pool may help. On the
machine we tested, four threads were fastest, and one thread was 44% slower.

**Many spectra in parallel.** Start with one BLAS thread per worker. Giving every
worker a larger pool can oversubscribe the machine.

We have **not** measured the multi-worker case in this project. The recommendation
is a practical starting point based on how processes compete for cores. The
sixfold slowdown above came from a single process whose BLAS pool covered the
host. It was not a measurement of multi-worker oversubscription.

## Setting it

Set thread counts in the environment before NumPy and SciPy are imported. For a
many-worker catalog run, begin with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python your_script.py
```

## The advisory

With the `performance` extra installed, the package warns once when a BLAS pool
covers all CPUs available to the process. It will not change the setting for
you.

```python
from gp_dla_finder.performance import blas_thread_report

dict(blas_thread_report())
```

To silence the advisory once you have made an informed choice:

```python
import warnings
from gp_dla_finder.performance import BLASPerformanceWarning

warnings.filterwarnings("ignore", category=BLASPerformanceWarning)
```

:::{note}
**Scope.** These measurements describe one process as a function of BLAS pool
size. They do **not** measure `workers × threads > cores`.

**The advisory is relative.** It fires when the BLAS pool covers every CPU
available to the process. The audited host did not support a stable absolute
thread threshold: the eight-thread result changed sign between measurements,
while the full-pool slowdown remained.

**Provisional.** The result comes from one machine and one BLAS implementation. A
realistic many-core Linux baseline has not been measured.
:::

## Reproducing these numbers

```bash
python tools/blas_audit.py --thread-sweep 1 2 4 8 10 --json audit.json
python tools/benchmark.py --samples 10000 --repeats 5 --json bench.json
```

Both commands record CPU and platform information, BLAS identity, detected
thread pools, the thread environment, warm-up and repetition policies, and the
timing distribution. The retained JSON is the evidence for the table above.
