# Rayon + simdeez Benchmark Demo

This example project demonstrates that multithreading and SIMD can be used in conjuction for a ~70x improvement in performance.

```
scalar                  time:   [3.0758 ms 3.0914 ms 3.1091 ms]
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) high mild
  3 (3.00%) high severe

par_scalar              time:   [558.61 us 562.73 us 567.50 us]
Found 8 outliers among 100 measurements (8.00%)
  3 (3.00%) high mild
  5 (5.00%) high severe

runtime_select          time:   [97.475 us 97.948 us 98.448 us]
Found 9 outliers among 100 measurements (9.00%)
  2 (2.00%) high mild
  7 (7.00%) high severe

par_runtime_select      time:   [41.774 us 42.186 us 42.643 us]
Found 13 outliers among 100 measurements (13.00%)
  6 (6.00%) high mild
  7 (7.00%) high severe
```
