# Weight residency measurements

* `weights.bfp16` 6,913,965,970 B, 506 records
* packed bfp16ebs8 6,909,788,160 B (270 tensors)
* raw fp32 1-D 4,089,856 B (236 tensors)
* total params 6,143,056,384
* resident 6,913,878,016 B vs fp32 24,572,225,536 B = **3.554x** compression

## Steady state (N=3 iterations, 8 sample tensors)

* reads: 24
* unpack calls: **0**

## If decoded per call

* unpack calls: 24
* unpack bytes: 610,467,840
