# Parallel implementation of genetic algorithm in CUDA

Created as an experiment with CUDA to test parallel computation on graphic card with asynchronus data exchange with the device.

## Done
- basic functionality and control flow
- async data transfer
- infinite population - data that do not fit in the memory is exchanged with storage drive

## TODO
- **migration**
- better selection function (for now implemented the simplest one - Trunc)
- consider different mutation function
- test convergence
