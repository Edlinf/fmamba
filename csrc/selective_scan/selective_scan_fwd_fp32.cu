/******************************************************************************
 * Copyright (c) 2024, Yujie Zhu.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_fwd_kernel_zoh_s.cuh"
#include "selective_scan_fwd_kernel_zoh_f.cuh"
#include "selective_scan_fwd_kernel_foh_s.cuh"
#include "selective_scan_fwd_kernel_foh_f.cuh"

template void selective_scan_fwd_zohs_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_zohs_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_zohf_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_zohf_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_fohs_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_fohs_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_fohf_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_fohf_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);