/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_bwd_kernel_zoh_s.cuh"
#include "selective_scan_bwd_kernel_zoh_f.cuh"
#include "selective_scan_bwd_kernel_foh_s.cuh"
#include "selective_scan_bwd_kernel_foh_f.cuh"

template void selective_scan_bwd_zohs_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_zohf_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_fohs_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_fohf_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);