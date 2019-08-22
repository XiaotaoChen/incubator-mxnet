/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file Quantization_int8.cu
 * \brief
 * \author Jingqiu Zhou & Ruize Hou
*/

#include "./quantization_int8-inl.h"
#include <cuda.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "../common/cuda_utils.h"

#include<stdio.h>

#define QUANT_LEVEL 255
#define SYMETIC_QUANT_LEVLE 127
#define THREAD_PER_BLOCK 256
namespace mxnet {
namespace op {
    // template <typename DType>
    // struct QUANT_WEIGHT_GPU_MINMAX {
    //     __device__ static void Map(int i, DType* data, DType* out, DType* src_max, DType* src_min)
    //     {
    //         __shared__ DType quant_unit;
    //         __shared__ DType S_min_f;
    //         __shared__ DType S_max_f;
    //         int tidx = threadIdx.x;
    //         //compute quantization inside each block
    //         if (tidx == 0) {

                // S_min_f = *src_min;
                // S_max_f = *src_max;

                // //insure 0 in the range
                // //calculate a possible quant_unit
                // if (S_min_f > DType(-1e-6)) {
                //     S_min_f = DType(-1e-6);
                // }
                // if (S_max_f < DType(1e-6)) {
                //     S_max_f = DType(1e-6);
                // }
                // quant_unit = (S_max_f - S_min_f) / DType(QUANT_LEVEL);
                // DType delta = quant_unit + S_min_f / ceil(-S_min_f / quant_unit);
                // //adjust range
                // quant_unit = quant_unit - delta;
                // S_max_f = S_max_f - delta * DType(QUANT_LEVEL) / DType(2.);
                // S_min_f = S_min_f + delta * DType(QUANT_LEVEL) / DType(2.);
    //         }

    //         __syncthreads();
            // DType temp = *(data + i) > S_max_f ? S_max_f : *(data + i);
            // temp = temp < S_min_f ? S_min_f : temp;
            // *(out + i) = floor((temp - S_min_f) / quant_unit + 0.5) * quant_unit + S_min_f;
    //     }
    // };

    template <typename DType>
    struct QUANT_WEIGHT_GPU_POWER2 {
        __device__ static void Map(int i, DType* data, DType* out, DType* log2t)
        {

            __shared__ DType quant_unit;

            int tidx = threadIdx.x;
            //compute quantization inside each block
            if (tidx < 1) {
                quant_unit = (::pow(2.0, ::ceil(*log2t)) * DType(2.0)) / DType(QUANT_LEVEL);
            }

            __syncthreads();
            DType int8_val = DType(floor(*(data + i) / quant_unit + 0.5));
            int8_val = int8_val > DType(QUANT_LEVEL / 2 - 1) ? DType(QUANT_LEVEL / 2 - 1) : int8_val;
            int8_val = int8_val < -DType(QUANT_LEVEL / 2) ? -DType(QUANT_LEVEL / 2) : int8_val;
            *(out + i) = int8_val * quant_unit;
        }
    };

    template <typename DType>
    struct QUANT_WEIGHT_GPU_MINMAX {
        __device__ static void Map(int i, DType* data, DType* out, DType* src_max)
        {
            DType S_max_f, S_min_f, quant_unit;
            S_max_f = *src_max;
            S_min_f = - S_max_f;

            //calculate a possible quant_unit
            if (S_min_f > DType(-1e-6)) {
                S_min_f = DType(-1e-6);
            }
            if (S_max_f < DType(1e-6)) {
                S_max_f = DType(1e-6);
            }
            quant_unit = S_max_f / DType(SYMETIC_QUANT_LEVLE);
            DType temp = *(data + i) > S_max_f ? S_max_f : *(data + i);     // min(data[i], S_max_f)
            temp = temp < S_min_f ? S_min_f : temp;                           // max(temp, S_min_f)
            // DType floor_data = floor(temp/ quant_unit);
            DType floor_data = floor(temp/ quant_unit + 0.5);
            *(out + i) = floor_data * quant_unit;

        }
    };

    template <typename DType>
    struct QUANT_ACT_GPU_MINMAX {
        __device__ static void Map(int i, DType* data, DType* out, DType* S_act, int quant_countdown, int is_train)
        {
            if (quant_countdown > 0 && is_train > 0) {
                *(out + i) = *(data + i);
            }
            else {
                DType S_max_f = *S_act;
                DType S_min_f = - S_max_f;
                DType quant_unit;

                quant_unit = S_max_f / DType(SYMETIC_QUANT_LEVLE);
                //use i= 0 to update the recorded max/min
                DType temp = *(data + i) > S_max_f ? S_max_f : *(data + i);     // min(data[i], S_max_f)
                temp = temp < S_min_f ? S_min_f : temp;                           // max(temp, S_min_f)

                //Make data in [S_min_f, S_max_f]
                DType floor_data = floor(temp/ quant_unit);
                *(out + i) = floor_data * quant_unit;
            }
        }
    };

    template <typename DType>
    struct UPDATE_MINMAX {
        __device__ static void Map(int i, DType* S_act, DType* max_S, DType* min_S, DType decay, bool init)
        {
            DType S_max_f = *S_act;
            DType S_min_f = *(S_act + 1);
            if (init) {
                S_max_f = *max_S;
                S_min_f = *min_S;
            } else {
                S_max_f = *S_act * decay + (1 - decay) * (*max_S);
                S_min_f = *(S_act + 1) * decay + (1 - decay) * (*min_S);
            }
            if (S_max_f < 1e-6) {
                S_max_f = 1e-6;
            }
            if (S_min_f > -1e-6) {
                S_min_f = -1e-6;
            }
            *S_act = S_max_f;
            *(S_act + 1) = S_min_f;
        }
    };

    template <typename DType>
    struct UPDATE_MINMAX_WITHOUT_DECAY {
        __device__ static void Map(int i, DType* s_act, DType* max_S, DType* min_S) {
            *s_act = *max_S;
            *(s_act + 1) = *min_S;
        }
    };

    /*

    // This code maybe faster. But we didn't confirm whether it is correct or wrong.
    // Here we use thrust::min_element && thrust::max_element instead.

    template <typename DType>
    struct REDUCE_MINMAX {
        __device__ static void Map(int i, DType* src_max, DType* dst_max, DType* src_min, DType* dst_min, int pre_num)
        {
            //moving pinters
            int tid = threadIdx.x;

            __shared__ DType max_arr[THREAD_PER_BLOCK];
            __shared__ DType min_arr[THREAD_PER_BLOCK];

            //load data into shared memory
            if (2 * i + 1 < pre_num) {
                max_arr[tid] = *(src_max + 2 * i) > *(src_max + 2 * i + 1) ? *(src_max + 2 * i) : *(src_max + 2 * i + 1);
                min_arr[tid] = *(src_min + 2 * i) < *(src_min + 2 * i + 1) ? *(src_min + 2 * i) : *(src_min + 2 * i + 1);
            } else {
                max_arr[tid] = *(src_max + 2 * i);
                min_arr[tid] = *(src_min + 2 * i);
            }

            //dst_max[blockIdx.x*THREAD_PER_BLOCK+tid]=max_arr[tid];
            //dst_min[blockIdx.x*THREAD_PER_BLOCK+tid]=min_arr[tid];

            __syncthreads();
            //call the function
            //compute max/min
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && 2 * (i + s) < pre_num) {
                    max_arr[tid] = max_arr[tid] > max_arr[tid + s] ? max_arr[tid] : max_arr[tid + s];
                    min_arr[tid] = min_arr[tid] < min_arr[tid + s] ? min_arr[tid] : min_arr[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                dst_max[blockIdx.x] = max_arr[0];
                dst_min[blockIdx.x] = min_arr[0];
            }
        }
    };
    */

    template <typename DType>
    struct GRAD_POWER2 {
        __device__ static void Map(int i, DType* data, DType* gdata, DType* out, DType* log2t)
        {
            __shared__ DType quant_unit;

            int tidx = threadIdx.x;
            //compute quantization inside each block
            if (tidx < 1) {
                quant_unit = (::pow(2.0, ::ceil(*log2t)) * DType(2.0)) / DType(QUANT_LEVEL);
            }
            __syncthreads();

            DType int8_val = DType(floor(*(data + i) / quant_unit + 0.5));
            //int8_val=int8_val>DType(QUANT_LEVEL/2-1)?DType(QUANT_LEVEL/2-1):int8_val;
            //int8_val=int8_val<-DType(QUANT_LEVEL/2)?-DType(QUANT_LEVEL/2):int8_val;
            DType dv_ds = int8_val - (*(data + i) / quant_unit);
            if (int8_val > DType(QUANT_LEVEL / 2 - 1)) {
                dv_ds = DType(QUANT_LEVEL / 2 - 1);
            } else if (int8_val < -DType(QUANT_LEVEL / 2)) {
                dv_ds = -DType(QUANT_LEVEL / 2);
            }
            DType local_grad = logf(2.0) * quant_unit * dv_ds;

            *(out + i) = *(gdata + i) * local_grad;
        }
    };

    template <typename DType>
    struct GRAD_WEIGHT_POWER2 {
        __device__ static void Map(int i, DType* data, DType* gdata, DType* out, DType* log2t)
        {
            __shared__ DType quant_unit;

            int tidx = threadIdx.x;
            //compute quantization inside each block
            if (tidx < 1) {
                quant_unit = (::pow(2.0, ::ceil(*log2t)) * DType(2.0)) / DType(QUANT_LEVEL);
            }
            __syncthreads();

            DType int8_val = DType(floor(*(data + i) / quant_unit + 0.5));
            DType factor = int8_val > DType(QUANT_LEVEL / 2 - 1) ? DType(0.) : DType(1.);
            factor = int8_val < -DType(QUANT_LEVEL / 2) ? DType(0.) : factor;

            *(out + i) = *(gdata + i) * factor;
        }
    };

    template <typename DType>
    struct INIT_LOG2T {
        __device__ static void Map(int i, DType* log2t, DType* max_val, DType* min_val)
        {
            DType t = DType(::abs(*max_val) > ::abs(*min_val) ? std::abs(*max_val) : std::abs(*min_val));
            t = t > DType(1.) ? t : DType(1.);
            *(log2t) = log2f(t);
            *(log2t + 1) = DType(0.);
            *(log2t + 2) = DType(0.);
        }
    };

    template <typename DType>
    struct UPDATE_LOG2T {
        __device__ static void Map(int i, DType* log2t, DType* grad)
        {
            DType alpha = 1e-3;
            DType beta1 = 0.9;
            DType beta2 = 0.999;
            DType epsilon = 1e-8;

            *(log2t + 1) = beta1 * (*(log2t + 1)) + (1. - beta1) * (*grad);
            *(log2t + 2) = beta2 * (*(log2t + 2)) + (1. - beta2) * (*grad) * (*grad);
            DType mt = *(log2t + 1) / (1 - beta1 * beta1);
            DType vt = *(log2t + 2) / (1 - beta2 * beta2);
            *(log2t) -= alpha * tanhf(mt / (sqrtf(vt) + epsilon));
        }
    };
    template <typename DType>
    struct REDUCE_POWER2 {
        __device__ static void Map(int i, DType* grad_src, DType* grad_dst, int pre_num)
        {
            //moving pinters
            int tid = threadIdx.x;

            __shared__ DType sum_grad[THREAD_PER_BLOCK];

            //load data into shared memory
            if (2 * i + 1 < pre_num) {
                sum_grad[tid] = *(grad_src + 2 * i) + *(grad_src + 2 * i + 1);
            } else if (2 * i + 1 == pre_num) {
                sum_grad[tid] = *(grad_src + 2 * i);
            } else {
                sum_grad[tid] = DType(0.);
            }

            __syncthreads();
            //call the function
            //compute max/min
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && 2 * (i + s) < pre_num) {
                    sum_grad[tid] = sum_grad[tid] + sum_grad[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                grad_dst[blockIdx.x] = sum_grad[0];
            }
        }
    };

    template <typename DType>
    struct ABS
    {
        __device__ static void Map(int i, DType* src, DType* dst) {
            DType tmp = *(src + i);
            *(dst + i) = tmp > 0 ? tmp : -tmp;
        }
    };
}
}
namespace mshadow {
template <typename DType>
void Find_minmax(int num, DType * src, DType * max_target, DType * min_target)
{
    DType* temp;
    temp = thrust::max_element(thrust::device, src, src + num);
    cudaMemcpy(max_target, temp, sizeof(DType), cudaMemcpyDeviceToDevice);
    temp = thrust::min_element(thrust::device, src, src + num);
    cudaMemcpy(min_target, temp, sizeof(DType), cudaMemcpyDeviceToDevice);
}

template <typename DType>
void Find_abs_max(int num, DType * src, DType * max_target, DType* min_target)
{
    DType* temp;
    temp = thrust::max_element(thrust::device, src, src + num);
    cudaMemcpy(max_target, temp, sizeof(DType), cudaMemcpyDeviceToDevice);
    DType* max_val; // on cpu
    max_val = (DType*)malloc(sizeof(DType));
    cudaMemcpy(max_val, temp, sizeof(DType), cudaMemcpyDeviceToHost);
    *max_val = - (*max_val);
    cudaMemcpy(min_target, max_val, sizeof(DType), cudaMemcpyHostToDevice);
    free(max_val);
}

template <typename DType>
void print_device(DType* data, int num, std::string flag) {
    DType* temp;
    temp = (DType*) malloc(sizeof(DType) * num);
    cudaMemcpy(temp, data, sizeof(DType) * num, cudaMemcpyDeviceToHost);
    printf("--------------------------- %s ---------------------------\n", flag.c_str());
    for (int i=0; i< num; i++) {
        printf("%f ",temp[i]);
    }
    printf("\n");
    free(temp);
}

template <typename DType>
void quantization_int8_weight(std::string qmod, Tensor<gpu, 3, DType> data, Tensor<gpu, 3, DType>& out, 
                              Tensor<gpu, 1, DType> aux, Stream<gpu>* s, bool init, int is_train)
{
    //find min and max
    int num = out.size(0) * out.size(1) * out.size(2);
    //int offset = (num + 2 * THREAD_PER_BLOCK) / (2 * THREAD_PER_BLOCK);
    //choose quantization path
    if (qmod == std::string("minmax") || init) {
        if (is_train > 0) {
            //declare space for reduction
            DType* target_max;
            DType* target_min;
            cudaMalloc((void**)&target_max, sizeof(DType));
            cudaMalloc((void**)&target_min, sizeof(DType));
            DType* data_abs;
            cudaMalloc((void**)&data_abs, sizeof(DType) * num);
            mxnet::op::mxnet_op::Kernel<mxnet::op::ABS<DType>, gpu>::Launch(s, num, data.dptr_, data_abs);
            Find_abs_max(num, data_abs, target_max, target_min);
            // print_device(data_abs, num, std::string("data abs"));
            print_device(target_max, 1, std::string("target max"));
            // print_device(data.dptr_, num, std::string("source data"));
            // print_device(out.dptr_, num, std::string("quantized data"));
            mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_MINMAX_WITHOUT_DECAY<DType>, gpu>::Launch(s, 1, aux.dptr_, 
            target_max, target_min);
            cudaFree(target_max);
            cudaFree(target_min);
            cudaFree(data_abs);
        }
        //perform quantization
        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU_MINMAX<DType>, gpu>::Launch(s, num, data.dptr_, out.dptr_, aux.dptr_);
        
    } else if (qmod == std::string("power2")) {
        if (init) {
            DType* target_max;
            DType* target_min;
            cudaMalloc((void**)&target_max, sizeof(DType));
            cudaMalloc((void**)&target_min, sizeof(DType));
            Find_minmax(num, data.dptr_, target_max, target_min);
            mxnet::op::mxnet_op::Kernel<mxnet::op::INIT_LOG2T<DType>, gpu>::Launch(s, 1, aux.dptr_, target_max, target_min);
            cudaFree(target_max);
            cudaFree(target_min);
        }
        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU_POWER2<DType>, gpu>::Launch(s, num,
            data.dptr_, out.dptr_,
            aux.dptr_);
    }
}
template <typename DType>
void quantization_int8_act(std::string qmod, Tensor<gpu, 3, DType> data, Tensor<gpu, 3, DType>& out, 
                           Tensor<gpu, 1, DType> &aux, DType decay, Stream<gpu>* s, 
                           int quant_countdown, bool init, int is_train)
{

    int num = out.size(0) * out.size(1) * out.size(2);
    //int offset = (num + 2 * THREAD_PER_BLOCK) / (2 * THREAD_PER_BLOCK);
    if (qmod == std::string("minmax")) {
        if (is_train > 0) 
        {
            DType* target_max;
            DType* target_min;
            cudaMalloc((void**)&target_max, sizeof(DType));
            cudaMalloc((void**)&target_min, sizeof(DType));
            // //find the max and min first
            // Find_minmax(num, data.dptr_, target_max, target_min);
            DType* data_abs;
            cudaMalloc((void**)&data_abs, sizeof(DType) * num);
            mxnet::op::mxnet_op::Kernel<mxnet::op::ABS<DType>, gpu>::Launch(s, num, data.dptr_, data_abs);
            Find_abs_max(num, data_abs, target_max, target_min);

            //Then, update the min and max
            mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_MINMAX<DType>, gpu>::Launch(s, 1, aux.dptr_, target_max, target_min, decay, init);
            cudaFree(target_max);
            cudaFree(target_min);
            cudaFree(data_abs);
        }

        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_ACT_GPU_MINMAX<DType>, gpu>::Launch(s, num, data.dptr_, out.dptr_, 
                                                                                         aux.dptr_, quant_countdown, is_train);
    } else if (qmod == std::string("power2")) {
        if (init) {
            DType* target_max;
            DType* target_min;

            cudaMalloc((void**)&target_max, sizeof(DType));
            cudaMalloc((void**)&target_min, sizeof(DType));
            Find_minmax(num, data.dptr_, target_max, target_min);
            mxnet::op::mxnet_op::Kernel<mxnet::op::INIT_LOG2T<DType>, gpu>::Launch(s, 1, aux.dptr_, target_max, target_min);
            cudaFree(target_max);
            cudaFree(target_min);
        }
        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU_POWER2<DType>, gpu>::Launch(s, num,
            data.dptr_, out.dptr_,
            aux.dptr_);
    }
}

template <typename DType>
void quantization_grad(std::string qmod, Tensor<gpu, 3, DType>& gdata, Tensor<gpu, 3, DType> grad,
                       Tensor<gpu, 3, DType> data, Tensor<gpu, 1, DType>& aux, Stream<gpu>* s)
{

    int num = grad.size(0) * grad.size(1) * grad.size(2);
    int offset = (num + 2 * THREAD_PER_BLOCK) / (2 * THREAD_PER_BLOCK);
    DType* Temp;
    cudaMalloc((void**)&Temp, sizeof(DType) * offset * 2);
    /*
  DType ori_grad_cpu[3];
  cudaMemcpy(ori_grad_cpu,grad.dptr_,sizeof(DType),cudaMemcpyDeviceToHost);
  cudaMemcpy(ori_grad_cpu+1,data.dptr_,sizeof(DType),cudaMemcpyDeviceToHost);
  cudaMemcpy(ori_grad_cpu+2,gdata.dptr_,sizeof(DType),cudaMemcpyDeviceToHost);
  std::cout<<ori_grad_cpu[0]<<" "<<ori_grad_cpu[1]<<" "<<ori_grad_cpu[2]<<std::endl;
  */
    //compute gradient for threash hold
    mxnet::op::mxnet_op::Kernel<mxnet::op::GRAD_POWER2<DType>, gpu>::Launch(s, num, data.dptr_, grad.dptr_, gdata.dptr_, aux.dptr_);
    /*
  DType ori_grad_cpu[3];
  cudaMemcpy(ori_grad_cpu,grad.dptr_,sizeof(DType),cudaMemcpyDeviceToHost);
  cudaMemcpy(ori_grad_cpu+1,data.dptr_,sizeof(DType),cudaMemcpyDeviceToHost);
  cudaMemcpy(ori_grad_cpu+2,gdata.dptr_,sizeof(DType),cudaMemcpyDeviceToHost);
  std::cout<<ori_grad_cpu[0]<<" "<<ori_grad_cpu[1]<<" "<<ori_grad_cpu[2]<<std::endl;
  */
    //reduce gradient for threash hold
    int current_num = num;
    int pre_num;
    int current_i;
    DType* src_grad = gdata.dptr_;
    DType* dst_grad = Temp;
    DType* inter_media;
    bool first_iter = true;

    while (current_num > 1) {
        //after this iteration num of ele
        pre_num = current_num;
        current_i = (current_num + 1) / 2;

        mxnet::op::mxnet_op::Kernel<mxnet::op::REDUCE_POWER2<DType>, gpu>::Launch(s, current_i, src_grad, dst_grad, pre_num);

        current_num = (current_num + 2 * THREAD_PER_BLOCK - 1) / (THREAD_PER_BLOCK * 2);
        //current_num=current_i;
        if (first_iter) {
            src_grad = dst_grad;
            dst_grad = Temp + offset;
            first_iter = false;
        } else {
            inter_media = src_grad;
            src_grad = dst_grad;
            dst_grad = inter_media;
        }
    }
    //compute grad
    mxnet::op::mxnet_op::Kernel<mxnet::op::GRAD_WEIGHT_POWER2<DType>, gpu>::Launch(s, num, data.dptr_, grad.dptr_, gdata.dptr_, aux.dptr_);
    //update aux
    mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_LOG2T<DType>, gpu>::Launch(s, 1, aux.dptr_, src_grad);
    /*
  DType grad_cpu[2];
  cudaMemcpy(grad_cpu,src_grad,sizeof(DType),cudaMemcpyDeviceToHost);
  std::cout<<grad_cpu[0]<<std::endl;
  */
    cudaFree(Temp);
}
}

namespace mxnet {
namespace op {
    template <>
    Operator* CreateOp<gpu>(Quantization_int8Para param, int dtype)
    {
        Operator* op = nullptr;
        MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new Quantization_int8Op<gpu, DType>(param);
        });
        return op;
    }

} // namespace op
} // namespace mxnet
