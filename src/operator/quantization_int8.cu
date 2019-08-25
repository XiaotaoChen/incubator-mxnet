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

    template <typename DType>
    struct QUANT_DATA_POWER2 {
        __device__ static void Map(int i, DType* data, DType* out, DType* log2t)
        {

            __shared__ DType quant_unit;

            int tidx = threadIdx.x;
            //compute quantization inside each block
            if (tidx < 1) {
                quant_unit = ::pow(2.0, ::ceil(*log2t)) / DType(SYMETIC_QUANT_LEVLE);
            }

            __syncthreads();
            // DType int8_val = DType(floor(*(data + i) / quant_unit + 0.5));
            DType int8_val = DType(round(*(data + i) / quant_unit));
            int8_val = int8_val > DType(SYMETIC_QUANT_LEVLE) ? DType(SYMETIC_QUANT_LEVLE) : int8_val;
            int8_val = int8_val < -DType(SYMETIC_QUANT_LEVLE) ? -DType(SYMETIC_QUANT_LEVLE) : int8_val;
            *(out + i) = int8_val * quant_unit;
        }
    };

    template <typename DType>
    struct QUANT_DATA_MINMAX {
        __device__ static void Map(int i, DType* data, DType* out, DType* src_max)
        {
            DType S_max_f, S_min_f, quant_unit;
            S_max_f = *src_max;
            S_min_f = - S_max_f;
            quant_unit = S_max_f / DType(SYMETIC_QUANT_LEVLE);
            DType temp = *(data + i) > S_max_f ? S_max_f : *(data + i);     // min(data[i], S_max_f)
            temp = temp < S_min_f ? S_min_f : temp;                           // max(temp, S_min_f)
            DType round_data = round(temp/ quant_unit);
            *(out + i) = round_data * quant_unit;

        }
    };

    template <typename DType>
    struct UPDATE_MINMAX_WITH_DECAY {
        __device__ static void Map(int i, DType* S_act, DType* max_S, DType decay)
        {
            DType S_max_f = *S_act;
            S_max_f = (*S_act) * decay + (1 - decay) * (*max_S);
            if (S_max_f < 1e-6) {
                S_max_f = 1e-6;
            }
            *S_act = S_max_f;
            *(S_act + 1) = - S_max_f;
        }
        //Update with EMA.
    };

    template <typename DType>
    struct UPDATE_MINMAX_WITHOUT_DECAY {
        __device__ static void Map(int i, DType* s_act, DType* max_S) {
            *s_act = *max_S;
            *(s_act + 1) = - (*max_S);
        }
    };
    // the gradients of q(x,s) of log2t
    template <typename DType>
    struct DATA_GRAD_POWER2 {
        __device__ static void Map(int i, DType* data, DType* gdata, DType* out, DType* log2t)
        {
            __shared__ DType quant_unit;

            int tidx = threadIdx.x;
            //compute quantization inside each block
            if (tidx < 1) {
                quant_unit = ::pow(2.0, ::ceil(*log2t)) / DType(SYMETIC_QUANT_LEVLE);
            }
            __syncthreads();

            DType int8_val = DType(round(*(data + i) / quant_unit));
            //int8_val=int8_val>DType(QUANT_LEVEL/2-1)?DType(QUANT_LEVEL/2-1):int8_val;
            //int8_val=int8_val<-DType(QUANT_LEVEL/2)?-DType(QUANT_LEVEL/2):int8_val;
            DType dv_ds = int8_val - (*(data + i) / quant_unit);
            if (int8_val > DType(SYMETIC_QUANT_LEVLE)) {
                dv_ds = DType(SYMETIC_QUANT_LEVLE);
            } else if (int8_val < -DType(SYMETIC_QUANT_LEVLE)) {
                dv_ds = -DType(SYMETIC_QUANT_LEVLE);
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
                quant_unit = ::pow(2.0, ::ceil(*log2t)) / DType(SYMETIC_QUANT_LEVLE);
            }
            __syncthreads();

            DType int8_val = DType(round(*(data + i) / quant_unit));
            DType factor = int8_val > DType(SYMETIC_QUANT_LEVLE) ? DType(0.) : DType(1.);
            factor = int8_val < -DType(SYMETIC_QUANT_LEVLE) ? DType(0.) : factor;

            *(out + i) = *(gdata + i) * factor;
        }
    };

    template <typename DType>
    struct INIT_LOG2T {
        __device__ static void Map(int i, DType* log2t, DType* max_val)
        {
            DType t = (*max_val);
            t = t > DType(1.) ? t : DType(1.);
            *(log2t) = log2f(t);
            *(log2t + 1) = DType(0.);
            *(log2t + 2) = DType(0.);
            //*(log2t) means the f (s = 2^f)
            //*(log2t + 1) && *(log2t + 2) is used to update f.
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
}
}
namespace mshadow {

template <typename DType>
void Find_max(int num, DType * src, DType * max_target)
{
    DType* temp;
    temp = thrust::max_element(thrust::device, src, src + num);
    DType max_val, min_val;
    cudaMemcpy(&max_val, temp, sizeof(DType), cudaMemcpyDeviceToHost);
    temp = thrust::min_element(thrust::device, src, src + num);
    cudaMemcpy(&min_val, temp, sizeof(DType), cudaMemcpyDeviceToHost);
    //And we need max(max_val, -min_val) to get the max of abs.
    min_val = -min_val;
    if (max_val > min_val)
    {
        cudaMemcpy(max_target, &max_val, sizeof(DType), cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMemcpy(max_target, &min_val, sizeof(DType), cudaMemcpyHostToDevice);
    }
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
    if (qmod == std::string("minmax")) {
        DType* target_max;
        cudaMalloc((void**)&target_max, sizeof(DType));
        //perfrom reduction , fing min max
        Find_max(num, data.dptr_, target_max);
        mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_MINMAX_WITHOUT_DECAY<DType>, gpu>::Launch(s, 1, aux.dptr_, 
        target_max);
        // print_device(aux.dptr_, 2, std::string("weight aux"));
        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_DATA_MINMAX<DType>, gpu>::Launch(s, num, data.dptr_, out.dptr_, aux.dptr_);
        
    } else if (qmod == std::string("power2")) {
        // print_device<DType>(aux.dptr_, 3, std::string("power2 w aux"));
        if (is_train > 0 && init) {
            DType* target_max;
            cudaMalloc((void**)&target_max, sizeof(DType));
            Find_max(num, data.dptr_, target_max);
            mxnet::op::mxnet_op::Kernel<mxnet::op::INIT_LOG2T<DType>, gpu>::Launch(s, 1, aux.dptr_, target_max);
            // print_device<DType>(target_max, 1, std::string("power2 w target max"));
            // print_device<DType>(aux.dptr_, 3, std::string("power2 w aux"));
            cudaFree(target_max);
        }
        // print_device<DType>(data.dptr_, num, std::string("power2 w data"));
        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_DATA_POWER2<DType>, gpu>::Launch(s, num,
            data.dptr_, out.dptr_,
            aux.dptr_);
        // print_device<DType>(out.dptr_, num, std::string("power2 w out"));
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
            // print_device<DType>(data.dptr_, num, std::string("act data"));
            DType* target_max;
            cudaMalloc((void**)&target_max, sizeof(DType));
            //find the max and min first
            Find_max(num, data.dptr_, target_max);
            //Then, update the min and max
            if (init) {
              mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_MINMAX_WITHOUT_DECAY<DType>, gpu>::Launch(s, 1, aux.dptr_, 
                target_max);
            }
            else {
              mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_MINMAX_WITH_DECAY<DType>, gpu>::Launch(s, 1, aux.dptr_, target_max, decay);
            }
            
            cudaFree(target_max);
            // print_device<DType>(aux.dptr_, 3, std::string("act aux"));
        }
        mxnet::op::mxnet_op::Kernel<
          mxnet::op::QUANT_DATA_MINMAX<DType>, gpu>::Launch(
              s, num, data.dptr_, out.dptr_, aux.dptr_);
        // print_device<DType>(out.dptr_, num, std::string("act out"));
    } else if (qmod == std::string("power2")) {
        // print_device<DType>(aux.dptr_, 3, std::string("power2 act aux"));
        if (is_train > 0 && init) {
            DType* target_max;
            cudaMalloc((void**)&target_max, sizeof(DType));
            Find_max(num, data.dptr_, target_max);
            mxnet::op::mxnet_op::Kernel<mxnet::op::INIT_LOG2T<DType>, gpu>::Launch(s, 1, aux.dptr_, target_max);
            // print_device<DType>(target_max, 1, std::string("power2 act target max"));
            // print_device<DType>(aux.dptr_, 3, std::string("power2 act aux"));
            cudaFree(target_max);
        }
        // print_device<DType>(data.dptr_, num, std::string("power2 act data"));
        mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_DATA_POWER2<DType>, gpu>::Launch(s, num,
            data.dptr_, out.dptr_,
            aux.dptr_);
        // print_device<DType>(out.dptr_, num, std::string("power2 act out"));
    }
}

template <typename DType>
void quantization_grad(std::string qmod, Tensor<gpu, 3, DType>& gdata, Tensor<gpu, 3, DType> grad,
                       Tensor<gpu, 3, DType> data, Tensor<gpu, 1, DType>& aux, Stream<gpu>* s)
{
    int num = grad.size(0) * grad.size(1) * grad.size(2);
    //The gdata is only a temporary variable here. The GRAD_WEIGHT_POWER2 is where it get the true value.
    mxnet::op::mxnet_op::Kernel<mxnet::op::DATA_GRAD_POWER2<DType>, gpu>::Launch(s, num, data.dptr_, grad.dptr_, gdata.dptr_, aux.dptr_);
    //compute grad
    DType * res;
    cudaMalloc((void**)&res, sizeof(DType));
    DType temp = thrust::reduce(thrust::device, gdata.dptr_, gdata.dptr_ + num);
    mxnet::op::mxnet_op::Kernel<mxnet::op::GRAD_WEIGHT_POWER2<DType>, gpu>::Launch(s, num, data.dptr_, grad.dptr_, gdata.dptr_, aux.dptr_);
    //update aux
    cudaMemcpy(res, &temp, sizeof(DType), cudaMemcpyHostToDevice);
    // print_device<DType>(res, 1, std::string("power2 res"));
    // print_device<DType>(aux.dptr_, 1, std::string("power2 grad aux"));
    //Move it to a CUDA memory. Then it can launch the UPDATE_LOG2T correctly.(It is execute in cuda.)
    mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_LOG2T<DType>, gpu>::Launch(s, 1, aux.dptr_, res);
    // print_device<DType>(aux.dptr_, 1, std::string("power2 grad aux"));
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
