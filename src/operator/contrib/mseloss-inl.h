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
 * \file mseloss-inl.h
* \author Xiaotao Chen
*/

#ifndef MXNET_OPERATOR_MSELOSS_INL_H_
#define MXNET_OPERATOR_MSELOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/control_flow_op.h"
#include "../tensor/indexing_op.h"
#include "../quantization/quantization_utils.h"
// #include "./operator_common.h"
// #include "./mshadow_op.h"
// #include "./mxnet_op.h"
// #include "./tensor/control_flow_op.h"
// #include "./tensor/indexing_op.h"
// #include "./quantization/quantization_utils.h"


namespace mxnet {
namespace op {

namespace Mseloss_enum {
enum MselossOpInputs {kData};
enum MselossOpOutputs {kOut};
enum MselossOpResource {kTempSpace};
}  // namespace mseloss_enum


template <typename xpu, typename DType>
void print_data_1D(mshadow::Tensor<xpu, 1, DType> data, mshadow::Stream<xpu> *s, const OpContext &ctx, std::string flag) {
    mshadow::Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    DType* temp;
    temp = (DType*) malloc(data.shape_.Size() * sizeof(DType));
    mshadow::Tensor<cpu, 1, DType> temp_tensor(temp, data.shape_, s_cpu);
    mshadow::Copy(temp_tensor, data, s);
    printf("--------------------------- %s ---------------------------\n", flag.c_str());
    for (int i=0; i< temp_tensor.size(0); i++) {
      printf("%f ", temp_tensor[i]);
    }
    printf("\n");
    free(temp);
}

template <typename xpu, typename DType>
void print_data_4D(mshadow::Tensor<xpu, 4, DType> data, mshadow::Stream<xpu> *s, const OpContext &ctx, std::string flag) {
    mshadow::Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    DType* temp;
    temp = (DType*) malloc(data.shape_.Size() * sizeof(DType));
    mshadow::Tensor<cpu, 4, DType> temp_tensor(temp, data.shape_, s_cpu);
    mshadow::Copy(temp_tensor, data, s);
    printf("--------------------------- %s ---------------------------\n", flag.c_str());
    for (int i=0; i< temp_tensor.size(0); i++) {
     for (int j=0; j< temp_tensor.size(1); j++) {
       for (int k=0; k< temp_tensor.size(2); k++) {
         for (int q=0; q< temp_tensor.size(3); q++) {
           printf("%f ", temp_tensor[i][j][k][q]);
         }
         printf("\n");
       }
       printf("\n");
     } 
     printf("\n");
    }
    printf("\n");
    free(temp);
}


template<typename DType>
struct find_maxabs {
  MSHADOW_XINLINE static void Map(int i, DType *imin_range, DType* imax_range) {
    if (i < 1){
      *imax_range = MaxAbs(*imin_range, *imax_range);
    }
  }
};

template<typename xpu, typename DType>
void find_max(const OpContext &ctx, const TBlob &data, mshadow::Stream<xpu> *s, 
              mshadow::Tensor<xpu, 1, char> &temp_reduce_space, TBlob &in_min_t, TBlob &in_max_t,
              const mxnet::TShape &src_shape, const mxnet::TShape &dst_shape){
    using namespace mshadow;
    using namespace mshadow::expr;
    broadcast::Reduce<red::minimum, 2, DType, mshadow::op::identity>(
        s, in_min_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data.reshape(src_shape));
    broadcast::Reduce<red::maximum, 2, DType, mshadow::op::identity>(
        s, in_max_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data.reshape(src_shape));

    // the maxabs value is save in in_max_t
    mxnet_op::Kernel<find_maxabs<DType>, xpu>::Launch(s, 1, in_min_t.dptr<DType>(), in_max_t.dptr<DType>());
}

struct MselossPara : public dmlc::Parameter<MselossPara> {
  int nbits;
  float gamma;
  DMLC_DECLARE_PARAMETER(MselossPara) {
    DMLC_DECLARE_FIELD(nbits).set_default(8)
    .describe("the target number of bits of quantization, default to 8.");
    DMLC_DECLARE_FIELD(gamma).set_default(0.001)
    .describe("the coefficient of mseloss weights");
  }
};

template<typename xpu, typename DType>
class MselossOp : public Operator {
 public:
  explicit MselossOp(MselossPara param) {
    this->param_ = param;
    QUANT_LEVEL = std::pow(2, param.nbits) - 1;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[Mseloss_enum::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[Mseloss_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[Mseloss_enum::kData].shape_[0],
                               in_data[Mseloss_enum::kData].shape_[1], 1, 1);
      data = in_data[Mseloss_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[Mseloss_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[Mseloss_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[Mseloss_enum::kOut].get<xpu, 4, DType>(s);
    }
    mshadow::Copy(out, data, s);
  }
  

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, real_t> data_grad;
    Tensor<xpu, 4, real_t> out_data_grad;
    Tensor<xpu, 4, real_t> data;
    Tensor<xpu, 4, real_t> out;
    if (out_grad[Mseloss_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[Mseloss_enum::kOut].shape_[0],
                               out_grad[Mseloss_enum::kOut].shape_[1], 1, 1);
      data = in_data[Mseloss_enum::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[Mseloss_enum::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);

      out_data_grad = out_grad[Mseloss_enum::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      data_grad = in_grad[Mseloss_enum::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[Mseloss_enum::kData].get<xpu, 4, real_t>(s);
      out = out_data[Mseloss_enum::kOut].get<xpu, 4, real_t>(s);

      out_data_grad = out_grad[Mseloss_enum::kOut].get<xpu, 4, real_t>(s);
      data_grad = in_grad[Mseloss_enum::kData].get<xpu, 4, real_t>(s);
    }

    size_t channels = data.shape_[0];
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[Mseloss_enum::kTempSpace].get_space_typed<xpu, 1, uint8_t>(
      Shape1(data.shape_.Size() * sizeof(real_t) + channels * sizeof(real_t)), s);
    
    uint64_t allocated_bytes = 0ULL;
    Tensor<xpu, 4, real_t> quanted_data(reinterpret_cast<real_t*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += data.shape_.Size() * sizeof(real_t);

    Tensor<xpu, 1, real_t> maxs(reinterpret_cast<real_t*>(workspace.dptr_ + allocated_bytes), Shape1(channels), s);
    allocated_bytes += channels * sizeof(real_t);

    maxs = maxall_except_dim<0>(F<mshadow_op::abs>(data));
    maxs = maxs + 1e-6f; // avoid the values of some channel is zero

    const ScalarExp<real_t> quant_level_rev(1.0f / QUANT_LEVEL);
    quanted_data = F<mshadow_op::round>(data/mshadow::expr::broadcast<0>(maxs * quant_level_rev, data.shape_)) * \
                                             mshadow::expr::broadcast<0>(maxs * quant_level_rev, data.shape_);
    

    // mshadow::Copy(data_grad, out_data_grad, s);
    
    // print_data_4D<xpu, real_t>(data, s, ctx, "src data");
    // print_data_4D<xpu, real_t>(quanted_data, s, ctx, "quanted data");
    // print_data_4D<xpu, real_t>(out_data_grad, s, ctx, "out data grad");
    // std::cout << "nbits: " << param_.nbits << ", gamma: " << param_.gamma << ", quant_level: " << QUANT_LEVEL << std::endl;
    Assign(data_grad, req[Mseloss_enum::kData], out_data_grad + (data - quanted_data) * param_.gamma)
  }

 private:
  MselossPara param_;
  int QUANT_LEVEL;

};  // class MselossOp

template<typename xpu>
Operator* CreateOp(MselossPara type, int dtype);

#if DMLC_USE_CXX11
class MselossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;

    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const TShape &dshape = in_shape->at(Mseloss_enum::kData);
    out_shape->clear();
    out_shape->push_back(dshape);
    aux_shape->clear();
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }

    out_type->clear();
    out_type->push_back(dtype);
    aux_type->clear();
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MselossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_Mseloss";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[Mseloss_enum::kOut], 
            in_data[Mseloss_enum::kData], 
            out_data[Mseloss_enum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                           std::vector<int> *in_type) const override;

 private:
  MselossPara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MSELOSS_INL_H_

