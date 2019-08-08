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
 * Copyright (c) 2015 by Contributors
 * \file quantization_int8-inl.h
 * \brief
 * \author Jingqiu Zhou
*/
#ifndef MXNET_OPERATOR_QUANTIZATION_INT8_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_INT8_INL_H_


#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./tensor/elemwise_binary_broadcast_op.h"
#include "../common/random_generator.h"
#include "./random/sampler.h"
#include "./random/sample_op.h"
#include<cuda.h>

namespace mxnet {
namespace op {

namespace Quantization_int8 {
enum Quantization_int8OpInputs {kData};
enum Quantization_int8OpOutputs {kOut};
enum Quantization_int8OpAuxiliary {kMinmax};
enum Quantization_int8OpResource {kRandom};
}  // namespace leakyrelu

struct Quantization_int8Para : public dmlc::Parameter<Quantization_int8Para> {
  // use int for enumeration
  bool is_weight;
  bool is_train;
  int delay_quant;
  float ema_decay;
  std::string quant_mod; 
  DMLC_DECLARE_PARAMETER(Quantization_int8Para) {
    DMLC_DECLARE_FIELD(is_weight).set_default(true)
    .describe("if true, this quantization layer is used for weight");
    DMLC_DECLARE_FIELD(is_train).set_default(true)
    .describe("if true, this quantization layer is used for training");
    DMLC_DECLARE_FIELD(delay_quant).set_default(2000)
    .describe("number of steps before quatization is used");
    DMLC_DECLARE_FIELD(ema_decay).set_default(0.9)
    .describe("the rate at which quantization range decay in ema");
    DMLC_DECLARE_FIELD(quant_mod).set_default("power2")
    .describe("select quantization mode");
  }
};

template<typename xpu, typename DType>
class Quantization_int8Op : public Operator {
 public:
  explicit Quantization_int8Op(Quantization_int8Para param) {
    param_ = param;
    quant_countdown = param_.delay_quant;
    decay_rate=DType(param_.ema_decay);
    init=true;
    is_train=param_.is_train;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(in_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 3, DType> out;
    Tensor<xpu, 1, DType> aux;

    int n = in_data[Quantization_int8::kData].shape_[0];
    int k = (in_data[Quantization_int8::kData].ndim() > 1) ? in_data[Quantization_int8::kData].shape_[1] : 1;
    Shape<3> dshape = Shape3(n, k, in_data[Quantization_int8::kData].Size()/n/k);
    data = in_data[Quantization_int8::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    out = out_data[Quantization_int8::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    aux = aux_args[Quantization_int8::kMinmax].get<xpu, 1, DType>(s);
    if(param_.is_weight){
        quantization_int8_weight(param_.quant_mod,data,out,aux,s,init);
        init=false;
    } else {
        quantization_int8_act(param_.quant_mod,data,out,aux,decay_rate,s,quant_countdown,init,is_train);
        quant_countdown=quant_countdown>0?quant_countdown-1:quant_countdown;
        init = false;
    }
    fflush(stdout);
  }
  

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(out_grad.size(), 1U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
 

    Tensor<xpu, 3, DType> gdata;
    Tensor<xpu, 3, DType> grad;
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 1, DType> aux;
    int n = out_grad[Quantization_int8::kOut].shape_[0];
    int k = (out_grad[Quantization_int8::kOut].ndim() > 1) ? out_grad[Quantization_int8::kOut].shape_[1] : 1;
    Shape<3> dshape = Shape3(n, k, out_grad[Quantization_int8::kOut].Size()/n/k);
    grad = out_grad[Quantization_int8::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    gdata = in_grad[Quantization_int8::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    data = in_data[Quantization_int8::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    aux = aux_args[Quantization_int8::kMinmax].get<xpu, 1, DType>(s);
    if(param_.quant_mod==std::string("minmax")){
      mshadow::Copy(gdata, grad, s);
    } else if(param_.quant_mod==std::string("power2")){
      quantization_grad(param_.quant_mod,gdata,grad,
                        data,aux,
                        s);
    }
    fflush(stdout);
  }

 private:
  Quantization_int8Para param_;
  int quant_countdown;
  DType decay_rate;
  bool init;
  bool is_train;

};  // class LeakyReLUOp

template<typename xpu>
Operator* CreateOp(Quantization_int8Para type, int dtype);

#if DMLC_USE_CXX11
class Quantization_int8Prop : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(Quantization_int8::kData);
    const Shape<1> dshape_aux = Shape1(3);
    //const TShape dshape_aux = shape_aux;
    
    if (dshape.ndim() == 0) return false;

    out_shape->clear();
    out_shape->push_back(dshape);
    
    aux_shape->clear();
    aux_shape->push_back(TShape(dshape_aux));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = -1;
    for (const int& type : *in_type) {
      type_assign(&dtype, type);
    }
    for (const int& type : *out_type) {
      type_assign(&dtype, type);
    }
    for (const int& type : *aux_type) {
      type_assign(&dtype, type);
    }

    for (size_t i = 0; i < in_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, dtype);
    }
    for (size_t i = 0; i < out_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*out_type, i, dtype);
    }
    for (size_t i = 0; i < aux_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*aux_type, i, dtype);
    }
    return dtype != -1;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new Quantization_int8Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Quantization_int8";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[Quantization_int8::kOut], out_data[Quantization_int8::kData]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[Quantization_int8::kOut], in_grad[Quantization_int8::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {

    return {{in_data[Quantization_int8::kData], out_data[Quantization_int8::kOut]}};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"minmax"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return std::vector<ResourceRequest>();
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
  Quantization_int8Para param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Qunatization_Int8_INL_H_

