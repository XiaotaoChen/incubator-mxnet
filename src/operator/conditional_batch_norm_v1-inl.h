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
 * Copyright (c) 2020 by Contributors
 * \file conditional_batch_norm_v1-inl.h
 * \brief
 * \author Xiaotao Chen
*/
#ifndef MXNET_OPERATOR_BATCH_NORM_V1_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_V1_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace cond_batchnorm_v1 {
enum BatchNormOpInputs {kData, kCondition, kThenGamma, kThenBeta, kElseGamma, kElseBeta};
enum BatchNormOpOutputs {kOut, kMean, kVar};
enum BatchNormOpAuxiliary {kThenMovingMean, kThenMovingVar, kElseMovingMean, kElseMovingVar};
enum BatchNormBackResource {kTempSpace};
}  // namespace cond_batchnorm_v1

struct CondBatchNormV1Param : public dmlc::Parameter<CondBatchNormV1Param> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  bool share_weight;
  DMLC_DECLARE_PARAMETER(CondBatchNormV1Param) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(share_weight).set_default(false)
    .describe("share gamma/beta or not.");
  }
};

template<typename xpu>
class CondBatchNormV1Op : public Operator {
 public:
  explicit CondBatchNormV1Op(CondBatchNormV1Param param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 6U);
    CHECK_EQ(aux_states.size(), 4U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
      CHECK_EQ(req[cond_batchnorm_v1::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[cond_batchnorm_v1::kData].shape_[1]) /
                         static_cast<real_t>(in_data[cond_batchnorm_v1::kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[cond_batchnorm_v1::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[cond_batchnorm_v1::kData].shape_[0],
                               in_data[cond_batchnorm_v1::kData].shape_[1], 1, 1);
      data = in_data[cond_batchnorm_v1::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[cond_batchnorm_v1::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[cond_batchnorm_v1::kData].get<xpu, 4, real_t>(s);
      out = out_data[cond_batchnorm_v1::kOut].get<xpu, 4, real_t>(s);
    }
    Tensor<xpu, 1> condition = in_data[cond_batchnorm_v1::kCondition].get<xpu, 1, real_t>(s);
    
    Tensor<cpu, 1> cpu_condition = mshadow::NewTensor<cpu, real_t>(condition.shape_, 0.0f);
    mshadow::Copy(cpu_condition, condition, s);

    Tensor<xpu, 1> slope, bias, moving_mean, moving_var;
    if (cpu_condition[0] > 0.5) {
      slope = in_data[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
      bias = in_data[cond_batchnorm_v1::kThenBeta].get<xpu, 1, real_t>(s);
      moving_mean = aux_states[cond_batchnorm_v1::kThenMovingMean].get<xpu, 1, real_t>(s);
      moving_var = aux_states[cond_batchnorm_v1::kThenMovingVar].get<xpu, 1, real_t>(s);
    }
    else {
      if (param_.share_weight) {
        slope = in_data[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
        bias = in_data[cond_batchnorm_v1::kThenBeta].get<xpu, 1, real_t>(s);
      }
      else {
        slope = in_data[cond_batchnorm_v1::kElseGamma].get<xpu, 1, real_t>(s);
        bias = in_data[cond_batchnorm_v1::kElseBeta].get<xpu, 1, real_t>(s);
      }
      moving_mean = aux_states[cond_batchnorm_v1::kElseMovingMean].get<xpu, 1, real_t>(s);
      moving_var = aux_states[cond_batchnorm_v1::kElseMovingVar].get<xpu, 1, real_t>(s);
    }

    if (param_.fix_gamma) slope = 1.f;

    // whether use global statistics
    if (ctx.is_train && !param_.use_global_stats) {
      Tensor<xpu, 1> mean = out_data[cond_batchnorm_v1::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[cond_batchnorm_v1::kVar].get<xpu, 1, real_t>(s);
      CHECK(req[cond_batchnorm_v1::kMean] == kNullOp || req[cond_batchnorm_v1::kMean] == kWriteTo);
      CHECK(req[cond_batchnorm_v1::kVar] == kNullOp || req[cond_batchnorm_v1::kVar] == kWriteTo);
      // The first three steps must be enforced.
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(
          data - broadcast<1>(mean, data.shape_)));
      Assign(out, req[cond_batchnorm_v1::kOut], broadcast<1>(slope, out.shape_) *
             (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
             broadcast<1>(bias, out.shape_));
    } else {
      Assign(out, req[cond_batchnorm_v1::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
      // Set mean and var tensors to their moving values
      Tensor<xpu, 1> mean = out_data[cond_batchnorm_v1::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[cond_batchnorm_v1::kVar].get<xpu, 1, real_t>(s);
      mean = F<mshadow_op::identity>(moving_mean);
      var  = F<mshadow_op::identity>(moving_var);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_GE(in_data.size(), 6U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_GE(in_grad.size(), 6U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    const real_t scale = static_cast<real_t>(out_grad[cond_batchnorm_v1::kOut].shape_[1]) /
                         static_cast<real_t>(out_grad[cond_batchnorm_v1::kOut].shape_.Size());
    if (in_data[cond_batchnorm_v1::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[cond_batchnorm_v1::kOut].shape_[0],
                               out_grad[cond_batchnorm_v1::kOut].shape_[1], 1, 1);
      data = in_data[cond_batchnorm_v1::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[cond_batchnorm_v1::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[cond_batchnorm_v1::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[cond_batchnorm_v1::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[cond_batchnorm_v1::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[cond_batchnorm_v1::kData].get<xpu, 4, real_t>(s);
    }

    Tensor<xpu, 1> condition = in_data[cond_batchnorm_v1::kCondition].get<xpu, 1, real_t>(s);
    Tensor<cpu, 1> cpu_condition = mshadow::NewTensor<cpu, real_t>(condition.shape_, 0.0f);
    mshadow::Copy(cpu_condition, condition, s);


    Tensor<xpu, 1> mean = out_data[cond_batchnorm_v1::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[cond_batchnorm_v1::kVar].get<xpu, 1, real_t>(s);

    Tensor<xpu, 1> slope, gslope, gbias, moving_mean, moving_var;
    auto gamma_req = OpReqType::kWriteTo;
    auto beta_req = OpReqType::kWriteTo;

    if (cpu_condition[0] > 0.5) {
      slope = in_data[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
      gslope = in_grad[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
      gbias = in_grad[cond_batchnorm_v1::kThenBeta].get<xpu, 1, real_t>(s);
      gamma_req = req[cond_batchnorm_v1::kThenGamma];
      beta_req = req[cond_batchnorm_v1::kThenBeta];
      moving_mean = aux_states[cond_batchnorm_v1::kThenMovingMean].get<xpu, 1, real_t>(s);
      moving_var = aux_states[cond_batchnorm_v1::kThenMovingVar].get<xpu, 1, real_t>(s);
    }
    else {
      if (param_.share_weight) {
        slope = in_data[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
        gslope = in_grad[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
        gbias = in_grad[cond_batchnorm_v1::kThenBeta].get<xpu, 1, real_t>(s);
        gamma_req = req[cond_batchnorm_v1::kThenGamma];
        beta_req = req[cond_batchnorm_v1::kThenBeta];
      }
      else {
        slope = in_data[cond_batchnorm_v1::kElseGamma].get<xpu, 1, real_t>(s);
        gslope = in_grad[cond_batchnorm_v1::kElseGamma].get<xpu, 1, real_t>(s);
        gbias = in_grad[cond_batchnorm_v1::kElseBeta].get<xpu, 1, real_t>(s);
        gamma_req = req[cond_batchnorm_v1::kElseGamma];
        beta_req = req[cond_batchnorm_v1::kElseBeta];
        
        // assign 0.0f gradients to unruned branch
        Tensor<xpu, 1> unruned_gslope = in_grad[cond_batchnorm_v1::kThenGamma].get<xpu, 1, real_t>(s);
        Tensor<xpu, 1> unruned_gbias = in_grad[cond_batchnorm_v1::kThenBeta].get<xpu, 1, real_t>(s);
        Assign(unruned_gslope, req[cond_batchnorm_v1::kThenGamma], 0.0f);
        Assign(unruned_gbias, req[cond_batchnorm_v1::kThenBeta], 0.0f);
      }
      moving_mean = aux_states[cond_batchnorm_v1::kElseMovingMean].get<xpu, 1, real_t>(s);
      moving_var = aux_states[cond_batchnorm_v1::kElseMovingVar].get<xpu, 1, real_t>(s);
    }

    if (param_.fix_gamma) slope = 1.f;

    if (ctx.is_train && !param_.use_global_stats) {
      // get requested temp space
      Tensor<xpu, 2> workspace = ctx.requested[cond_batchnorm_v1::kTempSpace].get_space<xpu>(
          mshadow::Shape2(3, mean.shape_[0]), s);
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];
      Tensor<xpu, 1> tmp = workspace[2];

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
      // cal
      gvar = sumall_except_dim<1>((grad * broadcast<1>(slope, data.shape_)) *
                                  (data - broadcast<1>(mean, data.shape_)) *
                                  -0.5f *
                                  F<mshadow_op::power>(broadcast<1>(var + param_.eps, data.shape_),
                                                       -1.5f));
      gmean = sumall_except_dim<1>(grad * broadcast<1>(slope, data.shape_));
      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
      tmp = scale * sumall_except_dim<1>(-2.0f * (data - broadcast<1>(mean, data.shape_)));
      tmp *= gvar;
      gmean += tmp;
      // assign
      if (!param_.fix_gamma) {
        Assign(gslope, gamma_req,
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, gamma_req, 0.0f);
      }
      Assign(grad_in, req[cond_batchnorm_v1::kData],
             (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                   data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      Assign(gbias, beta_req, sumall_except_dim<1>(grad));
    } else {
      // use global statistics with freeze moving mean and var.
      if (!param_.fix_gamma) {
        Assign(gslope, gamma_req,
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(moving_mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, gamma_req, 0.0f);
      }
      Assign(gbias, beta_req, sumall_except_dim<1>(grad));
      Assign(grad_in, req[cond_batchnorm_v1::kData], (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(
                 1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

 private:
  CondBatchNormV1Param param_;
};  // class CondBatchNormV1Op

template<typename xpu>
Operator *CreateOp(CondBatchNormV1Param param, int dtype);


#if DMLC_USE_CXX11
class CondBatchNormV1Prop : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 6U) << "Input:[data, condition, then_gamma, then_beta, else_gamma, else_beta]";
    const mxnet::TShape &dshape = in_shape->at(0);
    if (!mxnet::ndim_is_known(dshape)) return false;

    // batch_size
    in_shape->at(1) = mxnet::TShape(Shape1(dshape[0]));
    for (size_t i=2; i<in_shape->size(); i++) {
      in_shape->at(i) = mxnet::TShape(Shape1(dshape[1]));
    }
    
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;
    for (size_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    for (size_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        UNIFORM_TYPE_CHECK((*aux_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype_param);
    int n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (int i = 1; i < n_out; ++i ) out_type->push_back(dtype_param);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CondBatchNormV1Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Conditional_BatchNorm_v1";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[cond_batchnorm_v1::kOut],
            out_data[cond_batchnorm_v1::kMean],
            out_data[cond_batchnorm_v1::kVar],
            in_data[cond_batchnorm_v1::kData],
            in_data[cond_batchnorm_v1::kThenGamma],
            in_data[cond_batchnorm_v1::kElseGamma]
           };
  }

  std::vector<ResourceRequest> BackwardResource(
      const mxnet::ShapeVector &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "condition", "then_gamma", "then_beta", "else_gamma", "else_beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"then_moving_mean", "then_moving_var", "else_moving_mean", "else_moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
      std::vector<int> *in_type) const override;

  inline const CondBatchNormV1Param& getParam() const {
    return param_;
  }

 private:
  CondBatchNormV1Param param_;
};  // class CondBatchNormV1Prop

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_V1_INL_H_
