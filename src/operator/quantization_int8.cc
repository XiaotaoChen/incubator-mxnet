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
 * \file Quantization_int8.cc
 * \brief
 * \author Jingqiu Zhou
*/

#include "./quantization_int8-inl.h"

#include <nnvm/op_attr_types.h>

namespace mshadow{
template<typename DType>
void quantization_int8_weight(std::string qmod,
                              Tensor<cpu, 3, DType> data,Tensor<cpu, 3, DType> &out,
                              Tensor<cpu, 1, DType> &aux,
                              Stream<cpu> *s,
                              bool init){ 
    //the quantization function
    int dim1 = data.shape_[0];
    int dim2 = data.shape_[1];
    int dim3 = data.shape_[2];
    //find the minimum and maximum
    DType S_min = data[0][0][0];
    DType S_max = data[0][0][0];

    for(int idx1=0;idx1<dim1;idx1++){
        for(int idx2=0;idx2<dim2;idx2++){
            for(int idx3=0;idx3<dim3;idx3++){
                S_min = S_min>data[idx1][idx2][idx3]?data[idx1][idx2][idx3]:S_min;
                S_max = S_max>data[idx1][idx2][idx3]?S_max:data[idx1][idx2][idx3];
            }
        }
    }
    //quantiza the input
    DType S_unit = (S_max-S_min)/255;
    DType temp = 0.;

    for(int idx1=0;idx1<dim1;idx1++){
        for(int idx2=0;idx2<dim2;idx2++){
            for(int idx3=0;idx3<dim3;idx3++){
                temp = floor((data[idx1][idx2][idx3]-S_min)/S_unit+0.5);
                out[idx1][idx2][idx3]=temp*S_unit+S_min;
            }
        }
    }
}

template<typename DType>
void quantization_int8_act(std::string qmod,
                           Tensor<cpu, 3, DType> data,Tensor<cpu, 3, DType> &out,
                           Tensor<cpu, 1, DType> &aux,DType decay,Stream<cpu> *s,
                           int quant_countdown,bool init,bool is_train){
    //the quantization function
    int dim1 = data.shape_[0];
    int dim2 = data.shape_[1];
    int dim3 = data.shape_[2];

    DType S_min = aux[1];
    DType S_max = aux[0];
    //find the minimum and maximum
    if(is_train){
        S_min = data[0][0][0];
        S_max = data[0][0][0];

        for(int idx1=0;idx1<dim1;idx1++){
            for(int idx2=0;idx2<dim2;idx2++){
                for(int idx3=0;idx3<dim3;idx3++){
                    S_min = S_min>data[idx1][idx2][idx3]?data[idx1][idx2][idx3]:S_min;
                    S_max = S_max>data[idx1][idx2][idx3]?S_max:data[idx1][idx2][idx3];
                }
            }
        }

        //quantiza the input
        if(!init){
            S_max = S_max*(1-decay)+aux[0]*decay;
            S_min = S_min*(1-decay)+aux[1]*decay;
        }
        aux[0]= S_max;
        aux[1]= S_min;
    }
    
    DType S_unit = (S_max-S_min)/255;
    DType temp = 0.;
    if(quant_countdown==0){
      for(int idx1=0;idx1<dim1;idx1++){
        for(int idx2=0;idx2<dim2;idx2++){
            for(int idx3=0;idx3<dim3;idx3++){
                temp = floor((data[idx1][idx2][idx3]-S_min)/S_unit+0.5);
                out[idx1][idx2][idx3]=temp*S_unit+S_min;
            }
        }
      }
    } else {
      for(int idx1=0;idx1<dim1;idx1++){
        for(int idx2=0;idx2<dim2;idx2++){
            for(int idx3=0;idx3<dim3;idx3++){
                out[idx1][idx2][idx3]=data[idx1][idx2][idx3];
            }
        }
      }
    }
}

template<typename DType>
void quantization_grad(std::string qmod,
                       Tensor<cpu, 3, DType> gdata,Tensor<cpu, 3, DType> &grad,
                       Tensor<cpu, 3, DType> data,Tensor<cpu, 1, DType> &aux,
                       Stream<cpu> *s){
    std::cout<<"cpu version not implemented"<<std::endl;
}
}

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(Quantization_int8Para param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Quantization_int8Op<cpu, DType>(param);
  });
  return op;
}

Operator *Quantization_int8Prop::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
   std::vector<TShape> out_shape, aux_shape;
   std::vector<int> out_type, aux_type;
   CHECK(InferType(in_type, &out_type, &aux_type));
   CHECK(InferShape(in_shape, &out_shape, &aux_shape));
   DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(Quantization_int8Para);

MXNET_REGISTER_OP_PROPERTY(Quantization_int8, Quantization_int8Prop)
.describe(R"code(perform simulated int8 quatization)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to activation function.")
.add_arguments(Quantization_int8Para::__FIELDS__());

NNVM_REGISTER_OP(Quantization_int8)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 1) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      }
    });

}  // namespace op
}  // namespace mxnet
