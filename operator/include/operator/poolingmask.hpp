/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 *         chunyinglv@openailab.com
 */
#ifndef __POOLINGMASK_HPP__
#define __POOLINGMASK_HPP__


#include "operator.hpp"
#include "poolingmask_param.hpp"

namespace TEngine {

class Poolingmask: public OperatorWithParam<Poolingmask, PoolmaskParam> {

public:

     Poolingmask () { name_="Poolingmask"; }
     Poolingmask(const Poolingmask& src)=default;

     virtual ~Poolingmask() {}

     void MethodToAlg(PoolmaskParam& param)
     {
         std::string& method=param.method;

         /* default alg */
         param.alg=kPoolmaskMax;

         if(method == "avg")
             param.alg=kPoolmaskAvg;
         else if(method == "rand")
             param.alg=kPoolmaskRand;
     }

     bool InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout) override;
     float GetFops(const std::vector<TEngine::TShape>& ishape, const std::vector<TEngine::TShape>& oshape) override;

     void SetSchema(void) override;

     void ParseParam(PoolmaskParam & param, Operator * op) override
     {
         ParsePredefinedParam(param,op);
         MethodToAlg(param);
         
         /* translate to onnx parameters */
         param.kernel_shape.resize(2);

         param.kernel_shape[0]=param.kernel_h;
         param.kernel_shape[1]=param.kernel_w;

         param.strides.resize(2);
         param.strides[0]=param.stride_h;
         param.strides[1]=param.stride_w;

         param.pads.resize(4);
         param.pads[0]=param.pad_h;
         param.pads[1]=param.pad_w;
         param.pads[2]=param.pad_h;
         param.pads[3]=param.pad_w;
         
     }

};


} //namespace TEngine


#endif
