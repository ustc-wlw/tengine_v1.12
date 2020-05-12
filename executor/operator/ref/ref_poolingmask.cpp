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
 * Copyright (c) 2019, Open AI Lab
 * Author: haoluo@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/poolingmask.hpp"
#include "kernel/poolingmask/ref_poolingmask_kernel.h"

namespace TEngine {

namespace RefPoolingmaskOps {

struct RefPoolingmask : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct op_data param;
    ref_poolingmask_kernel_t kernel_run;
    KernelRegistry<ref_poolingmask_kernel_t> kernel_registry;

    RefPoolingmask(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefPoolingmask::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    param.layout = layout;
    Poolingmask* poolingmask_op = dynamic_cast<Poolingmask*>(node->GetOp());
    PoolmaskParam* param_ = poolingmask_op->GetParam();
    param.kernels[0] = param_->kernel_h;
    param.kernels[1] = param_->kernel_w;
    param.strides[0] = param_->stride_h;
    param.strides[1] = param_->stride_w;
    param.pads[0] = param_->pad_h;
    param.pads[1] = param_->pad_w;
    if(param_->alg == kPoolmaskMax){
        param.alg = 0;
    }else if (param_->alg == kPoolmaskAvg)
    {
        param.alg = 1;
    }else{
        param.alg = 2;
    }
    
    param.method = param_->alg;
    param.caffe_flavor = param_->caffe_flavor;

    Tensor* input = node->GetInputTensor(0);
    param.batch = input->GetShape().GetN();
    param.channel = input->GetShape().GetC();
    param.input[0] = input->GetShape().GetH();
    param.input[1] = input->GetShape().GetW();

    Tensor* output = node->GetOutputTensor(0);
    param.output[0] = output->GetShape().GetH();
    param.output[1] = output->GetShape().GetW();

    if(input->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input->GetQuantParam();
        param.zero_point = (*quant_param)[0].zero_point;
    }

    auto i_quant = input->GetQuantParam();
    auto o_quant = output->GetQuantParam();
  #if 1
    if(input->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() != 1)
        {
            std::cerr << "Input data_type is INT8 ,and quant param num is not 1 !!!!\n";
            return false;
        }
        o_quant->resize(0);
        o_quant->push_back((*i_quant)[0]);
    }
#endif
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefPoolingmask::Reshape(Node* node)
{
    Poolingmask* poolingmask_op = dynamic_cast<Poolingmask*>(node->GetOp());
    PoolmaskParam* param_ = poolingmask_op->GetParam();
    param.kernels[0] = param_->kernel_h;
    param.kernels[1] = param_->kernel_w;

    Tensor* input = node->GetInputTensor(0);
    param.batch = input->GetShape().GetN();
    param.channel = input->GetShape().GetC();
    param.input[0] = input->GetShape().GetH();
    param.input[1] = input->GetShape().GetW();

    Tensor* output = node->GetOutputTensor(0);
    param.output[0] = output->GetShape().GetH();
    param.output[1] = output->GetShape().GetW();
    return true;
}

bool RefPoolingmask::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    Tensor* input = node->GetInputTensor(0);
    Tensor* output = node->GetOutputTensor(0);
    Tensor* output_mask = node->GetOutputTensor(1);
    const void* input_data = get_tensor_mem(input);
    void* output_data = get_tensor_mem(output);
    void* output_mask_data = get_tensor_mem(output_mask);

    if(kernel_run(input_data, output_data, output_mask_data, &param) < 0)
        return false;

    return true;
}

void RefPoolingmask::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_poolingmask_kernel_t )ref_poolingmask_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    // kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefPoolingmask* ops = new RefPoolingmask();

    LOG_DEBUG() << "Demo RefPoolingmaskOp is selected\n";

    return ops;
}

}    // namespace RefPoolingmaskOps

void RegisterRefPoolingmaskOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Poolingmask", RefPoolingmaskOps::SelectFunc, 8000);
}

}    // namespace TEngine
