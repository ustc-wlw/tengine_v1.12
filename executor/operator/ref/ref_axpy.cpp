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
 * Copyright (c) 2018, Open AI Lab
 * Author: wanglw 2020-04-26
 */
#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/axpy.hpp"

#include "kernel/axpy/ref_axpy_kernel.h"

namespace TEngine
{

namespace RefAxpyOps
{
const int default_prio = 1500;
struct RefAxpy : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefAxpy()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct ref_axpy_param op_param;
    ref_axpy_t kernel_run;
    int8_t** out_data_ptrs;
    KernelRegistry<ref_axpy_t> kernel_registry;
};

void RefAxpy::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_axpy_t )ref_axpy_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
#endif
}

bool RefAxpy::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    
    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefAxpy::Run(Node *node)
    {
        // unsigned long start_time=get_cur_time();
        const Tensor *scale_data_tensor = node->GetInputTensor(0);
        const Tensor *x_data_tensor = node->GetInputTensor(1);
        const Tensor *pro_data_tensor = node->GetInputTensor(2);

        const std::vector<int>& dims=x_data_tensor->GetShape().GetDim();

        Tensor *output_tensor = node->GetOutputTensor(0);
        float* output_data=(float*)get_tensor_mem(output_tensor);

        const std::vector<int>& scale_dims=scale_data_tensor->GetShape().GetDim();
        const std::vector<int>& pro_dims=pro_data_tensor->GetShape().GetDim();

        //scale_data   [b,256,1,1]
        float *scale_data = (float *)get_tensor_mem(scale_data_tensor);
        //x_data [b,256,8,8]
        float *x_data = (float *)get_tensor_mem(x_data_tensor);
        //pro_data   [b,256,8,8]
        float *pro_data = (float *)get_tensor_mem(pro_data_tensor);

        const int batch_dim = dims[0];
        const int channel_dim=dims[1];
        const int spatial_dim =dims[2]*dims[3];

        op_param.batch_number = batch_dim;
        op_param.channel_number = channel_dim;
        op_param.channel_size = spatial_dim;

        int ret = kernel_run(x_data,output_data,scale_data,pro_data,&op_param);
        if(ret < 0){
            return false;
        }
        // for(int n=0; n<batch_dim; ++n){
        //     for(int c=0; c<channel_dim; ++c){
        //         int scale_offset = n * channel_dim + c;
        //         int data_offset = scale_offset*spatial_dim;
        //         for(int s=0; s<spatial_dim; ++s){
        //             output_data[data_offset+s] = scale_data[scale_offset]*x_data[data_offset+s]+pro_data[data_offset+s];
        //         }
        //     }
        // }
        
        return true;
    }

bool RefAxpy::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefAxpy* ops = new RefAxpy();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    LOG_DEBUG() << "RefAxpy is selected\n";

    return ops;
}


} //namespace RefAxpyOps

using namespace RefAxpyOps;

void RegisterRefAxpyOps(void)
{
    RefAxpy *ops = new RefAxpy();

    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME,
                                                  "Axpy", RefAxpyOps::SelectFunc,
                                                  RefAxpyOps::default_prio);
}

} //namespace TEngine
