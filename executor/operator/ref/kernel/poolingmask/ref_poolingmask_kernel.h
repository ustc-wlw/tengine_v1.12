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
 * Author: wanglw at 20200427
 */

#ifndef __REF_POOLING_KERNEL_H__
#define __REF_POOLING_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
struct op_data
{
    int layout; // default nchw
    int batch;
    int channel;
    int method;
    int input[2];// input height width
    int output[2];// output height width
    int kernels[2];// kernel_h  kernel_w
    int strides[2];// stride_h  stride_w
    int pads[2];// stride_h  stride_w
    int caffe_flavor;
    int zero_point;
    int align[4];

    int global;// global pooling = 1;
    int alg;// Maxpooling = 0; AveragePooling = 1
};

typedef int (*ref_poolingmask_kernel_t)(const void* input, void* output,void* output_mask, struct op_data* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_poolingmask_fp32.c"
#endif


#ifdef __cplusplus
}
#endif

#endif
