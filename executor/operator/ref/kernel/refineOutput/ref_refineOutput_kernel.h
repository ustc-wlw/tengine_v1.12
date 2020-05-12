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

#ifndef __REF_REFINEOUTPUT_KERNEL_H__
#define __REF_REFINEOUTPUT_KERNEL_H__

#include <stdint.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

struct ddo_param
{
    int num_classes;
    int keep_top_k;
    int nms_top_k;
    float confidence_threshold; // 第二阶段类别置信度阈值
    float nms_threshold;
    float objectness_threshold;// 第一阶段前景背景分类阈值
    float out_scale;
    float scale[3];
    std::vector<Box> bbox_rects;
};

typedef int (*ref_RefineOutput_kernel_t)(const void* odm_location, const void* odm_confidence,const void* location, const void* confidence, const void* priorbox,
                                            std::vector<int> dims, ddo_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_refineOutput_fp32.c"
#endif

// #ifdef CONFIG_KERNEL_FP16
// #include "ref_detectionOutput_fp16.c"
// #endif
// #ifdef CONFIG_KERNEL_INT8
// #include "ref_detectionOutput_int8.c"
// #endif

#ifdef __cplusplus
}
#endif

#endif
