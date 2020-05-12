/**
 * 根据预选框和算法输出的预选框偏移量计算物体检测框的位置
 * 预选框偏移: loc_ptr
 * 预选框个数： num_prior
 * 预选框初始位置及计算偏移量的variance值: prior_ptr
 * 回归后输出的检测框: boxes
 * */
void get_boxes2(std::vector<Box>& boxes, int num_prior, const float* loc_ptr, const float* prior_ptr)
{
    for(int i = 0; i < num_prior; i++)
    {
        const float* loc = loc_ptr + i * 4;
        const float* pbox = prior_ptr + i * 4;
        const float* pvar = pbox + num_prior * 4;
        // center size
        // pbox [xmin,ymin,xmax,ymax]
        float pbox_w = pbox[2] - pbox[0];
        float pbox_h = pbox[3] - pbox[1];
        float pbox_cx = (pbox[0] + pbox[2]) * 0.5f;
        float pbox_cy = (pbox[1] + pbox[3]) * 0.5f;

        // loc []
        float bbox_cx = pvar[0] * loc[0] * pbox_w + pbox_cx;
        float bbox_cy = pvar[1] * loc[1] * pbox_h + pbox_cy;
        float bbox_w = pbox_w * exp(pvar[2] * loc[2]);
        float bbox_h = pbox_h * exp(pvar[3] * loc[3]);
        // bbox [xmin,ymin,xmax,ymax]
        boxes[i].x0 = bbox_cx - bbox_w * 0.5f;
        boxes[i].y0 = bbox_cy - bbox_h * 0.5f;
        boxes[i].x1 = bbox_cx + bbox_w * 0.5f;
        boxes[i].y1 = bbox_cy + bbox_h * 0.5f;
    }
}

/**
 * 预选框偏移: loc_ptr
 * 预选框个数： num_prior
 * 预选框初始位置及计算偏移量的variance值: prior_ptr
 * 回归后输出的检测框: bbox
 * refineDet网络第一次预选框位置回归后预选框位置：decode_prior_ptr
 * */
void get_decoded_boxes(float* decode_prior_ptr,int num_prior, const float* loc_ptr, const float* prior_ptr){
    for(int i = 0;i<num_prior;i++){
        const float* loc = loc_ptr + i * 4;
        const float* pbox = prior_ptr + i * 4;
        const float* pvar = pbox + num_prior * 4;
        float* dpbox = decode_prior_ptr + i * 4;
        float* dpvar = dpbox + num_prior * 4;

        float pbox_w = pbox[2] - pbox[0];
        float pbox_h = pbox[3] - pbox[1];
        float pbox_cx = (pbox[0] + pbox[2]) * 0.5f;
        float pbox_cy = (pbox[1] + pbox[3]) * 0.5f;

        float bbox_cx = pvar[0] * loc[0] * pbox_w + pbox_cx;
        float bbox_cy = pvar[1] * loc[1] * pbox_h + pbox_cy;
        float bbox_w = pbox_w * exp(pvar[2] * loc[2]);
        float bbox_h = pbox_h * exp(pvar[3] * loc[3]);

        dpbox[0] = bbox_cx - bbox_w * 0.5f;
        dpbox[1] = bbox_cy - bbox_h*0.5f;
        dpbox[2] = bbox_cx + bbox_w*0.5f;
        dpbox[3] = bbox_cy + bbox_h*0.5f;

        dpvar[0] = pvar[0];
        dpvar[1] = pvar[1];
        dpvar[2] = pvar[2];
        dpvar[3] = pvar[3];
    }
}

static inline float intersection_area2(const Box& a, const Box& b)
{
    if(a.x0 > b.x1 || a.x1 < b.x0 || a.y0 > b.y1 || a.y1 < b.y0)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
    float inter_height = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);

    return inter_width * inter_height;
}

void nms_sorted_bboxes2(const std::vector<Box>& bboxes, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = bboxes.size();

    std::vector<float> areas(n);
    for(int i = 0; i < n; i++)
    {
        const Box& r = bboxes[i];

        float width = r.x1 - r.x0;
        float height = r.y1 - r.y0;

        areas[i] = width * height;
    }

    for(int i = 0; i < n; i++)
    {
        const Box& a = bboxes[i];

        int keep = 1;
        for(int j = 0; j < ( int )picked.size(); j++)
        {
            const Box& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area2(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if(inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if(keep)
            picked.push_back(i);
    }
}

int ref_RefineOutput_fp32(const float* odm_location, const float* odm_confidence, const float* location, const float* confidence, const float* priorbox,
                             std::vector<int> dims, ddo_param* param_)
{
    const int num_priorx4 = dims[2];
    const int num_prior = num_priorx4 / 4;
    const int num_classes = param_->num_classes;

    int b = 0;
    const float* loc_ptr = location + b * num_priorx4;
    const float* conf_ptr = confidence + b * num_prior * num_classes;
    const float* prior_ptr = priorbox + b * num_priorx4 * 2;

    const float* odm_loc_ptr = odm_location + b * num_priorx4;
    const float* odm_conf_ptr = odm_confidence + b * num_prior * 2;

    float* decode_prior_ptr = (float*)malloc(sizeof(float)*num_priorx4*2);
    get_decoded_boxes(decode_prior_ptr,num_prior,odm_loc_ptr,prior_ptr);

    std::vector<Box> boxes(num_prior);
    get_boxes2(boxes, num_prior, loc_ptr, decode_prior_ptr);

    std::vector<std::vector<Box>> temp_all_box;
    temp_all_box.clear();
    temp_all_box.resize(num_classes);

    for(int i = 1; i < num_classes; i++)
    {
        std::vector<Box> class_box;
        for(int j = 0; j < num_prior; j++)
        {
            float score = conf_ptr[j * num_classes + i];
            float object_socre = odm_conf_ptr[j * 2 + 1];// 前景类置信度

            if(score > param_->confidence_threshold && object_socre > param_->objectness_threshold)
            {
                // 只有前景类得分置信度和具体目标类别得分都大于阈值，该预选框才作为候选前景框
                boxes[j].score = score;
                boxes[j].class_idx = i;
                class_box.push_back(boxes[j]);
            }
        }
        // sort
        std::sort(class_box.begin(), class_box.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

        if(param_->nms_top_k < ( int )class_box.size())
        {
            class_box.resize(param_->nms_top_k);
        }
        // apply nms

        std::vector<int> picked;
        nms_sorted_bboxes2(class_box, picked, param_->nms_threshold);
        // select

        for(int j = 0; j < ( int )picked.size(); j++)
        {
            int z = picked[j];
            temp_all_box[i].push_back(class_box[z]);
        }
    }

    param_->bbox_rects.clear();
    for(int i = 0; i < param_->num_classes; i++)
    {
        const std::vector<Box> class_bbox_rects = temp_all_box[i];
        param_->bbox_rects.insert(param_->bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
    }

    // global sort inplace
    std::sort(param_->bbox_rects.begin(), param_->bbox_rects.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });

    // keep_top_k
    if(param_->keep_top_k < ( int )param_->bbox_rects.size())
    {
        param_->bbox_rects.resize(param_->keep_top_k);
    }

    free(decode_prior_ptr);
    return 0;
}
