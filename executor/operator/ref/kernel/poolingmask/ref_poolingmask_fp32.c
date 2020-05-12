
void Generic_AvgPool(const float *input, float *output,
                            int inc, int inh, int inw,
                            int outh, int outw,
                            int k_h, int k_w,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w)
{
    int in_hw = inh * inw;
    int out_hw = outh * outw;
    for (int c = 0; c < inc; c++)
    {
        int c_skip = c * in_hw;
        int oc_skip = c * out_hw;

        for (int ph = 0; ph < outh; ph++)
        {
            for (int pw = 0; pw < outw; pw++)
            {
                int h_start = ph * stride_h - pad_h;
                int h_end = h_start + k_h;
                if(h_end > inh + pad_h){
                    h_end = inh + pad_h;
                }
                int w_start = pw * stride_w - pad_w;
                int w_end = w_start + k_w;
                if(w_end > inw + pad_w){
                    w_end = inw + pad_w;
                }
                int pool_size = (h_end - h_start) * (w_end - w_start);

                h_start = h_start > 0 ? h_start : 0;
                w_start = w_start > 0 ? w_start : 0;
                h_end = h_end < inh ? h_end : inh;
                w_end = w_end < inw ? w_end : inw;

                const int out_index = oc_skip + ph * outw + pw;
                output[out_index] = 0.f;
                for (int h = h_start; h < h_end; h++)
                {
                    for (int w = w_start; w < w_end; w++)
                    {
                        output[out_index] += input[c_skip + h * inw + w];
                    }
                } // end ksize_h,ksize_w
                output[out_index] /= pool_size;
            }
        }
    }
}

void Generic_MaxPool(const float *input, float *output,float* mask,
                            int inc, int inh, int inw,
                            int outh, int outw,
                            int k_h, int k_w,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w)
{
    int in_hw = inh * inw;
    int out_hw = outh * outw;
    for (int c = 0; c < inc; c++)
    {
        int c_skip = c * in_hw;
        int oc_skip = c * out_hw;

        for (int ph = 0; ph < outh; ph++)
        {
            int h_start = ph * stride_h - pad_h;
            int h_end = h_start + k_h;
            if(h_end > inh){
                h_end = inh;
            }
            h_start = h_start > 0 ? h_start : 0;

            for (int pw = 0; pw < outw; pw++)
            {
                int w_start = pw * stride_w - pad_w;
                int w_end = w_start + k_w;
                if(w_end > inw){
                    w_end = inw;
                }
                w_start = w_start > 0 ? w_start : 0;

                const int out_index = oc_skip + ph * outw + pw;
                output[out_index] = input[c_skip + h_start * inw + w_start];
                mask[out_index] = (c_skip + h_start * inw + w_start);
                for (int h = h_start; h < h_end; h++)
                {
                    for (int w = w_start; w < w_end; w++)
                    {
                        int in_index = c_skip + h * inw + w;

                        if (input[in_index] > output[out_index])
                        {

                            output[out_index] = input[in_index];
                            mask[out_index] = in_index;
                        }
                    }
                } // end ksize_h,ksize_w
            }
        }
    }
}

void Global_MaxPool(const float*input,float* output,float* mask,
    int inc, int in_hw)
{
    // float* out_ptr = output;
    // float* in_ptr = input;
    // for(int c=0;c<inc;c++)
    // {
    //     float max_ = in_ptr[0];
    //     for(int j=0;j<in_hw;j++)
    //     {
    //         max_=std::max(max_,in_ptr[0]);
    //         in_ptr++;
    //     }
    //     *out_ptr=max_;
    //     out_ptr++;
    // }

    float max = 0.0f;
    float tmp = 0.0f;
    for(int c = 0; c < inc; c++){
        max = input[c * in_hw];
        for(int j = 0; j< in_hw; j++){
            tmp = input[c*in_hw + j];
            max = max > tmp ? max : tmp;
        }
        output[c] = max;
    }
}

void Global_AvgPool(const float*input,float* output,
    int inc,int in_hw)
{
    // float* out_ptr = output;
    // float* in_ptr = input;
    // for(int c=0;c<inc;c++)
    // {
    //     float sum=0.f;
    //     for(int j=0;j<in_hw;j++)
    //     {
    //         sum+=in_ptr[0];
    //         in_ptr++;
    //     }
    //     *out_ptr=sum/in_hw;
    //     out_ptr++;
    // }
    for(int c = 0;c < inc; c++){
        float sum = 0.f;
        for(int j= 0;j<in_hw;j++){
            sum += input[c*in_hw + j];
        }
        output[c] = sum / in_hw;
    }
}

static int ref_poolingmask_fp32(const float* input, float* output,float* output_mask, struct op_data* param)
{
    int input_chw = param->channel * param->input[0] * param->input[1];
    int output_chw = param->channel * param->output[0] * param->output[1];
    int in_hw = param->input[0] * param->input[1];

    for(int n = 0; n < param->batch; n++)
    {
        const float* input_cur = input + n * input_chw;
        float* output_cur = output + n*output_chw;
        float* output_mask_cur = output_mask + n*output_chw;
        
        if(param->global){
            if(param->alg == 0){// Gloabel MaxPooling
                Global_MaxPool(input_cur,output_cur,output_mask_cur,param->channel,in_hw);
            }
            else if (param->alg == 1) // Global AveragePooling
            {
                Global_AvgPool(input_cur,output_cur,param->channel,in_hw);
            }
        }
        else if(param->alg == 0){// MaxPooling
            Generic_MaxPool(input_cur,output_cur,output_mask_cur,
                            param->channel,param->input[0],param->input[1],
                            param->output[0],param->output[1],
                            param->kernels[0],param->kernels[1],
                            param->strides[0], param->strides[1],
                            param->pads[0], param->pads[1]);
        }
        else if(param->alg == 1){// AveragePooling
            Generic_AvgPool(input_cur,output_cur,
                            param->channel,param->input[0],param->input[1],
                            param->output[0],param->output[1],
                            param->kernels[0],param->kernels[1],
                            param->strides[0], param->strides[1],
                            param->pads[0], param->pads[1]);

        }else{
            printf("Pooling Type Error!!!\n");
            return 0;
        }
        
    }
    return 0;
}
