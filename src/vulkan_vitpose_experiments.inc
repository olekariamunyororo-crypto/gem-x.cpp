// Validated defaults, with environment overrides for projection experiments.
if (!device->fp16 && !device->coopmat2 && !device->coopmat_support &&
    strcmp(device->properties.deviceName.data(),"NVIDIA GeForce RTX 5070 Ti")==0) {
    const char *norm=getenv("GEMX_VITPOSE_NORM");
    if (!norm || strcmp(norm,"1")==0)
        ggml_vk_create_pipeline(device,device->gemx_norm_pipeline,"gemx_norm_affine",
            sizeof(gemx_norm_spv),reinterpret_cast<const unsigned char *>(gemx_norm_spv),"main",4,8,{1,1,1},{},1);
    const char *tile=getenv("GEMX_VITPOSE_RECT");
    if (!tile) tile="64x64";
    std::vector<uint32_t> spec;
    if(tile && strcmp(tile,"128x32")==0) spec={128,128,32,16,32,32,2,4,2,1,32};
    if(tile && strcmp(tile,"64x32")==0) spec={64,64,32,16,32,32,2,4,2,1,32};
    if(tile && strcmp(tile,"32x64")==0) spec={64,32,64,16,32,32,2,4,2,1,32};
    if(tile && strcmp(tile,"64x64")==0) spec={64,64,64,16,32,64,2,4,2,1,32};
    if(tile && strcmp(tile,"128x64")==0) spec={128,128,64,16,64,32,2,4,2,1,32};
    if(tile && strcmp(tile,"32x32")==0) spec={32,32,32,16,32,32,2,2,2,1,32};
    if(tile && strcmp(tile,"32x128")==0) spec={128,32,128,16,32,32,2,4,2,1,32};
    if(tile && strcmp(tile,"64x128")==0) spec={128,64,128,16,32,64,2,4,4,1,32};
    const char *bk=getenv("GEMX_VITPOSE_RECT_BK");
    if (!spec.empty() && bk && strcmp(bk,"32")==0 && spec[1]+spec[2]<=160) spec[3]=32;
    if (!spec.empty() && bk && strcmp(bk,"64")==0 && spec[1]+spec[2]<=64) spec[3]=64;
    if (!spec.empty())
        ggml_vk_create_pipeline(device,device->gemx_rect_pipeline,"gemx_projection_rect",
            matmul_f32_f32_fp32_len,matmul_f32_f32_fp32_data,"main",3,sizeof(vk_mat_mat_push_constants),
            {spec[1],spec[2],1},ggml_vk_mul_mm_spec(spec,true),32);
}

// QKV register-microtile experiments, independent of the expansion tile.
if (!device->fp16 && !device->coopmat2 && !device->coopmat_support &&
    strcmp(device->properties.deviceName.data(),"NVIDIA GeForce RTX 5070 Ti")==0) {
    const char *layout=getenv("GEMX_VITPOSE_QKV_LAYOUT");
    if (!layout) layout="64x64-t128";
    std::vector<uint32_t> spec;
    if(layout && strcmp(layout,"128x64-t256")==0) spec={256,128,64,16,32,32,1,4,2,1,32};
    if(layout && strcmp(layout,"64x64-t128")==0) spec={128,64,64,16,32,32,1,4,2,1,32};
    if(layout && strcmp(layout,"64x32-t128")==0) spec={128,64,32,16,32,16,1,2,2,1,32};
    if(layout && strcmp(layout,"128x32-w1")==0) spec={128,128,32,16,32,32,1,4,2,1,32};
    if(layout && strcmp(layout,"128x128-t256")==0) spec={256,128,128,16,32,64,1,4,4,1,32};
    if(layout && strcmp(layout,"128x64-w1")==0) spec={128,128,64,16,64,32,1,4,2,1,32};
    if (!spec.empty())
        ggml_vk_create_pipeline(device,device->gemx_qkv_pipeline,"gemx_qkv_microtile",
            matmul_f32_f32_fp32_len,matmul_f32_f32_fp32_data,"main",3,sizeof(vk_mat_mat_push_constants),
            {spec[1],spec[2],1},ggml_vk_mul_mm_spec(spec,true),32);
}
