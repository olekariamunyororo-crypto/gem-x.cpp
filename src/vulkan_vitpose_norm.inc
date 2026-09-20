// F32 layer norm + channel scale + bias. Keep GGML's reduction order.
static bool gemx_can_fuse_norm(ggml_backend_vk_context *ctx, const ggml_cgraph *graph, int i) {
    if (!ctx->device->gemx_norm_pipeline ||
        !ggml_can_fuse(graph, i, {GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD})) return false;
    const auto *norm=graph->nodes[i], *mul=graph->nodes[i+1], *add=graph->nodes[i+2];
    if (mul->src[0]!=norm || add->src[0]!=mul) return false;
    const auto *x=norm->src[0], *w=mul->src[1], *b=add->src[1];
    if (x->ne[0]!=1280 || x->ne[1]!=197 || x->ne[2]!=2 || x->ne[3]!=1 ||
        ggml_nelements(w)!=1280 || ggml_nelements(b)!=1280) return false;
    for (const auto *t : {x,w,b,norm,mul,add}) {
        if (t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t) || get_misalign_bytes(ctx,t)) return false;
    }
    return true;
}
static void gemx_fused_norm(ggml_backend_vk_context *ctx, vk_context &subctx, ggml_cgraph *graph, int i) {
    auto *norm=graph->nodes[i], *mul=graph->nodes[i+1], *add=graph->nodes[i+2];
    struct push { uint32_t width; float eps; } pc{1280,0};
    memcpy(&pc.eps,norm->op_params,sizeof(float));
    auto &pipeline=ctx->device->gemx_norm_pipeline;
    ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
    ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{
        ggml_vk_tensor_subbuffer(ctx,norm->src[0]),ggml_vk_tensor_subbuffer(ctx,mul->src[1]),
        ggml_vk_tensor_subbuffer(ctx,add->src[1]),ggml_vk_tensor_subbuffer(ctx,add)},pc,{394,1,1});
}
