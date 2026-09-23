# Adapt a build-directory copy; keep the pinned upstream submodule pristine.
# Fail closed if the insertion point changes on a GGML update.
#
# 2026-09-22 performance extensions:
# - Broader device support for experimental pipelines
# - Optional adaptive fence wait insertion (GEMX_FENCE_POLL_SLEEP)

set(_vitpose_vk_source "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-vulkan/ggml-vulkan.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_vitpose_vk_source}")
file(READ "${_vitpose_vk_source}" _vitpose_vk_text)

function(_vitpose_vk_replace anchor replacement)
  string(FIND "${_vitpose_vk_text}" "${anchor}" position)
  string(FIND "${_vitpose_vk_text}" "${anchor}" last_position REVERSE)
  if(position EQUAL -1 OR NOT position EQUAL last_position)
    message(FATAL_ERROR "ViTPose Vulkan insertion point changed; review the GGML adapter or set GEMX_VITPOSE_TILES=OFF: ${anchor}")
  endif()
  string(REPLACE "${anchor}" "${replacement}" updated "${_vitpose_vk_text}")
  set(_vitpose_vk_text "${updated}" PARENT_SCOPE)
endfunction()

# --- original tile / pipeline insertions ---
set(_vitpose_vk_anchor "    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, effective_src1_type);")
_vitpose_vk_replace("${_vitpose_vk_anchor}" "${_vitpose_vk_anchor}\n#include \"vulkan_vitpose_tiles.inc\"")

set(_vitpose_vk_anchor "    vk_matmul_pipeline pipeline_matmul_f32 {};")
_vitpose_vk_replace("${_vitpose_vk_anchor}" "${_vitpose_vk_anchor}\n    vk_pipeline gemx_vitpose_down_pipeline;")

_vitpose_vk_replace("    // mul mat vec\n" "#include \"vulkan_vitpose_down_pipeline.inc\"\n    // mul mat vec\n")

set(_vitpose_vk_anchor "    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);")
_vitpose_vk_replace("${_vitpose_vk_anchor}" "    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, gemx_vitpose_split_pipeline ? gemx_vitpose_split_pipeline : pipeline);")

# Compile the optional fused normalization shader without editing GGML sources.
set(_gemx_norm_header "${CMAKE_CURRENT_BINARY_DIR}/gemx-vulkan/vitpose_norm_spv.inc")
add_custom_command(OUTPUT "${_gemx_norm_header}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/gemx-vulkan"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -O --target-env=vulkan1.2 -mfmt=c
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vitpose_norm.comp" -o "${_gemx_norm_header}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vitpose_norm.comp" VERBATIM)
add_custom_target(gemx-vitpose-shaders DEPENDS "${_gemx_norm_header}")
add_dependencies(ggml-vulkan gemx-vitpose-shaders)
target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gemx-vulkan")

_vitpose_vk_replace("#include \"ggml-vulkan-shaders.hpp\"" "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t gemx_norm_spv[] =\n#include \"vitpose_norm_spv.inc\"\n;")
_vitpose_vk_replace("    vk_pipeline gemx_vitpose_down_pipeline;" "    vk_pipeline gemx_vitpose_down_pipeline;\n    vk_pipeline gemx_norm_pipeline, gemx_rect_pipeline, gemx_qkv_pipeline;")
_vitpose_vk_replace("#include \"vulkan_vitpose_down_pipeline.inc\"" "#include \"vulkan_vitpose_down_pipeline.inc\"\n#include \"vulkan_vitpose_experiments.inc\"")
_vitpose_vk_replace("static void ggml_vk_norm(" "#include \"vulkan_vitpose_norm.inc\"\nstatic void ggml_vk_norm(")
_vitpose_vk_replace("        ggml_vk_norm(ctx, compute_ctx, src0, node);" "        if (ctx->num_additional_fused_ops == 2) gemx_fused_norm(ctx, compute_ctx, cgraph, node_idx);\n        else ggml_vk_norm(ctx, compute_ctx, src0, node);")
_vitpose_vk_replace("            if (num_adds) {" "            if (gemx_can_fuse_norm(ctx, cgraph, i)) {\n                ctx->num_additional_fused_ops = 2;\n                fusion_string = \"NORM_MUL_ADD\";\n                op_srcs_fused_elementwise[0] = false;\n                op_srcs_fused_elementwise[1] = true;\n                op_srcs_fused_elementwise[2] = true;\n            } else if (num_adds) {")

# --- Adaptive fence wait (optional, 2026-09-22) ---
_vitpose_vk_replace(
  "void ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {"
  "#include \"vulkan_vitpose_fence.inc\"\nvoid ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {")

# Write the adapted source
set(_vitpose_vk_copy "${CMAKE_CURRENT_BINARY_DIR}/gemx-vulkan/ggml-vulkan.cpp")
file(CONFIGURE OUTPUT "${_vitpose_vk_copy}" CONTENT "${_vitpose_vk_text}" @ONLY)

get_target_property(_vitpose_vk_sources ggml-vulkan SOURCES)
set(_vitpose_vk_replaced FALSE)
foreach(_vitpose_vk_entry IN LISTS _vitpose_vk_sources)
  if(_vitpose_vk_entry STREQUAL "ggml-vulkan.cpp" OR _vitpose_vk_entry STREQUAL "${_vitpose_vk_source}")
    list(REMOVE_ITEM _vitpose_vk_sources "${_vitpose_vk_entry}")
    list(APPEND _vitpose_vk_sources "${_vitpose_vk_copy}")
    set(_vitpose_vk_replaced TRUE)
  endif()
endforeach()
if(NOT _vitpose_vk_replaced)
  message(FATAL_ERROR "Cannot locate GGML Vulkan compilation unit for ViTPose tile adapter")
endif()
set_property(TARGET ggml-vulkan PROPERTY SOURCES "${_vitpose_vk_sources}")
target_include_directories(ggml-vulkan PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-vulkan")
