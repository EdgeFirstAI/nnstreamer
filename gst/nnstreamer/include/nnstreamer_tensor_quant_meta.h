/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Tensor Quantization Metadata
 * Copyright (C) 2026 Au-Zone Technologies <www.au-zone.com>
 */
/**
 * @file   nnstreamer_tensor_quant_meta.h
 * @date   18 Feb 2026
 * @brief  GstMeta carrying per-output-tensor quantization parameters
 * @author Sébastien Taylor <sebastien@au-zone.com>
 *
 * This GstMeta type is attached to output buffers by tensor_filter to
 * propagate quantization parameters (scale, zero_point, scheme) downstream.
 * Consumers can read it to dequantize without hardcoding quant params.
 *
 * The NnsTensorQuantInfo struct and helper functions can be used without
 * GStreamer (only GLib + tensor_typedef.h). The GstMeta parts require
 * <gst/gst.h> and are guarded accordingly.
 */
#ifndef __NNS_TENSOR_QUANT_META_H__
#define __NNS_TENSOR_QUANT_META_H__

#include <glib.h>
#include <nnstreamer_plugin_api_util.h>

G_BEGIN_DECLS

/**
 * @brief Quantization scheme — tells the consumer HOW to dequantize.
 *
 * Affine (asymmetric): real_value = scale * (quantized_value - zero_point)
 * Symmetric:           real_value = scale * quantized_value  (zero_point always 0)
 */
typedef enum {
  NNS_QUANT_NONE = 0,
  NNS_QUANT_AFFINE_PER_TENSOR,      /**< real = scale * (q - zero_point) */
  NNS_QUANT_SYMMETRIC_PER_TENSOR,   /**< real = scale * q; zero_point = 0 */
  NNS_QUANT_AFFINE_PER_CHANNEL,     /**< per-axis asymmetric */
  NNS_QUANT_SYMMETRIC_PER_CHANNEL,  /**< per-axis symmetric */
} NnsTensorQuantScheme;

/**
 * @brief Per-tensor quantization info.
 * For per-tensor: num_params=1, axis=-1
 * For per-channel: num_params=N, axis=quantized dimension index
 */
typedef struct _NnsTensorQuantInfo {
  NnsTensorQuantScheme scheme;  /**< quantization scheme */
  tensor_type dtype;            /**< quantized data type (uint8, int8, ...) */
  guint num_params;             /**< 1 for per-tensor, N for per-channel */
  gint axis;                    /**< -1 for per-tensor, dim index for per-channel */
  gdouble *scales;              /**< array[num_params], g_malloc'd */
  gint64 *zero_points;          /**< array[num_params], g_malloc'd */
} NnsTensorQuantInfo;

/**
 * @brief Set per-tensor affine (asymmetric) quantization.
 */
void nns_tensor_quant_info_set_affine (NnsTensorQuantInfo *info,
    tensor_type dtype, gdouble scale, gint64 zero_point);

/**
 * @brief Set per-tensor symmetric quantization (zero_point forced to 0).
 */
void nns_tensor_quant_info_set_symmetric (NnsTensorQuantInfo *info,
    tensor_type dtype, gdouble scale);

/**
 * @brief Set per-channel quantization (deep copies the arrays).
 */
void nns_tensor_quant_info_set_per_channel (NnsTensorQuantInfo *info,
    NnsTensorQuantScheme scheme, tensor_type dtype,
    guint num_params, gint axis,
    const gdouble *scales, const gint64 *zero_points);

/**
 * @brief Clear (free arrays in) a NnsTensorQuantInfo.
 */
void nns_tensor_quant_info_clear (NnsTensorQuantInfo *info);

/*
 * GstMeta integration — requires GStreamer headers.
 * Only available when <gst/gst.h> has been included before this header.
 */
#ifdef __GST_H__

/**
 * @brief GstMeta carrying per-output-tensor quantization parameters.
 * Attached to every output buffer by tensor_filter when available.
 */
typedef struct {
  GstMeta meta;
  guint num_tensors;
  NnsTensorQuantInfo quant[NNS_TENSOR_SIZE_LIMIT];
} GstNnsTensorQuantMeta;

GType gst_nns_tensor_quant_meta_api_get_type (void);
const GstMetaInfo *gst_nns_tensor_quant_meta_get_info (void);

#define GST_NNS_TENSOR_QUANT_META_API_TYPE \
    (gst_nns_tensor_quant_meta_api_get_type ())
#define GST_NNS_TENSOR_QUANT_META_INFO \
    (gst_nns_tensor_quant_meta_get_info ())

/**
 * @brief Get quant meta from buffer, or NULL if not present.
 */
GstNnsTensorQuantMeta *gst_buffer_get_nns_tensor_quant_meta (GstBuffer *buffer);

/**
 * @brief Add quant meta to buffer and return it for population.
 */
GstNnsTensorQuantMeta *gst_buffer_add_nns_tensor_quant_meta (GstBuffer *buffer);

#endif /* __GST_H__ */

G_END_DECLS

#endif /* __NNS_TENSOR_QUANT_META_H__ */
