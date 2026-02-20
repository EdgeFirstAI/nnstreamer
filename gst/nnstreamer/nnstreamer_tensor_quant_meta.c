/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Tensor Quantization Metadata
 * Copyright (C) 2026 Au-Zone Technologies <www.au-zone.com>
 */
/**
 * @file   nnstreamer_tensor_quant_meta.c
 * @date   18 Feb 2026
 * @brief  GstMeta implementation for tensor quantization parameters
 * @author Sébastien Taylor <sebastien@au-zone.com>
 */

#include <string.h>
#include <gst/gst.h>
#include "include/nnstreamer_tensor_quant_meta.h"

/**
 * @brief GstMeta init callback — zero-initialize the meta.
 */
static gboolean
_quant_meta_init (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  GstNnsTensorQuantMeta *qmeta = (GstNnsTensorQuantMeta *) meta;
  (void) params;
  (void) buffer;

  qmeta->num_tensors = 0;
  memset (qmeta->quant, 0, sizeof (qmeta->quant));

  return TRUE;
}

/**
 * @brief GstMeta free callback — release heap-allocated arrays.
 */
static void
_quant_meta_free (GstMeta * meta, GstBuffer * buffer)
{
  GstNnsTensorQuantMeta *qmeta = (GstNnsTensorQuantMeta *) meta;
  guint i;
  (void) buffer;

  for (i = 0; i < qmeta->num_tensors; i++)
    nns_tensor_quant_info_clear (&qmeta->quant[i]);
}

/**
 * @brief GstMeta transform callback — deep copy on buffer copy.
 */
static gboolean
_quant_meta_transform (GstBuffer * dest, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  GstNnsTensorQuantMeta *src_qmeta = (GstNnsTensorQuantMeta *) meta;
  GstNnsTensorQuantMeta *dst_qmeta;
  guint i;
  (void) buffer;
  (void) data;

  if (!GST_META_TRANSFORM_IS_COPY (type))
    return FALSE;

  dst_qmeta = gst_buffer_add_nns_tensor_quant_meta (dest);
  if (!dst_qmeta)
    return FALSE;

  dst_qmeta->num_tensors = src_qmeta->num_tensors;
  for (i = 0; i < src_qmeta->num_tensors; i++) {
    NnsTensorQuantInfo *sq = &src_qmeta->quant[i];
    NnsTensorQuantInfo *dq = &dst_qmeta->quant[i];

    if (sq->scheme == NNS_QUANT_NONE || sq->num_params == 0) {
      memset (dq, 0, sizeof (*dq));
      continue;
    }

    nns_tensor_quant_info_set_per_channel (dq, sq->scheme, sq->dtype,
        sq->num_params, sq->axis, sq->scales, sq->zero_points);
  }

  return TRUE;
}

/**
 * @brief Register the GstMeta API type.
 *
 * This code may exist in both libnnstreamer.so (non-plugin) and the GStreamer
 * plugin libnnstreamer.so, each with its own static variable. To avoid a
 * double-registration GType collision, check g_type_from_name() first.
 */
GType
gst_nns_tensor_quant_meta_api_get_type (void)
{
  static GType type = 0;
  static const gchar *tags[] = { NULL };

  if (g_once_init_enter (&type)) {
    GType _type = g_type_from_name ("GstNnsTensorQuantMetaAPI");
    if (_type == 0)
      _type = gst_meta_api_type_register ("GstNnsTensorQuantMetaAPI", tags);
    g_once_init_leave (&type, _type);
  }

  return type;
}

/**
 * @brief Register the GstMeta info.
 */
const GstMetaInfo *
gst_nns_tensor_quant_meta_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;

  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *info = gst_meta_register (
        GST_NNS_TENSOR_QUANT_META_API_TYPE,
        "GstNnsTensorQuantMeta",
        sizeof (GstNnsTensorQuantMeta),
        _quant_meta_init,
        _quant_meta_free,
        _quant_meta_transform);
    g_once_init_leave (&meta_info, info);
  }

  return meta_info;
}

/**
 * @brief Get quant meta from buffer, or NULL if not present.
 */
GstNnsTensorQuantMeta *
gst_buffer_get_nns_tensor_quant_meta (GstBuffer * buffer)
{
  return (GstNnsTensorQuantMeta *) gst_buffer_get_meta (buffer,
      GST_NNS_TENSOR_QUANT_META_API_TYPE);
}

/**
 * @brief Add quant meta to buffer and return it for population.
 */
GstNnsTensorQuantMeta *
gst_buffer_add_nns_tensor_quant_meta (GstBuffer * buffer)
{
  return (GstNnsTensorQuantMeta *) gst_buffer_add_meta (buffer,
      GST_NNS_TENSOR_QUANT_META_INFO, NULL);
}
