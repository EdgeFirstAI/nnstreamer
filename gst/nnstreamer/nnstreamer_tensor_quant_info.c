/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Tensor Quantization Info Helpers
 * Copyright (C) 2026 Au-Zone Technologies <www.au-zone.com>
 */
/**
 * @file   nnstreamer_tensor_quant_info.c
 * @date   18 Feb 2026
 * @brief  NnsTensorQuantInfo helper functions (no GStreamer dependency)
 * @author Sébastien Taylor <sebastien@au-zone.com>
 */

#include <string.h>
#include "include/nnstreamer_tensor_quant_meta.h"
#include "include/nnstreamer_util.h"

/**
 * @brief Set per-tensor affine (asymmetric) quantization.
 */
void
nns_tensor_quant_info_set_affine (NnsTensorQuantInfo * info,
    tensor_type dtype, gdouble scale, gint64 zero_point)
{
  nns_tensor_quant_info_clear (info);
  info->scheme = NNS_QUANT_AFFINE_PER_TENSOR;
  info->dtype = dtype;
  info->num_params = 1;
  info->axis = -1;
  info->scales = (gdouble *) g_malloc (sizeof (gdouble));
  info->zero_points = (gint64 *) g_malloc (sizeof (gint64));
  info->scales[0] = scale;
  info->zero_points[0] = zero_point;
}

/**
 * @brief Set per-tensor symmetric quantization (zero_point forced to 0).
 */
void
nns_tensor_quant_info_set_symmetric (NnsTensorQuantInfo * info,
    tensor_type dtype, gdouble scale)
{
  nns_tensor_quant_info_clear (info);
  info->scheme = NNS_QUANT_SYMMETRIC_PER_TENSOR;
  info->dtype = dtype;
  info->num_params = 1;
  info->axis = -1;
  info->scales = (gdouble *) g_malloc (sizeof (gdouble));
  info->zero_points = (gint64 *) g_malloc (sizeof (gint64));
  info->scales[0] = scale;
  info->zero_points[0] = 0;
}

/**
 * @brief Set per-channel quantization (deep copies the arrays).
 */
void
nns_tensor_quant_info_set_per_channel (NnsTensorQuantInfo * info,
    NnsTensorQuantScheme scheme, tensor_type dtype,
    guint num_params, gint axis,
    const gdouble * scales, const gint64 * zero_points)
{
  nns_tensor_quant_info_clear (info);
  info->scheme = scheme;
  info->dtype = dtype;
  info->num_params = num_params;
  info->axis = axis;
  info->scales = (gdouble *) _g_memdup (scales, sizeof (gdouble) * num_params);
  info->zero_points = (gint64 *) _g_memdup (zero_points,
      sizeof (gint64) * num_params);
}

/**
 * @brief Clear (free arrays in) a NnsTensorQuantInfo.
 */
void
nns_tensor_quant_info_clear (NnsTensorQuantInfo * info)
{
  g_free (info->scales);
  g_free (info->zero_points);
  memset (info, 0, sizeof (*info));
}
