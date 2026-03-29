/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file   tensor_filter_tensorflow_lite.cc
 * @date   7 May 2018
 * @brief  Tensorflow-lite module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite, tensorflow2-lite)
 * for tensor_filter. The meson build system generates two .so files
 * (e.g., TF-Lite and TF2-Lite) from this source code.
 */

#include <algorithm>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_conf.h>
#include <nnstreamer_util.h>
#include <nnstreamer_tensor_quant_meta.h>

#ifdef HAVE_EDGEFIRST_HAL
#include <edgefirst/hal.h>
#endif

#if TFLITE_VERSION_MAJOR < 2
#pragma message("tensor_filter of TensorFlow Lite version 1.X is deprecated. Please use of TF Lite 2.X")
#endif

#if TFLITE_VERSION_MAJOR >= 2 || TFLITE_VERSION_MINOR >= 13
#if USE_TENSORFLOW2_HEADER_PATH
#include <tensorflow2/lite/kernels/register.h>
#include <tensorflow2/lite/model.h>
#else
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#endif
#else
#include <tensorflow/contrib/lite/kernels/register.h>
#include <tensorflow/contrib/lite/model.h>
#endif

/** control delegate headers */
#ifdef TFLITE_XNNPACK_DELEGATE_SUPPORTED
#if USE_TENSORFLOW2_HEADER_PATH
#include <tensorflow2/lite/delegates/xnnpack/xnnpack_delegate.h>
#else
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#endif
#endif

#ifdef TFLITE_GPU_DELEGATE_SUPPORTED
#if USE_TENSORFLOW2_HEADER_PATH
#include <tensorflow2/lite/delegates/gpu/delegate.h>
#else
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif
#endif

#ifdef TFLITE_NNAPI_DELEGATE_SUPPORTED
#if USE_TENSORFLOW2_HEADER_PATH
#include <tensorflow2/lite/delegates/nnapi/nnapi_delegate.h>
#else
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#endif
#endif

#ifdef TFLITE_EXTERNAL_DELEGATE_SUPPORTED
#if USE_TENSORFLOW2_HEADER_PATH
#include <tensorflow2/lite/delegates/external/external_delegate.h>
#else
#include <tensorflow/lite/delegates/external/external_delegate.h>
#endif
#endif

#ifdef TFLITE_QNN_DELEGATE_SUPPORTED
#include <QNN/TFLiteDelegate/QnnTFLiteDelegate.h>
#endif

#if !defined(TFLITE_SUBPLUGIN_NAME)
#warning "The sub-plugin name for tensorflow-lite is not defined."
#define TFLITE_SUBPLUGIN_NAME "tensorflow-lite"
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Possible tensorflow-lite delegates.
 */
typedef enum {
  TFLITE_DELEGATE_NONE = 0,
  TFLITE_DELEGATE_GPU,
  TFLITE_DELEGATE_NNAPI,
  TFLITE_DELEGATE_XNNPACK,
  TFLITE_DELEGATE_EXTERNAL,
  TFLITE_DELEGATE_QNN,

  TFLITE_DELEGATE_MAX
} tflite_delegate_e;

/**
 * @brief Internal enum for QNN backend type.
 */
typedef enum QNNBackendType {
  QNN_BACKEND_UNDEFINED = 0,
  QNN_BACKEND_GPU,
  QNN_BACKEND_HTP,
  QNN_BACKEND_DSP
} QNNBackendType;

/**
 * @brief Internal enum for QNN Performance mode.
 */
typedef enum QNNPerformanceMode {
  QNN_PERFMODE_Default = 0,
  QNN_PERFMODE_HighPerf,
  QNN_PERFMODE_PowerSaver
} QNNPerformanceMode;

/**
 * @brief Option to open tf-lite model.
 */
typedef struct {
  const gchar *model_file; /**< path to tensorflow-lite model file */
  const gchar *accelerators; /**< accelerators set for this subplugin */
  tflite_delegate_e delegate; /**< tensorflow-lite delegate */
  gint num_threads; /**< the number of threads */
  const gchar *ext_delegate_path; /**< path to external delegate lib */
  GHashTable *ext_delegate_kv_table; /**< external delegate key values options */
  QNNBackendType qnn_backend_type; /**< QNN Delegate backend type */
  QNNPerformanceMode qnn_performance_mode; /**< QNN Delegate performance mode */
  bool use_default_delegates; /**< whether to use default delegates in resolver */
  const gchar *camera_adaptor_format; /**< e.g., "rgbx", "rgba", "bgra"; NULL if disabled */
  bool dmabuf_enabled; /**< DmaBuf:true enables DMA-BUF zero-copy; false = sysmem (default) */
} tflite_option_s;

/**
 * @brief Possible accelerators.
 */
static const gchar *tflite_accl_support[] = { ACCL_CPU_NEON_STR,
  ACCL_CPU_SIMD_STR, ACCL_CPU_STR, ACCL_GPU_STR, ACCL_NPU_STR, NULL };

#if defined(__x86_64__) || defined(__aarch64__) || defined(__arm__)
static const gchar *tflite_accl_auto = ACCL_CPU_SIMD_STR;
#else
static const gchar *tflite_accl_auto = ACCL_CPU_STR;
#endif
static const gchar *tflite_accl_default = ACCL_CPU_STR;

static GstTensorFilterFrameworkStatistics tflite_internal_stats = {
  .total_invoke_num = 0,
  .total_invoke_latency = 0,
  .total_overhead_latency = 0,
};

/**
 * @brief Local definition of VxDmaBufDesc, matching vx_delegate_dmabuf.h.
 *
 * Since NNStreamer loads the delegate via dlsym (no compile-time dependency),
 * we define a compatible struct here for the RequestDmaBuf output parameter.
 */
typedef struct {
  int fd;           /**< DMA-BUF file descriptor */
  size_t size;      /**< Buffer size in bytes */
  void *map_ptr;    /**< Optional mmap'd pointer (NULL if not mapped) */
} VxDmaBufDesc;

/**
 * @brief VxDelegate DMA-BUF API function pointers loaded via dlsym.
 *
 * These are loaded at runtime from the external delegate library (libvx_delegate.so)
 * for backward compatibility — older delegate versions and platforms without NPU
 * hardware simply won't have these symbols, and inference falls back to CPU paths.
 */
typedef struct {
  gboolean loaded;       /**< TRUE if dlsym loading was attempted */
  gboolean available;    /**< TRUE if core DMA-BUF symbols were found */

  /* Core DMA-BUF registration */
  TfLiteBufferHandle (*RegisterDmaBuf) (TfLiteDelegate *, int, size_t, int);
  TfLiteStatus (*UnregisterDmaBuf) (TfLiteDelegate *, TfLiteBufferHandle);
  TfLiteStatus (*BindDmaBufToTensor) (TfLiteDelegate *, TfLiteBufferHandle, int);
  gboolean (*IsDmaBufSupported) (TfLiteDelegate *);

  /* Cache synchronization */
  TfLiteStatus (*SyncForDevice) (TfLiteDelegate *, TfLiteBufferHandle);
  TfLiteStatus (*SyncForCpu) (TfLiteDelegate *, TfLiteBufferHandle);

  /* Output DMA-BUF allocation (export mode) */
  TfLiteBufferHandle (*RequestDmaBuf) (TfLiteDelegate *, int, int, void *);
  TfLiteStatus (*ReleaseDmaBuf) (TfLiteDelegate *, TfLiteBufferHandle);
  int (*GetDmaBufFd) (TfLiteDelegate *, TfLiteBufferHandle);

  /* Granular CPU access bracketing */
  TfLiteStatus (*BeginCpuAccess) (TfLiteDelegate *, TfLiteBufferHandle, int);
  TfLiteStatus (*EndCpuAccess) (TfLiteDelegate *, TfLiteBufferHandle, int);

  /* Retrieve the inner DerivedDelegateData* (TfLiteExternalDelegate wraps it) */
  TfLiteDelegate *(*GetInstance) (void);
} VxDmaBufAPI;

static VxDmaBufAPI vx_dmabuf_api = {};

/**
 * @brief VxDelegate CameraAdaptor API function pointers loaded via dlsym.
 *
 * CameraAdaptor injects NPU-side Slice/Reverse ops for channel format
 * conversion (e.g., 4ch RGBx → 3ch RGB) so the CPU videoconvert element
 * can be eliminated from the pipeline.
 */
typedef struct {
  gboolean loaded;       /**< TRUE if dlsym loading was attempted */
  gboolean available;    /**< TRUE if core CameraAdaptor symbols were found */

  TfLiteStatus (*SetFormat) (TfLiteDelegate *, int, const char *);
  TfLiteStatus (*SetFormats) (TfLiteDelegate *, int, const char *, const char *);
  gboolean (*IsSupported) (const char *);
  int (*GetInputChannels) (const char *);
  int (*GetOutputChannels) (const char *);
} VxCameraAdaptorAPI;

static VxCameraAdaptorAPI vx_camera_api = {};

/**
 * @brief Load VxDelegate CameraAdaptor API symbols via dlsym.
 * @param lib_path Path to the external delegate shared library.
 */
static void
vx_camera_api_load (const char *lib_path)
{
  void *handle;

  if (vx_camera_api.loaded || !lib_path)
    return;

  vx_camera_api.loaded = TRUE;

  handle = dlopen (lib_path, RTLD_LAZY | RTLD_NOLOAD);
  if (!handle)
    handle = dlopen (lib_path, RTLD_LAZY);
  if (!handle) {
    nns_logi ("VxDelegate CameraAdaptor API not available: %s", dlerror ());
    return;
  }

#define LOAD_VX_CAM_SYM(field, sym) \
  *(void **) (&vx_camera_api.field) = dlsym (handle, sym)

  LOAD_VX_CAM_SYM (SetFormat, "VxCameraAdaptorSetFormat");
  LOAD_VX_CAM_SYM (SetFormats, "VxCameraAdaptorSetFormats");
  LOAD_VX_CAM_SYM (IsSupported, "VxCameraAdaptorIsSupported");
  LOAD_VX_CAM_SYM (GetInputChannels, "VxCameraAdaptorGetInputChannels");
  LOAD_VX_CAM_SYM (GetOutputChannels, "VxCameraAdaptorGetOutputChannels");

#undef LOAD_VX_CAM_SYM

  if (vx_camera_api.SetFormat && vx_camera_api.IsSupported
      && vx_camera_api.GetInputChannels) {
    vx_camera_api.available = TRUE;
    nns_logi ("VxDelegate CameraAdaptor API loaded from %s", lib_path);
  } else {
    nns_logi ("VxDelegate CameraAdaptor API not found in %s "
        "(symbols missing — older delegate version)", lib_path);
  }
}

/**
 * @brief Load VxDelegate DMA-BUF API symbols via dlsym.
 * @param lib_path Path to the external delegate shared library.
 *
 * Uses RTLD_NOLOAD first to check if already loaded (by TfLiteExternalDelegateCreate),
 * then falls back to RTLD_LAZY. This is safe to call multiple times — subsequent
 * calls are no-ops.
 */
static void
vx_dmabuf_api_load (const char *lib_path)
{
  void *handle;

  if (vx_dmabuf_api.loaded || !lib_path)
    return;

  vx_dmabuf_api.loaded = TRUE;

  /* The library should already be loaded by TfLiteExternalDelegateCreate */
  handle = dlopen (lib_path, RTLD_LAZY | RTLD_NOLOAD);
  if (!handle)
    handle = dlopen (lib_path, RTLD_LAZY);
  if (!handle) {
    nns_logi ("VxDelegate DMA-BUF API not available: %s", dlerror ());
    return;
  }

#define LOAD_VX_SYM(field, sym) \
  *(void **) (&vx_dmabuf_api.field) = dlsym (handle, sym)

  LOAD_VX_SYM (RegisterDmaBuf, "VxDelegateRegisterDmaBuf");
  LOAD_VX_SYM (UnregisterDmaBuf, "VxDelegateUnregisterDmaBuf");
  LOAD_VX_SYM (BindDmaBufToTensor, "VxDelegateBindDmaBufToTensor");
  LOAD_VX_SYM (IsDmaBufSupported, "VxDelegateIsDmaBufSupported");
  LOAD_VX_SYM (SyncForDevice, "VxDelegateSyncForDevice");
  LOAD_VX_SYM (SyncForCpu, "VxDelegateSyncForCpu");

  /* Output DMA-BUF allocation (optional) */
  LOAD_VX_SYM (RequestDmaBuf, "VxDelegateRequestDmaBuf");
  LOAD_VX_SYM (ReleaseDmaBuf, "VxDelegateReleaseDmaBuf");
  LOAD_VX_SYM (GetDmaBufFd, "VxDelegateGetDmaBufFd");
  LOAD_VX_SYM (BeginCpuAccess, "VxDelegateBeginCpuAccess");
  LOAD_VX_SYM (EndCpuAccess, "VxDelegateEndCpuAccess");
  LOAD_VX_SYM (GetInstance, "VxDelegateGetInstance");

#undef LOAD_VX_SYM

  /* Require the core symbols for DMA-BUF to be considered available */
  if (vx_dmabuf_api.RegisterDmaBuf && vx_dmabuf_api.UnregisterDmaBuf
      && vx_dmabuf_api.BindDmaBufToTensor && vx_dmabuf_api.IsDmaBufSupported) {
    vx_dmabuf_api.available = TRUE;
    nns_logi ("VxDelegate DMA-BUF API loaded from %s", lib_path);
  } else {
    nns_logi ("VxDelegate DMA-BUF API partially available from %s "
        "(some symbols missing)", lib_path);
  }

  /* Don't dlclose — the library is in use by the delegate */
}

/* ========== HAL Delegate DMA-BUF API (EDGEAI-1189) ================= */

#ifdef HAVE_EDGEFIRST_HAL
/**
 * @brief HAL Delegate DMA-BUF API function pointers loaded via dlsym.
 *
 * These are loaded at runtime from any external delegate library that
 * exports the hal_dmabuf_* symbols defined in edgefirst/hal.h.
 * Currently supported by the Neutron delegate (libneutron_delegate.so)
 * on i.MX 95 and will be supported by the VX delegate in the future.
 *
 * Unlike VxDmaBufAPI, the HAL delegate API uses a simpler model:
 * the delegate manages its own DMA-BUF allocations and exposes them
 * via hal_dmabuf_get_tensor_info(). No register/bind/request needed.
 */
typedef struct {
  gboolean loaded;       /**< TRUE if dlsym loading was attempted */
  gboolean available;    /**< TRUE if hal_dmabuf_is_supported returned 1 */

  int (*is_supported) (hal_delegate_t delegate);
  int (*get_tensor_info) (hal_delegate_t delegate, int tensor_index,
                          hal_dmabuf_tensor_info *info, size_t info_size);
  int (*sync_for_device) (hal_delegate_t delegate, int tensor_index);
  int (*sync_for_cpu) (hal_delegate_t delegate, int tensor_index);
  hal_delegate_t (*get_instance) (void);  /**< Retrieve delegate's internal handle */
} HalDmaBufAPI;

static HalDmaBufAPI hal_dmabuf_api = {};

/**
 * @brief Load HAL Delegate DMA-BUF API symbols via dlsym.
 * @param lib_path Path to the external delegate shared library.
 *
 * Probes for hal_dmabuf_* symbols in the delegate library. If found and
 * hal_dmabuf_is_supported() returns 1 for the given delegate, sets
 * hal_dmabuf_api.available = TRUE.
 *
 * @param delegate The TfLiteDelegate pointer (cast to hal_delegate_t for probing).
 */
static void
hal_dmabuf_api_load (const char *lib_path, void *delegate G_GNUC_UNUSED)
{
  void *handle;

  if (hal_dmabuf_api.loaded || !lib_path)
    return;

  hal_dmabuf_api.loaded = TRUE;

  handle = dlopen (lib_path, RTLD_LAZY | RTLD_NOLOAD);
  if (!handle)
    handle = dlopen (lib_path, RTLD_LAZY);
  if (!handle) {
    nns_logi ("HAL Delegate DMA-BUF API not available: %s", dlerror ());
    return;
  }

#define LOAD_HAL_SYM(field, sym) \
  *(void **) (&hal_dmabuf_api.field) = dlsym (handle, sym)

  LOAD_HAL_SYM (is_supported, "hal_dmabuf_is_supported");
  LOAD_HAL_SYM (get_tensor_info, "hal_dmabuf_get_tensor_info");
  LOAD_HAL_SYM (sync_for_device, "hal_dmabuf_sync_for_device");
  LOAD_HAL_SYM (sync_for_cpu, "hal_dmabuf_sync_for_cpu");
  LOAD_HAL_SYM (get_instance, "hal_dmabuf_get_instance");

#undef LOAD_HAL_SYM

  if (hal_dmabuf_api.is_supported && hal_dmabuf_api.get_tensor_info) {
    /* Symbols found — mark available. Runtime support check deferred to
     * setupHalDmaBuf() which has access to the correct delegate handle. */
    hal_dmabuf_api.available = TRUE;
    nns_logi ("HAL Delegate DMA-BUF API symbols found in %s", lib_path);
  } else {
    nns_logi ("HAL Delegate DMA-BUF API not found in %s "
        "(symbols missing — not a HAL-aware delegate)", lib_path);
  }

  /* Don't dlclose — the library is in use by the delegate */
}
#endif /* HAVE_EDGEFIRST_HAL */

/* ========== NnsDmaBufInputPool: single DMA-BUF buffer pool ========== */

typedef struct {
  GstBufferPool parent;
  int fd;           /**< delegate-allocated DMA-BUF fd */
  gsize buf_size;   /**< buffer size in bytes */
  gsize offset;     /**< byte offset within DMA-BUF for tensor data */
} NnsDmaBufInputPool;

typedef struct {
  GstBufferPoolClass parent_class;
} NnsDmaBufInputPoolClass;

static GType nns_dmabuf_input_pool_get_type (void);

static GstFlowReturn
nns_dmabuf_pool_alloc (GstBufferPool *pool, GstBuffer **buffer,
    GstBufferPoolAcquireParams *params)
{
  NnsDmaBufInputPool *self = (NnsDmaBufInputPool *) pool;
  UNUSED (params);

  GstAllocator *alloc = gst_dmabuf_allocator_new ();
  gsize alloc_size = self->offset + self->buf_size;
  GstMemory *mem = gst_dmabuf_allocator_alloc_with_flags (
      alloc, self->fd, alloc_size, GST_FD_MEMORY_FLAG_DONT_CLOSE);
  gst_object_unref (alloc);

  if (G_UNLIKELY (!mem))
    return GST_FLOW_ERROR;

  /* If there is an offset within the DMA-BUF, expose only the tensor data
   * region so downstream sees the correct size via gst_memory_get_sizes(). */
  if (self->offset > 0)
    gst_memory_resize (mem, self->offset, self->buf_size);

  *buffer = gst_buffer_new ();
  gst_buffer_append_memory (*buffer, mem);
  return GST_FLOW_OK;
}

static gboolean
nns_dmabuf_pool_set_config (GstBufferPool *pool, GstStructure *config)
{
  /* This pool wraps a single delegate-owned DMA-BUF fd. Multiple buffers
   * would all reference the same fd, causing data corruption. Force 1. */
  GstCaps *caps;
  guint size, min, max;
  gst_buffer_pool_config_get_params (config, &caps, &size, &min, &max);
  if (min != 1 || max != 1) {
    gst_buffer_pool_config_set_params (config, caps,
        ((NnsDmaBufInputPool *) pool)->buf_size, 1, 1);
  }
  return GST_BUFFER_POOL_CLASS (g_type_class_peek_parent (
      G_OBJECT_GET_CLASS (pool)))->set_config (pool, config);
}

static void
nns_dmabuf_pool_class_init (gpointer klass, gpointer class_data)
{
  UNUSED (class_data);
  GST_BUFFER_POOL_CLASS (klass)->alloc_buffer = nns_dmabuf_pool_alloc;
  GST_BUFFER_POOL_CLASS (klass)->set_config = nns_dmabuf_pool_set_config;
}

static void
nns_dmabuf_pool_init (GTypeInstance *instance, gpointer g_class)
{
  NnsDmaBufInputPool *self = (NnsDmaBufInputPool *) instance;
  UNUSED (g_class);
  self->fd = -1;
  self->buf_size = 0;
  self->offset = 0;
}

static GType
nns_dmabuf_input_pool_get_type (void)
{
  static gsize type_id = 0;
  if (g_once_init_enter (&type_id)) {
    GTypeInfo info = {
      sizeof (NnsDmaBufInputPoolClass), NULL, NULL,
      nns_dmabuf_pool_class_init, NULL, NULL,
      sizeof (NnsDmaBufInputPool), 0,
      nns_dmabuf_pool_init, NULL
    };
    GType t = g_type_register_static (
        GST_TYPE_BUFFER_POOL, "NnsDmaBufInputPool", &info, (GTypeFlags) 0);
    g_once_init_leave (&type_id, t);
  }
  return (GType) type_id;
}

/**
 * @brief Wrapper class for TFLite Interpreter to support model switching
 */
class TFLiteInterpreter
{
  public:
  TFLiteInterpreter ();
  ~TFLiteInterpreter ();

  int invoke (const GstTensorMemory *input, GstTensorMemory *output);
  int loadModel (int num_threads, tflite_delegate_e delegate);

  int setInputTensorProp ();
  int setOutputTensorProp ();
  int setInputTensorsInfo (const GstTensorsInfo *info);

  void setModelPath (const char *model_path);
  void setExtDelegate (const char *lib_path, GHashTable *key_val);
  void getExtDelegate (const char **lib_path, GHashTable **key_val);
  void setUseDefaultDelegates (gboolean use_default)
  {
    use_default_delegates = use_default;
  }
  void setCameraAdaptorFormat (const char *format)
  {
    g_free (camera_adaptor_format);
    camera_adaptor_format = format ? g_strdup (format) : nullptr;
  }
  /** @brief get current model path */
  const char *getModelPath ()
  {
    return model_path;
  }

  /** @brief return input tensor meta */
  const GstTensorsInfo *getInputTensorsInfo ()
  {
    return &inputTensorMeta;
  }
  /** @brief return output tensor meta */
  const GstTensorsInfo *getOutputTensorsInfo ()
  {
    return &outputTensorMeta;
  }

  /** @brief lock this interpreter */
  void lock ()
  {
    g_mutex_lock (&mutex);
  }
  /** @brief unlock this interpreter */
  void unlock ()
  {
    g_mutex_unlock (&mutex);
  }
  /** @brief cache input and output tensor ptr before invoke */
  int cacheInOutTensorPtr ();

  /** @brief set delegate for the tflite interpreter */
  void setDelegate (TfLiteDelegate *delegate, void (*deleter) (TfLiteDelegate *))
  {
    delegate_ptr = tflite::Interpreter::TfLiteDelegatePtr (delegate, deleter);
  }

  /** @brief get delegate for the tflite interpreter */
  TfLiteDelegate *getDelegate ()
  {
    return delegate_ptr.get ();
  }

  private:
  GMutex mutex;
  char *model_path;
  bool is_cached_after_first_invoke; /**< To cache again after first invoke */
  bool is_xnnpack_delegated; /**< To check if XNNPACK delegate is used */
  bool is_ethosu_delegated;
  char *ext_delegate_path; /**< path to external delegate lib */
  GHashTable *ext_delegate_kv_table; /**< external delegate key values options */
  QNNBackendType qnn_backend_type; /**< QNN Delegate backend type */
  QNNPerformanceMode qnn_performance_mode; /**< QNN Delegate performance mode */
  bool use_default_delegates; /**< whether to use default delegates in resolver */
  char *camera_adaptor_format; /**< Camera format (e.g., "rgbx"), NULL if disabled */

  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;

  GstTensorsInfo inputTensorMeta; /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta; /**< The tensor info of output tensors */
  std::vector<TfLiteTensor *> inputTensorPtr;
  std::vector<TfLiteTensor *> outputTensorPtr;

  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
  int setTensorProp (const std::vector<int> &tensor_idx_list, GstTensorsInfo *tensorMeta);

  tflite::Interpreter::TfLiteDelegatePtr delegate_ptr; /**< single delegate supported */
  friend class TFLiteCore;
};

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
  public:
  TFLiteCore (const GstTensorFilterProperties *prop);
  ~TFLiteCore ();
  int init (tflite_option_s *option);
  int loadModel ();
  gboolean compareModelPath (const char *model_path);
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo *info);
  int getOutputTensorDim (GstTensorsInfo *info);
  int setInputTensorDim (const GstTensorsInfo *info);
  int reloadModel (const char *model_path);
  int invoke (const GstTensorMemory *input, GstTensorMemory *output);
  /** @brief cache input and output tensor ptr before invoke */
  int cacheInOutTensorPtr ();

  /** @brief Get the TfLite delegate pointer (for VxDelegate DMA-BUF API calls) */
  TfLiteDelegate *getDelegate ()
  {
    return interpreter ? interpreter->getDelegate () : nullptr;
  }
  /** @brief Get the underlying tflite::Interpreter for direct Invoke() */
  tflite::Interpreter *getTfLiteInterpreter ()
  {
    return interpreter ? interpreter->interpreter.get () : nullptr;
  }
  /** @brief Get cached input tensor pointer by index */
  TfLiteTensor *getInputTensorPtr (unsigned int i)
  {
    return (interpreter && i < interpreter->inputTensorPtr.size ())
        ? interpreter->inputTensorPtr[i] : nullptr;
  }
  /** @brief Get cached output tensor pointer by index */
  TfLiteTensor *getOutputTensorPtr (unsigned int i)
  {
    return (interpreter && i < interpreter->outputTensorPtr.size ())
        ? interpreter->outputTensorPtr[i] : nullptr;
  }
  /** @brief Get the external delegate library path */
  const char *getExtDelegatePath ()
  {
    return interpreter ? interpreter->ext_delegate_path : nullptr;
  }

  /** @brief Release output DMA-BUF handles (call before interpreter destruction) */
  void releaseOutputDmaBuf ();
  /** @brief Request output DMA-BUFs from VxDelegate (call after model load) */
  gboolean setupOutputDmaBuf ();

  /** @brief Release input DMA-BUF handle and pool */
  void releaseInputDmaBuf ();
  /** @brief Request input DMA-BUF from VxDelegate and create buffer pool */
  gboolean setupInputDmaBuf ();

#ifdef HAVE_EDGEFIRST_HAL
  /** @brief Set up HAL delegate DMA-BUF input pool */
  gboolean setupHalDmaBuf ();
  /** @brief Release HAL delegate DMA-BUF resources */
  void releaseHalDmaBuf ();
#endif

  /** @brief Check if CameraAdaptor is active for this instance */
  gboolean hasCameraAdaptor () { return camera_adaptor.active; }

  /* CameraAdaptor state: NPU-side 4ch→3ch conversion */
  struct {
    gboolean active;       /**< TRUE if CameraAdaptor is configured */
    gchar *format;         /**< "rgbx", "rgba", etc. */
    int camera_channels;   /**< 4 for RGBx */
    int model_channels;    /**< 3 for RGB */
  } camera_adaptor;

  /* Phase 2: Output DMA-BUF state */
  struct {
    gboolean initialized;  /**< TRUE if output DMA-BUFs were requested */
    unsigned int num_outputs;
    TfLiteBufferHandle handles[NNS_TENSOR_SIZE_LIMIT];
    int fds[NNS_TENSOR_SIZE_LIMIT];
    gsize sizes[NNS_TENSOR_SIZE_LIMIT];
  } out_dmabuf;

  /* Phase 3: Input DMA-BUF state (delegate-owned, proposed upstream via pool) */
  struct {
    gboolean initialized;  /**< TRUE if input DMA-BUF was allocated */
    TfLiteBufferHandle handle;
    int fd;
    gsize size;            /**< actual buffer size (may differ from tensor->bytes with CameraAdaptor) */
    void *map_ptr;         /**< mmap'd pointer to DMA-BUF for CPU memcpy fallback */
    GstBufferPool *pool;   /**< pool wrapping the DMA-BUF fd, proposed upstream */
  } in_dmabuf;

#ifdef HAVE_EDGEFIRST_HAL
  /* HAL delegate DMA-BUF state (parallel to VxDelegate in_dmabuf) */
  struct {
    gboolean initialized;       /**< TRUE if HAL DMA-BUF input was set up */
    int input_fd;               /**< DMA-BUF fd for input tensor */
    size_t input_offset;        /**< byte offset within DMA-BUF */
    size_t input_size;          /**< tensor data size in bytes */
    int input_tensor_idx;       /**< TFLite tensor index */
    GstBufferPool *input_pool;  /**< pool wrapping the DMA-BUF fd */
    hal_delegate_t delegate_handle;  /**< delegate pointer for sync calls */
  } hal_dmabuf = {};
#endif

  bool dmabuf_enabled = false; /**< Whether DMA-BUF zero-copy is enabled for this instance */

  /** @brief Check whether DMA-BUF zero-copy is enabled */
  bool isDmaBufEnabled () const { return dmabuf_enabled; }

  /** @brief callback method to delete interpreter for shared model */
  friend void free_interpreter (void *instance);
  /** @brief callback method to replace interpreter for shared model */
  friend void replace_interpreter (void *instance, void *interpreter);

  private:
  int num_threads;
  accl_hw accelerator;
  tflite_delegate_e delegate;

  TFLiteInterpreter *interpreter;
  TFLiteInterpreter *interpreter_sub;

  gchar *shared_tensor_filter_key;
  gboolean checkSharedInterpreter (const GstTensorFilterProperties *prop);
  int reloadInterpreter (TFLiteInterpreter *new_interpreter);
  void setAccelerator (const char *accelerators, tflite_delegate_e d);
};

extern "C" {
void init_filter_tflite (void) __attribute__ ((constructor));
void fini_filter_tflite (void) __attribute__ ((destructor));
}

G_LOCK_DEFINE_STATIC (slock);

/**
 * @brief TFLiteInterpreter constructor
 */
TFLiteInterpreter::TFLiteInterpreter ()
    : delegate_ptr (nullptr, [] (TfLiteDelegate *) {})
{
  interpreter = nullptr;
  model = nullptr;
  model_path = nullptr;
  ext_delegate_path = nullptr;
  ext_delegate_kv_table = nullptr;
  camera_adaptor_format = nullptr;
  qnn_backend_type = QNN_BACKEND_UNDEFINED;
  qnn_performance_mode = QNN_PERFMODE_Default;
  use_default_delegates = FALSE;

  g_mutex_init (&mutex);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);

  is_cached_after_first_invoke = false;
  is_xnnpack_delegated = false;
  is_ethosu_delegated = false;
}

/**
 * @brief TFLiteInterpreter destructor
 */
TFLiteInterpreter::~TFLiteInterpreter ()
{
  g_mutex_clear (&mutex);
  g_free (model_path);
  g_free (ext_delegate_path);
  g_free (camera_adaptor_format);
  if (ext_delegate_kv_table)
    g_hash_table_unref (ext_delegate_kv_table);

  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief Internal implementation of TFLiteCore's invoke()
 */
int
TFLiteInterpreter::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  int64_t start_time, stop_time;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  start_time = g_get_monotonic_time ();

  /**
   * XNNPACK Delegate uses fixed buffer address for input/output tensors.
   * Therefore tensor data is to be manually copied from/to input/output
   * GStreamer buffers memory whose address changes at every round.
   */
  if (is_xnnpack_delegated || is_ethosu_delegated) {
    for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
      tensor_ptr = inputTensorPtr[i];
      g_assert (tensor_ptr->bytes == input[i].size);
      memcpy (tensor_ptr->data.raw, input[i].data, input[i].size);
    }
  } else {
    for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
      tensor_ptr = inputTensorPtr[i];
      tensor_ptr->data.raw = (char *) input[i].data;
    }

    for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
      tensor_ptr = outputTensorPtr[i];
      tensor_ptr->data.raw = (char *) output[i].data;
    }
  }

  stop_time = g_get_monotonic_time ();

  tflite_internal_stats.total_overhead_latency += stop_time - start_time;

  start_time = g_get_monotonic_time ();
  status = interpreter->Invoke ();

  /**
   * After the very first invoke, the output buffer address may change.
   * To handle the case, memcpy the output buffer directly.
   */
  if (is_xnnpack_delegated || is_ethosu_delegated || !is_cached_after_first_invoke) {
    for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
      tensor_ptr = outputTensorPtr[i];
      g_assert (tensor_ptr->bytes == output[i].size);
      memcpy (output[i].data, tensor_ptr->data.raw, output[i].size);
    }
  }

  stop_time = g_get_monotonic_time ();

  tflite_internal_stats.total_invoke_latency += stop_time - start_time;
  tflite_internal_stats.total_invoke_num += 1;

#if (DBG)
  ml_logi ("Invoke() is finished: %" G_GINT64_FORMAT "ms, model path: %s",
      (stop_time - start_time) / 1000, getModelPath ());
  ml_logi ("%" G_GINT64_FORMAT " invoke average %" G_GINT64_FORMAT
           ", total overhead %" G_GINT64_FORMAT,
      tflite_internal_stats.total_invoke_num,
      (tflite_internal_stats.total_invoke_latency / tflite_internal_stats.total_invoke_num),
      tflite_internal_stats.total_overhead_latency);
#endif

  if (status != kTfLiteOk) {
    ml_loge ("Failed to invoke");
    return -1;
  }

  if (!is_cached_after_first_invoke) {
    if (cacheInOutTensorPtr () == 0) {
      is_cached_after_first_invoke = true;
    } else {
      ml_logw ("Failed to cache tensor memory ptr");
    }
  }

  return 0;
}

/**
 * @brief Internal implementation of TFLiteCore's loadModel()
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::loadModel (int num_threads, tflite_delegate_e delegate_e)
{
  TfLiteDelegate *delegate;
#if (DBG)
  gint64 start_time, stop_time;
  start_time = g_get_monotonic_time ();
#endif

  model = tflite::FlatBufferModel::BuildFromFile (model_path);
  if (!model) {
    ml_loge ("Failed to mmap model\n");
    return -1;
  }

  /**
   * If got any trouble at model, active below code. It'll be help to analyze.
   * model->error_reporter ();
   */

  interpreter = nullptr;

  if (use_default_delegates) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
  } else {
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
  }

  if (!interpreter) {
    ml_loge ("Failed to construct interpreter\n");
    return -2;
  }

  if (num_threads > 0) {
    int n = static_cast<int> (std::thread::hardware_concurrency ());

    num_threads = MIN (n, num_threads);
    ml_logi ("Set the number of threads (%d)", num_threads);
    interpreter->SetNumThreads (num_threads);
  }

  /** set delegate after the accelerator prop */
  switch (delegate_e) {
    case TFLITE_DELEGATE_XNNPACK:
      {
#if TFLITE_XNNPACK_DELEGATE_SUPPORTED
        /* set xnnpack delegate */
        TfLiteXNNPackDelegateOptions xnnpack_options
            = TfLiteXNNPackDelegateOptionsDefault ();
        xnnpack_options.num_threads = (num_threads > 1) ? num_threads : 0;

        is_xnnpack_delegated = true;
        ml_logw ("Input/output tensors should be memcpy-ed rather than explicitly assigning its ptr when XNNPACK Delegate is used.");
        ml_logw ("This could cause performance degradation if sizes of input/output tensors are large");

        delegate = TfLiteXNNPackDelegateCreate (&xnnpack_options);
        void (*deleter) (TfLiteDelegate *) = [] (TfLiteDelegate *delegate_) {
          TfLiteXNNPackDelegateDelete (delegate_);
        };

        setDelegate (delegate, deleter);
#else
        ml_logw ("NNStreamer was built without XNNPACK delegate. Given delegate option XNNPACK is ignored.");
#endif
        break;
      }
    case TFLITE_DELEGATE_GPU:
      {
#if TFLITE_GPU_DELEGATE_SUPPORTED
        /* set gpu delegate when accelerator set to GPU */
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default ();
        options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
        options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

        /**
         * NNStreamer filter for TFLite2 GPU delegate only supports OpenCL backend
         * since GLES v3.1 backend has a constraint that
         * Invoke() must be called from the same EGLContext.
         */
        options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;

        delegate = TfLiteGpuDelegateV2Create (&options);
        void (*deleter) (TfLiteDelegate *) = [] (TfLiteDelegate *delegate_) {
          TfLiteGpuDelegateV2Delete (delegate_);
        };

        setDelegate (delegate, deleter);
#else
        ml_logw ("NNStreamer was built without GPU delegate. Given delegate option GPU is ignored.");
#endif
        break;
      }
    case TFLITE_DELEGATE_NNAPI:
      {
#if TFLITE_NNAPI_DELEGATE_SUPPORTED
        /* set nnapi delegate when accelerator set to auto (cpu.neon in Android) or NPU */
        delegate = new tflite::StatefulNnApiDelegate ();
        void (*deleter) (TfLiteDelegate *) = [] (TfLiteDelegate *delegate_) {
          delete reinterpret_cast<tflite::StatefulNnApiDelegate *> (delegate_);
        };

        setDelegate (delegate, deleter);
#else
        ml_logw ("NNStreamer was built without NNAPI delegate. Given delegate option NNAPI is ignored.");
#endif
        break;
      }
    case TFLITE_DELEGATE_EXTERNAL:
      {
#ifdef TFLITE_EXTERNAL_DELEGATE_SUPPORTED
        TfLiteExternalDelegateOptions options;

        options = TfLiteExternalDelegateOptionsDefault (ext_delegate_path);

        if (strcmp(ext_delegate_path, "libethosu_delegate.so") == 0){
          is_ethosu_delegated = true;
	}
        /* Add optional key values to delegate configuration */
        if (ext_delegate_kv_table) {
          GHashTable *table = ext_delegate_kv_table;
          GHashTableIter iter;
          gchar *key, *value;

          g_hash_table_iter_init (&iter, table);
          while (g_hash_table_iter_next (&iter, (gpointer *) &key, (gpointer *) &value))
            options.insert (&options, key, value);
        }

        delegate = TfLiteExternalDelegateCreate (&options);
        void (*deleter) (TfLiteDelegate *) = [] (TfLiteDelegate *delegate_) {
          TfLiteExternalDelegateDelete (delegate_);
        };

        setDelegate (delegate, deleter);
#else
        ml_logw ("NNStreamer was built without external delegate. Given delegate option external is ignored.");
#endif
        break;
      }
    case TFLITE_DELEGATE_QNN:
      {
#ifdef TFLITE_QNN_DELEGATE_SUPPORTED
        QnnDelegateApiVersion qnn_delegate_api_version = TfLiteQnnDelegateGetApiVersion ();
        nns_logi ("QNN Delegate API version: %u.%u.%u", qnn_delegate_api_version.major,
            qnn_delegate_api_version.minor, qnn_delegate_api_version.patch);

        TfLiteQnnDelegateOptions options = TfLiteQnnDelegateOptionsDefault ();

        /* Set backend type (GPU, HTP, DSP) */
        switch (qnn_backend_type) {
          case QNN_BACKEND_GPU:
            options.backend_type = kGpuBackend;
            break;
          case QNN_BACKEND_HTP:
            options.backend_type = kHtpBackend;
            break;
          case QNN_BACKEND_DSP:
            options.backend_type = kDspBackend;
            break;
          default:
            ml_loge ("Please set proper QNN Backend option");
            return -1;
        }

        /* Set performance mode (default, high performance, power saver) */
        switch (qnn_performance_mode) {
          case QNN_PERFMODE_HighPerf:
            options.dsp_options.performance_mode = kDspHighPerformance;
            options.htp_options.performance_mode = kHtpHighPerformance;
            options.gpu_options.performance_mode = kGpuHigh;
            break;
          case QNN_PERFMODE_PowerSaver:
            options.dsp_options.performance_mode = kDspPowerSaver;
            options.htp_options.performance_mode = kHtpPowerSaver;
            options.gpu_options.performance_mode = kGpuLow;
            break;
          case QNN_PERFMODE_Default:
            // do nothing
            break;
          default:
            break;
        }

        /* Set log level as Warn */
        options.log_level = kLogLevelWarn;

        delegate = TfLiteQnnDelegateCreate (&options);
        void (*deleter) (TfLiteDelegate *) = [] (TfLiteDelegate *delegate_) {
          TfLiteQnnDelegateDelete (delegate_);
        };

        setDelegate (delegate, deleter);
#else
        ml_logw ("NNStreamer was built without QNN delegate. Given delegate option QNN is ignored.");
#endif
        break;
      }
    default:
      break;
  }

  /* Configure CameraAdaptor — MUST happen BEFORE ModifyGraphWithDelegate
   * because Init() reads camera_adaptor_configs to inject Slice ops. */
  if (camera_adaptor_format && delegate_e == TFLITE_DELEGATE_EXTERNAL) {
    vx_camera_api_load (ext_delegate_path);
    vx_dmabuf_api_load (ext_delegate_path);

    if (vx_camera_api.available && vx_dmabuf_api.GetInstance) {
      TfLiteDelegate *inner_dlg = vx_dmabuf_api.GetInstance ();
      if (inner_dlg) {
        int tensor_idx = interpreter->inputs ()[0];
        TfLiteStatus cam_status = vx_camera_api.SetFormat (
            inner_dlg, tensor_idx, camera_adaptor_format);
        if (cam_status == kTfLiteOk)
          nns_logi ("CameraAdaptor configured: format=%s tensor=%d",
              camera_adaptor_format, tensor_idx);
        else
          ml_logw ("CameraAdaptor SetFormat failed for '%s'",
              camera_adaptor_format);
      }
    }

  }

  delegate = getDelegate ();
  if (delegate != nullptr) {
    if (interpreter->ModifyGraphWithDelegate (delegate) != kTfLiteOk) {
      ml_loge ("Failed to apply delegate\n");
      return -2;
    }
  }

  if (interpreter->AllocateTensors () != kTfLiteOk) {
    ml_loge ("Failed to allocate tensors\n");
    return -2;
  }

#ifdef HAVE_EDGEFIRST_HAL
  /* Probe for HAL delegate DMA-BUF API after delegate is fully initialized.
   * The delegate must have allocated its internal buffers (AllocateTensors)
   * before hal_dmabuf_is_supported() can detect DMA-BUF availability. */
  if (delegate_e == TFLITE_DELEGATE_EXTERNAL && ext_delegate_path) {
    hal_dmabuf_api_load (ext_delegate_path, (void *) delegate);
  }
#endif

#if (DBG)
  stop_time = g_get_monotonic_time ();
  ml_logi ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow Lite
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFLiteInterpreter::getTensorType (TfLiteType tfType)
{
  switch (tfType) {
    case kTfLiteFloat32:
      return _NNS_FLOAT32;
    case kTfLiteUInt8:
      return _NNS_UINT8;
    case kTfLiteInt32:
      return _NNS_INT32;
    case kTfLiteBool:
#ifdef TFLITE_INT8
    case kTfLiteInt8:
#endif
      return _NNS_INT8;
#ifdef TFLITE_INT16
    case kTfLiteInt16:
      return _NNS_INT16;
#endif
    case kTfLiteInt64:
      return _NNS_INT64;
#ifdef TFLITE_FLOAT16
    case kTfLiteFloat16:
#ifdef FLOAT16_SUPPORT
      return _NNS_FLOAT16;
#else
      ml_loge ("NNStreamer requires -DFLOAT16_SUPPORT as a build option to enable float16 type. This binary does not have float16 feature enabled; thus, float16 type is not supported in this instance.\n");
      break;
#endif
#endif
    case kTfLiteString:
#ifdef TFLITE_COMPLEX64
    case kTfLiteComplex64:
#endif
    default:
      ml_loge ("Not supported Tensorflow Data Type: [%d].", tfType);
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @return 0 if OK. non-zero if error.
 * @note assume that the interpreter lock was already held.
 */
int
TFLiteInterpreter::getTensorDim (int tensor_idx, tensor_dim dim)
{
  TfLiteIntArray *tensor_dims = interpreter->tensor (tensor_idx)->dims;
  int len = tensor_dims->size;

  /* 0-init */
  for (guint i = 0; i < NNS_TENSOR_RANK_LIMIT; ++i)
    dim[i] = 0;

  if (len > NNS_TENSOR_RANK_LIMIT)
    return -EPERM;

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (tensor_dims->data, tensor_dims->data + len, dim);

  return 0;
}

/**
 * @brief extract and store the information of given tensor list
 * @param tensor_idx_list list of index of tensors in tflite interpreter
 * @param[out] tensorMeta tensors to set the info into
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::setTensorProp (
    const std::vector<int> &tensor_idx_list, GstTensorsInfo *tensorMeta)
{
  tensorMeta->num_tensors = tensor_idx_list.size ();

  for (unsigned int i = 0; i < tensorMeta->num_tensors; ++i) {
    int idx = tensor_idx_list[i];
    GstTensorInfo *info = gst_tensors_info_get_nth_info (tensorMeta, i);

    if (getTensorDim (idx, info->dimension)) {
      ml_loge ("failed to get the dimension of input tensors");
      return -1;
    }
    info->type = getTensorType (interpreter->tensor (idx)->type);
    info->name = g_strdup (interpreter->tensor (idx)->name);

#if (DBG)
    gchar *dim_str = gst_tensor_get_dimension_string (info->dimension);
    ml_logi ("tensorMeta[%d] >> name[%s], type[%d], dim[%s]", i, info->name,
        info->type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::setInputTensorProp ()
{
  gst_tensors_info_free (&inputTensorMeta);
  return setTensorProp (interpreter->inputs (), &inputTensorMeta);
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::setOutputTensorProp ()
{
  gst_tensors_info_free (&outputTensorMeta);
  return setTensorProp (interpreter->outputs (), &outputTensorMeta);
}

/**
 * @brief set the Dimension for Input Tensor.
 * @param info Structure for input tensor info.
 * @return 0 if OK. non-zero if error.
 * @note rank can be changed dependent on the model
 */
int
TFLiteInterpreter::setInputTensorsInfo (const GstTensorsInfo *info)
{
  TfLiteStatus status = kTfLiteOk;
  const std::vector<int> &input_idx_list = interpreter->inputs ();
  int input_rank, cur_rank, dim;

  /** Cannot change the number of inputs */
  if (info->num_tensors != input_idx_list.size ())
    return -EINVAL;

  for (unsigned int tensor_idx = 0; tensor_idx < info->num_tensors; ++tensor_idx) {
    tensor_type tf_type;
    const GstTensorInfo *tensor_info;

    tensor_info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, tensor_idx);
    cur_rank = gst_tensor_info_get_rank (tensor_info);

    /** cannot change the type of input */
    tf_type = getTensorType (interpreter->tensor (input_idx_list[tensor_idx])->type);
    if (tf_type != tensor_info->type)
      return -EINVAL;

    /**
     * Given that the rank intended by the user cannot be exactly determined,
     * iterate over all possible ranks starting from MIN rank to the actual rank
     * of the dimension array. In case of none of these ranks work, return error
     */
    for (input_rank = cur_rank - 1; input_rank > 0; input_rank--) {
      if (tensor_info->dimension[input_rank] > 1)
        break;
    }

    for (int rank = input_rank + 1; rank <= NNS_TENSOR_RANK_LIMIT; rank++) {
      std::vector<int> dims (rank);
      /* the order of dimension is reversed at CAPS negotiation */
      for (int idx = 0; idx < rank; idx++) {
        /** check overflow when storing uint32_t in int container */
        if (tensor_info->dimension[rank - idx - 1] > INT_MAX)
          return -ERANGE;
        dim = tensor_info->dimension[rank - idx - 1];
        dims[idx] = (dim > 0) ? dim : 1;
      }
      status = interpreter->ResizeInputTensor (input_idx_list[tensor_idx], dims);
      if (status == kTfLiteOk) {
        status = interpreter->AllocateTensors ();
        if (status == kTfLiteOk)
          break;
      }
    }

    /** return error when none of the ranks worked */
    if (status != kTfLiteOk)
      return -EPERM;
  }

  return 0;
}

/**
 * @brief update the model path
 */
void
TFLiteInterpreter::setModelPath (const char *_model_path)
{
  if (_model_path) {
    g_free (model_path);
    model_path = g_strdup (_model_path);
  }
}

/**
 * @brief update external delegate library path and options
 */
void
TFLiteInterpreter::setExtDelegate (const char *lib_path, GHashTable *key_val)
{
  g_free (ext_delegate_path);
  if (lib_path)
    ext_delegate_path = g_strdup (lib_path);
  else
    ext_delegate_path = nullptr;

  if (ext_delegate_kv_table)
    g_hash_table_unref (ext_delegate_kv_table);

  if (key_val) {
    g_hash_table_ref (key_val);
    ext_delegate_kv_table = key_val;
  } else
    ext_delegate_kv_table = nullptr;
}

/**
 * @brief get external delegate library path and options
 */
void
TFLiteInterpreter::getExtDelegate (const char **lib_path, GHashTable **key_val)
{
  *lib_path = ext_delegate_path;
  *key_val = ext_delegate_kv_table;
}

/**
 * @brief cache input and output tensor ptr before invoke
 * @return 0 on success. -errno on failure.
 */
int
TFLiteInterpreter::cacheInOutTensorPtr ()
{
  int tensor_idx;
  TfLiteTensor *tensor_ptr;
  GstTensorInfo *info;

  inputTensorPtr.clear ();
  inputTensorPtr.reserve (inputTensorMeta.num_tensors);
  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    info = gst_tensors_info_get_nth_info (&inputTensorMeta, i);
    if (tensor_ptr->bytes != gst_tensor_info_get_size (info))
      goto fail_exit;

    inputTensorPtr.push_back (tensor_ptr);
  }

  outputTensorPtr.clear ();
  outputTensorPtr.reserve (outputTensorMeta.num_tensors);
  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    info = gst_tensors_info_get_nth_info (&outputTensorMeta, i);
    if (tensor_ptr->bytes != gst_tensor_info_get_size (info))
      goto fail_exit;

    outputTensorPtr.push_back (tensor_ptr);
  }

  return 0;

fail_exit:
  inputTensorPtr.clear ();
  outputTensorPtr.clear ();
  return -EINVAL;
}

/**
 * @brief	TFLiteCore constructor
 */
TFLiteCore::TFLiteCore (const GstTensorFilterProperties *prop)
{
  num_threads = -1;
  accelerator = ACCL_NONE;
  delegate = TFLITE_DELEGATE_NONE;
  interpreter_sub = nullptr;
  shared_tensor_filter_key = NULL;

  if (prop->shared_tensor_filter_key) {
    shared_tensor_filter_key = g_strdup (prop->shared_tensor_filter_key);
    if (!checkSharedInterpreter (prop))
      interpreter = new TFLiteInterpreter ();
  } else
    interpreter = new TFLiteInterpreter ();

  if (interpreter == NULL)
    nns_logd ("Failed to allocate memory for interpreter");

  memset (&out_dmabuf, 0, sizeof (out_dmabuf));
  memset (&in_dmabuf, 0, sizeof (in_dmabuf));
  memset (&camera_adaptor, 0, sizeof (camera_adaptor));
}

/**
 * @brief callback method to destroy interpreter for shared model
 * @param interpreter TFLiteInterpreter*
 */
void
free_interpreter (void *interpreter)
{
  TFLiteInterpreter *self = reinterpret_cast<TFLiteInterpreter *> (interpreter);
  delete self;
}

/**
 * @brief	TFLiteCore destructor
 */
TFLiteCore::~TFLiteCore ()
{
  g_free (camera_adaptor.format);
  camera_adaptor.format = NULL;

  if (shared_tensor_filter_key) {
    G_LOCK (slock);
    if (!nnstreamer_filter_shared_model_remove (this, shared_tensor_filter_key, free_interpreter)) {
      nns_loge ("failed to remove shared model");
    }
    G_UNLOCK (slock);
    g_free (shared_tensor_filter_key);
  } else {
    delete interpreter;
  }
}

/**
 * @brief	check the shared interpreter
 * The shared model representation (interpreter) is allocated or shared.
 * If `shared_tensor_filter_key` is already existed, it will share the TFLiteInterpreter.
 * in the opposite case, the new TFLiteInterpreter will allocated and registered at the shared table.
 */
gboolean
TFLiteCore::checkSharedInterpreter (const GstTensorFilterProperties *prop)
{
  G_LOCK (slock);
  interpreter = (TFLiteInterpreter *) nnstreamer_filter_shared_model_get (
      this, shared_tensor_filter_key);

  if (!interpreter) {
    /* create new interpreter */
    TFLiteInterpreter *new_interpreter = new TFLiteInterpreter ();
    interpreter = (TFLiteInterpreter *) nnstreamer_filter_shared_model_insert_and_get (
        this, shared_tensor_filter_key, new_interpreter);
    if (!interpreter) {
      G_UNLOCK (slock);
      ml_loge ("Failed to insert the model representation!");
      g_free (shared_tensor_filter_key);
      shared_tensor_filter_key = NULL;
      delete new_interpreter;
      return FALSE;
    }
  }
  /* shared model exists */
  else if (g_strcmp0 (prop->model_files[0], interpreter->getModelPath ()) != 0) {
    ml_logw ("The model paths are not equal, models are not shared.");
    nnstreamer_filter_shared_model_remove (this, shared_tensor_filter_key, free_interpreter);
    G_UNLOCK (slock);
    g_free (shared_tensor_filter_key);
    shared_tensor_filter_key = NULL;
    return FALSE;
  }
  G_UNLOCK (slock);

  ml_logd ("The model representation is shared: key=[%s]", shared_tensor_filter_key);
  return TRUE;
}

/**
 * @brief	Set the accelerator for the tf engine
 */
void
TFLiteCore::setAccelerator (const char *accelerators, tflite_delegate_e d)
{
  accelerator = parse_accl_hw (
      accelerators, tflite_accl_support, tflite_accl_auto, tflite_accl_default);

  delegate = d;

  /* set possible tensorflow-lite delegate from accelerator */
  if (delegate == TFLITE_DELEGATE_NONE) {
    /** @todo update condition to set delegate from accl hw */
    switch (accelerator) {
      case ACCL_GPU:
        delegate = TFLITE_DELEGATE_GPU;
        break;
      default:
        break;
    }
  }

  ml_logd ("Set tensorflow-lite delegate %d", delegate);
  return;
}

/**
 * @brief	initialize the object with tflite model
 * @param	option options to initialize tf-lite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 *        -4 if the caching of input and output tensors failed.
 */
int
TFLiteCore::init (tflite_option_s *option)
{
  dmabuf_enabled = option->dmabuf_enabled;
  interpreter->setModelPath (option->model_file);
  interpreter->setExtDelegate (option->ext_delegate_path, option->ext_delegate_kv_table);
  interpreter->setUseDefaultDelegates (option->use_default_delegates);
  interpreter->setCameraAdaptorFormat (option->camera_adaptor_format);
  interpreter->qnn_backend_type = option->qnn_backend_type;
  interpreter->qnn_performance_mode = option->qnn_performance_mode;
  num_threads = option->num_threads;
  int err;

  setAccelerator (option->accelerators, option->delegate);
  g_message ("accl = %s", get_accl_hw_str (accelerator));

  if ((err = loadModel ())) {
    ml_loge ("Failed to load model (TensorFlow-lite interpreter->loadModel() has returned %d. Please check if the model, '%s', is accessible and compatible with the given TensorFlow-lite instance. For example, this TensorFlow-lite's version might not support the given model.\n",
        err, option->model_file);
    return -1;
  }

  /* Set up CameraAdaptor state after loadModel configured the delegate */
  if (option->camera_adaptor_format && vx_camera_api.available) {
    int cam_ch = vx_camera_api.GetInputChannels
        ? vx_camera_api.GetInputChannels (option->camera_adaptor_format) : 0;
    int mdl_ch = vx_camera_api.GetOutputChannels
        ? vx_camera_api.GetOutputChannels (option->camera_adaptor_format) : 0;

    if (cam_ch > 0 && mdl_ch > 0) {
      camera_adaptor.active = TRUE;
      camera_adaptor.format = g_strdup (option->camera_adaptor_format);
      camera_adaptor.camera_channels = cam_ch;
      camera_adaptor.model_channels = mdl_ch;
      nns_logi ("CameraAdaptor active: %s (%dch -> %dch)",
          camera_adaptor.format, cam_ch, mdl_ch);
    } else {
      ml_logw ("CameraAdaptor format '%s' not usable (input_ch=%d output_ch=%d)",
          option->camera_adaptor_format, cam_ch, mdl_ch);
    }
  }

  if (setInputTensorProp ()) {
    ml_loge ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setOutputTensorProp ()) {
    ml_loge ("Failed to initialize output tensor\n");
    return -3;
  }
  if (cacheInOutTensorPtr ()) {
    ml_loge ("Failed to cache input and output tensors storage\n");
    return -4;
  }
  return 0;
}

/**
 * @brief	compare the model path
 * @return TRUE if tflite core has the same model path
 */
gboolean
TFLiteCore::compareModelPath (const char *model_path)
{
  gboolean is_same;

  interpreter->lock ();
  is_same = (g_strcmp0 (model_path, interpreter->getModelPath ()) == 0);
  interpreter->unlock ();

  return is_same;
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::loadModel ()
{
  int err;

  interpreter->lock ();
  err = interpreter->loadModel (num_threads, delegate);
  interpreter->unlock ();

  return err;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setInputTensorProp ()
{
  int err;

  interpreter->lock ();
  err = interpreter->setInputTensorProp ();
  interpreter->unlock ();

  return err;
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setOutputTensorProp ()
{
  int err;

  interpreter->lock ();
  err = interpreter->setOutputTensorProp ();
  interpreter->unlock ();

  return err;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (GstTensorsInfo *info)
{
  interpreter->lock ();
  gst_tensors_info_copy (info, interpreter->getInputTensorsInfo ());
  interpreter->unlock ();

  /* Override input tensor info for CameraAdaptor: report camera's 4ch
   * and uint8 type to GStreamer so caps negotiate with RGBA tensor from
   * tensor_converter. Internal TFLite tensors remain at model's native
   * 3ch int8. The NPU's Slice (4ch→3ch) and DataConvert (uint8→int8)
   * ops handle the conversion during inference. */
  if (camera_adaptor.active && info->num_tensors > 0) {
    GstTensorInfo *ti = gst_tensors_info_get_nth_info (info, 0);
    if (ti->dimension[0] == (guint) camera_adaptor.model_channels) {
      ti->dimension[0] = camera_adaptor.camera_channels;
    }
    /* CameraAdaptor DataConvert handles int8→uint8 on NPU, so report
     * uint8 to GStreamer for correct caps negotiation */
    if (ti->type == _NNS_INT8) {
      ti->type = _NNS_UINT8;
    }
  }

  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getOutputTensorDim (GstTensorsInfo *info)
{
  interpreter->lock ();
  gst_tensors_info_copy (info, interpreter->getOutputTensorsInfo ());
  interpreter->unlock ();

  return 0;
}

/**
 * @brief set the Dimension for Input Tensor.
 * @param info Structure for input tensor info.
 * @return 0 if OK. non-zero if error.
 * @note rank can be changed dependent on the model
 */
int
TFLiteCore::setInputTensorDim (const GstTensorsInfo *info)
{
  int err;

  interpreter->lock ();
  err = interpreter->setInputTensorsInfo (info);
  interpreter->unlock ();

  return err;
}

/**
 * @brief Replace the interpreter, called by reloadModel
 *        Check input/output tensors have the same info
 * @param new_interpreter new interpreter to replace with
 * @return int 0 if ok, non-zero if error
 */
int
TFLiteCore::reloadInterpreter (TFLiteInterpreter *new_interpreter)
{
  TFLiteInterpreter *old_interpreter = interpreter;
  gboolean in_matched, out_matched;
  int ret = 0;

  old_interpreter->lock ();
  new_interpreter->lock ();

  in_matched = gst_tensors_info_is_equal (old_interpreter->getInputTensorsInfo (),
      new_interpreter->getInputTensorsInfo ());
  out_matched = gst_tensors_info_is_equal (old_interpreter->getOutputTensorsInfo (),
      new_interpreter->getOutputTensorsInfo ());

  if (!in_matched || !out_matched) {
    ml_loge ("The model has unmatched tensors info\n");
    ret = -EINVAL;
  } else {
    interpreter = new_interpreter;
  }

  new_interpreter->unlock ();
  old_interpreter->unlock ();

  return ret;
}

/**
 * @brief callback method to replace interpreter for shared model
 */
void
replace_interpreter (void *instance, void *interpreter)
{
  TFLiteCore *core = reinterpret_cast<TFLiteCore *> (instance);
  TFLiteInterpreter *interpreter_new = reinterpret_cast<TFLiteInterpreter *> (interpreter);
  if (core->reloadInterpreter (interpreter_new) != 0)
    nns_loge ("Failed to replace interpreter");
}

/**
 * @brief	reload a model
 * @param	tflite	: the class object
 * @param[in] model_path : the path of model file
 * @return 0 if OK. non-zero if error.
 * @note reloadModel() is asynchronously called with other callbacks. But, it requires
 *       extra memory size enough to temporarily hold both models during this function.
 */
int
TFLiteCore::reloadModel (const char *_model_path)
{
  TFLiteInterpreter *interpreter_temp = interpreter;
  const char *_ext_delegate_path;
  GHashTable *_ext_delegate_kv;

  if (!g_file_test (_model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("The path of model file(s), %s, to reload is invalid.", _model_path);
    return -EINVAL;
  }
  interpreter_sub = new TFLiteInterpreter ();
  interpreter_sub->setModelPath (_model_path);
  interpreter->getExtDelegate (&_ext_delegate_path, &_ext_delegate_kv);
  interpreter_sub->setExtDelegate (_ext_delegate_path, _ext_delegate_kv);

  /**
   * load a model into sub interpreter. This loading overhead is independent
   * with main one's activities.
   */
  if (interpreter_sub->loadModel (num_threads, delegate) != 0) {
    ml_loge ("Failed to load model %s\n", _model_path);
    goto error;
  }
  if (interpreter_sub->setInputTensorProp () != 0) {
    ml_loge ("Failed to initialize input tensor\n");
    goto error;
  }
  if (interpreter_sub->setOutputTensorProp () != 0) {
    ml_loge ("Failed to initialize output tensor\n");
    goto error;
  }
  if (interpreter_sub->cacheInOutTensorPtr () != 0) {
    ml_loge ("Failed to cache input and output tensors storage\n");
    goto error;
  }

  if (shared_tensor_filter_key) {
    /* update cores with new interpreter that has shared key */
    nnstreamer_filter_shared_model_replace (this, shared_tensor_filter_key,
        interpreter_sub, replace_interpreter, free_interpreter);
  } else {
    if (reloadInterpreter (interpreter_sub) != 0) {
      ml_loge ("Failed replace interpreter\n");
      goto error;
    }
    delete interpreter_temp;
  }

  return 0;

error:
  delete interpreter_sub;
  return -EINVAL;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  int err;

  interpreter->lock ();
  err = interpreter->invoke (input, output);
  interpreter->unlock ();

  return err;
}

/**
 * @brief cache input and output tensor ptr before invoke
 */
int
TFLiteCore::cacheInOutTensorPtr ()
{
  int err;

  interpreter->lock ();
  err = interpreter->cacheInOutTensorPtr ();
  interpreter->unlock ();

  return err;
}

/**
 * @brief Request output DMA-BUFs from VxDelegate for zero-copy output.
 * @return TRUE if output DMA-BUFs were successfully set up.
 */
gboolean
TFLiteCore::setupOutputDmaBuf ()
{
  if (out_dmabuf.initialized || !vx_dmabuf_api.available
      || !vx_dmabuf_api.RequestDmaBuf || !vx_dmabuf_api.GetDmaBufFd
      || !vx_dmabuf_api.GetInstance)
    return out_dmabuf.initialized;

  /* Get the inner VxDelegate pointer (not the TfLiteExternalDelegate wrapper) */
  TfLiteDelegate *dlg = vx_dmabuf_api.GetInstance ();
  tflite::Interpreter *interp = getTfLiteInterpreter ();
  if (!dlg || !interp)
    return FALSE;

  unsigned int num_out = interp->outputs ().size ();
  for (unsigned int i = 0; i < num_out; i++) {
    int tensor_idx = interp->outputs ()[i];
    TfLiteTensor *tensor = interp->tensor (tensor_idx);
    if (!tensor || tensor->bytes == 0) {
      nns_logi ("Output tensor %d has no size, skipping DMA-BUF", tensor_idx);
      releaseOutputDmaBuf ();
      return FALSE;
    }

    /* Populate descriptor with required buffer size for allocation */
    VxDmaBufDesc desc = { .fd = -1, .size = tensor->bytes, .map_ptr = NULL };

    /* kVxDmaBufOwnerDelegate = 1 (delegate allocates and owns the buffer) */
    TfLiteBufferHandle h = vx_dmabuf_api.RequestDmaBuf (
        dlg, tensor_idx, 1 /* kVxDmaBufOwnerDelegate */, &desc);
    if (h == kTfLiteNullBufferHandle) {
      nns_logi ("Output DMA-BUF request failed for tensor %d (%zu bytes), "
          "using CPU path", tensor_idx, tensor->bytes);
      releaseOutputDmaBuf ();
      return FALSE;
    }

    /* Use fd from descriptor if populated, otherwise query via GetDmaBufFd */
    int fd = (desc.fd >= 0) ? desc.fd : vx_dmabuf_api.GetDmaBufFd (dlg, h);
    if (fd < 0) {
      nns_logi ("Output DMA-BUF fd retrieval failed for tensor %d", tensor_idx);
      vx_dmabuf_api.ReleaseDmaBuf (dlg, h);
      releaseOutputDmaBuf ();
      return FALSE;
    }

    /* Bind output DMA-BUF to tensor so NPU writes directly to it */
    vx_dmabuf_api.BindDmaBufToTensor (dlg, h, tensor_idx);

    out_dmabuf.handles[i] = h;
    out_dmabuf.fds[i] = fd;
    out_dmabuf.sizes[i] = tensor->bytes;
  }

  out_dmabuf.num_outputs = num_out;
  out_dmabuf.initialized = TRUE;
  nns_logi ("Output DMA-BUF enabled: %u output tensors bound", num_out);
  return TRUE;
}

/**
 * @brief Release output DMA-BUF handles.
 */
void
TFLiteCore::releaseOutputDmaBuf ()
{
  if (!out_dmabuf.initialized)
    return;

  TfLiteDelegate *dlg = vx_dmabuf_api.GetInstance
      ? vx_dmabuf_api.GetInstance () : getDelegate ();
  if (dlg && vx_dmabuf_api.ReleaseDmaBuf) {
    for (unsigned int i = 0; i < out_dmabuf.num_outputs; i++) {
      if (out_dmabuf.handles[i] != kTfLiteNullBufferHandle)
        vx_dmabuf_api.ReleaseDmaBuf (dlg, out_dmabuf.handles[i]);
    }
  }

  memset (&out_dmabuf, 0, sizeof (out_dmabuf));
}

/**
 * @brief Request input DMA-BUF from VxDelegate and create a buffer pool.
 *
 * Allocates a delegate-owned DMA-BUF for input tensor 0, binds it persistently,
 * and wraps it in a GstBufferPool that can be proposed upstream via the
 * allocation query. Upstream elements (e.g., imxvideoconvert_g2d) that honor
 * the pool will write directly into this buffer, achieving true zero-copy.
 *
 * With CameraAdaptor active, the allocated buffer size accounts for 4-channel
 * upstream data (e.g., RGBx 200704 bytes) even though the model tensor is
 * 3-channel (e.g., RGB 150528 bytes).
 */
gboolean
TFLiteCore::setupInputDmaBuf ()
{
  if (in_dmabuf.initialized || !vx_dmabuf_api.available
      || !vx_dmabuf_api.RequestDmaBuf || !vx_dmabuf_api.GetDmaBufFd
      || !vx_dmabuf_api.GetInstance)
    return FALSE;

  TfLiteDelegate *dlg = vx_dmabuf_api.GetInstance ();
  tflite::Interpreter *interp = getTfLiteInterpreter ();
  if (!dlg || !interp || interp->inputs ().empty ())
    return FALSE;

  int tensor_idx = interp->inputs ()[0];
  TfLiteTensor *tensor = interp->tensor (tensor_idx);
  if (!tensor || tensor->bytes == 0)
    return FALSE;

  /* Compute actual buffer size — with CameraAdaptor, upstream sends
   * 4ch (e.g., 200704 bytes) but model tensor is 3ch (150528 bytes) */
  gsize alloc_size = tensor->bytes;
  if (camera_adaptor.active && camera_adaptor.model_channels > 0) {
    alloc_size = (tensor->bytes / camera_adaptor.model_channels)
        * camera_adaptor.camera_channels;
  }

  VxDmaBufDesc desc = { .fd = -1, .size = alloc_size, .map_ptr = NULL };
  TfLiteBufferHandle h = vx_dmabuf_api.RequestDmaBuf (
      dlg, tensor_idx, 1 /* kVxDmaBufOwnerDelegate */, &desc);
  if (h == kTfLiteNullBufferHandle) {
    nns_logi ("Input DMA-BUF request failed for tensor %d (%zu bytes), "
        "using per-frame registration path", tensor_idx, alloc_size);
    return FALSE;
  }

  /* Use fd from descriptor if populated, otherwise query via GetDmaBufFd */
  int fd = (desc.fd >= 0) ? desc.fd : vx_dmabuf_api.GetDmaBufFd (dlg, h);
  if (fd < 0) {
    vx_dmabuf_api.ReleaseDmaBuf (dlg, h);
    return FALSE;
  }

  /* Bind input DMA-BUF to tensor — persists across Invoke() calls */
  vx_dmabuf_api.BindDmaBufToTensor (dlg, h, tensor_idx);

  /* Create buffer pool wrapping this fd */
  NnsDmaBufInputPool *pool = (NnsDmaBufInputPool *)
      g_object_new (nns_dmabuf_input_pool_get_type (), NULL);
  pool->fd = fd;
  pool->buf_size = alloc_size;

  GstStructure *config = gst_buffer_pool_get_config (GST_BUFFER_POOL (pool));
  GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new ();
  gst_buffer_pool_config_set_params (config, NULL, alloc_size, 1, 1);
  gst_buffer_pool_config_set_allocator (config, dmabuf_alloc, NULL);
  gst_buffer_pool_set_config (GST_BUFFER_POOL (pool), config);
  gst_object_unref (dmabuf_alloc);

  in_dmabuf.initialized = TRUE;
  in_dmabuf.handle = h;
  in_dmabuf.fd = fd;
  in_dmabuf.size = alloc_size;
  in_dmabuf.pool = GST_BUFFER_POOL (pool);

  /* mmap the DMA-BUF for CPU memcpy fallback (when upstream doesn't honor
   * our proposed pool, e.g., G2D provides its own buffers) */
  in_dmabuf.map_ptr = mmap (NULL, alloc_size, PROT_READ | PROT_WRITE,
      MAP_SHARED, fd, 0);
  if (in_dmabuf.map_ptr == MAP_FAILED) {
    nns_logw ("Input DMA-BUF mmap failed (fd=%d size=%zu): %s — "
        "CPU fallback for CameraAdaptor will not work", fd, alloc_size,
        g_strerror (errno));
    in_dmabuf.map_ptr = NULL;
  }

  nns_logi ("Input DMA-BUF enabled: tensor=%d fd=%d size=%zu map_ptr=%p",
      tensor_idx, fd, alloc_size, in_dmabuf.map_ptr);
  return TRUE;
}

/**
 * @brief Release input DMA-BUF handle and pool.
 */
void
TFLiteCore::releaseInputDmaBuf ()
{
  if (!in_dmabuf.initialized)
    return;

  if (in_dmabuf.map_ptr) {
    munmap (in_dmabuf.map_ptr, in_dmabuf.size);
  }

  if (in_dmabuf.pool) {
    gst_buffer_pool_set_active (in_dmabuf.pool, FALSE);
    gst_object_unref (in_dmabuf.pool);
  }

  TfLiteDelegate *dlg = vx_dmabuf_api.GetInstance
      ? vx_dmabuf_api.GetInstance () : getDelegate ();
  if (dlg && vx_dmabuf_api.ReleaseDmaBuf
      && in_dmabuf.handle != kTfLiteNullBufferHandle) {
    vx_dmabuf_api.ReleaseDmaBuf (dlg, in_dmabuf.handle);
  }

  memset (&in_dmabuf, 0, sizeof (in_dmabuf));
}

#ifdef HAVE_EDGEFIRST_HAL
/**
 * @brief Set up HAL delegate DMA-BUF input pool.
 *
 * Queries the HAL delegate for DMA-BUF tensor info for input tensor 0
 * and creates a GstBufferPool wrapping the fd.
 * This is the HAL delegate equivalent of setupInputDmaBuf() (VxDelegate).
 */
gboolean
TFLiteCore::setupHalDmaBuf ()
{
  if (hal_dmabuf.initialized || !hal_dmabuf_api.available)
    return FALSE;

  tflite::Interpreter *interp = getTfLiteInterpreter ();
  TfLiteDelegate *dlg = getDelegate ();

  if (!interp || !dlg || interp->inputs ().empty ())
    return FALSE;

  /* Get the delegate's internal handle via hal_dmabuf_get_instance().
   * TfLiteExternalDelegate wraps the real delegate; the HAL API needs
   * the inner pointer that was registered with dmabuf_set_delegate(). */
  hal_delegate_t hal_dlg = NULL;
  if (hal_dmabuf_api.get_instance)
    hal_dlg = hal_dmabuf_api.get_instance ();
  if (!hal_dlg) {
    nns_logd ("HAL delegate: get_instance returned NULL");
    return FALSE;
  }

  /* Check runtime support with the correct delegate handle */
  if (hal_dmabuf_api.is_supported && !hal_dmabuf_api.is_supported (hal_dlg)) {
    nns_logd ("HAL delegate: is_supported returned 0 for %p", hal_dlg);
    return FALSE;
  }

  int tensor_idx = interp->inputs ()[0];
  hal_dmabuf_tensor_info info = {};
  info.fd = -1;

  int rc = hal_dmabuf_api.get_tensor_info (
      hal_dlg, tensor_idx, &info, sizeof (info));
  nns_logd ("HAL delegate: get_tensor_info(%p, %d) rc=%d fd=%d offset=%zu size=%zu",
      hal_dlg, tensor_idx, rc, info.fd, info.offset, info.size);
  if (rc != 0 || info.fd < 0) {
    nns_logi ("HAL DMA-BUF not available for input tensor %d (rc=%d fd=%d)",
        tensor_idx, rc, info.fd);
    return FALSE;
  }

  /* Create buffer pool wrapping this fd */
  NnsDmaBufInputPool *pool = (NnsDmaBufInputPool *)
      g_object_new (nns_dmabuf_input_pool_get_type (), NULL);
  pool->fd = info.fd;
  pool->buf_size = info.size;
  pool->offset = info.offset;

  GstStructure *config = gst_buffer_pool_get_config (GST_BUFFER_POOL (pool));
  GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new ();
  gst_buffer_pool_config_set_params (config, NULL, info.size, 1, 1);
  gst_buffer_pool_config_set_allocator (config, dmabuf_alloc, NULL);
  gst_buffer_pool_set_config (GST_BUFFER_POOL (pool), config);
  gst_object_unref (dmabuf_alloc);

  hal_dmabuf.initialized = TRUE;
  hal_dmabuf.input_fd = info.fd;
  hal_dmabuf.input_offset = info.offset;
  hal_dmabuf.input_size = info.size;
  hal_dmabuf.input_tensor_idx = tensor_idx;
  hal_dmabuf.input_pool = GST_BUFFER_POOL (pool);
  hal_dmabuf.delegate_handle = hal_dlg;  /* Inner delegate from get_instance() */

  nns_logi ("HAL DMA-BUF input enabled: tensor=%d fd=%d offset=%zu size=%zu",
      tensor_idx, info.fd, (size_t) info.offset, (size_t) info.size);
  return TRUE;
}

/**
 * @brief Release HAL delegate DMA-BUF resources.
 */
void
TFLiteCore::releaseHalDmaBuf ()
{
  if (!hal_dmabuf.initialized)
    return;

  if (hal_dmabuf.input_pool) {
    gst_buffer_pool_set_active (hal_dmabuf.input_pool, FALSE);
    gst_object_unref (hal_dmabuf.input_pool);
  }

  memset (&hal_dmabuf, 0, sizeof (hal_dmabuf));
}
#endif /* HAVE_EDGEFIRST_HAL */

/**
 * @brief Internal function to get the option for tf-lite model.
 */
static int
tflite_parseCustomOption (const GstTensorFilterProperties *prop, tflite_option_s *option)
{
  if (prop->num_models != 1 || prop->model_files[0] == NULL)
    return -1;

  option->model_file = prop->model_files[0];
  option->accelerators = prop->accl_str;
  option->delegate = TFLITE_DELEGATE_NONE;
  option->num_threads = -1;
  option->ext_delegate_path = nullptr;
  option->ext_delegate_kv_table = nullptr;
  option->qnn_backend_type = QNN_BACKEND_UNDEFINED;
  option->qnn_performance_mode = QNN_PERFMODE_Default;
  option->camera_adaptor_format = nullptr;
  option->dmabuf_enabled = true;  /* HAL 0.13.2 fixes GL + G2D offset import for Neutron fds */

  if (prop->custom_properties) {
    gchar **strv;
    guint i, len;

    strv = g_strsplit (prop->custom_properties, ",", -1);
    len = g_strv_length (strv);

    for (i = 0; i < len; ++i) {
      gchar **pair = g_strsplit (strv[i], ":", -1);

      if (g_strv_length (pair) > 1) {
        g_strstrip (pair[0]);
        g_strstrip (pair[1]);

        if (g_ascii_strcasecmp (pair[0], "NumThreads") == 0) {
          option->num_threads = (int) g_ascii_strtoll (pair[1], NULL, 10);
        } else if (g_ascii_strcasecmp (pair[0], "UseDefaultDelegates") == 0) {
          if (g_ascii_strcasecmp (pair[1], "true") == 0 || g_ascii_strcasecmp (pair[1], "1") == 0)
            option->use_default_delegates = TRUE;
          else if (g_ascii_strcasecmp (pair[1], "false") == 0 || g_ascii_strcasecmp (pair[1], "0") == 0)
            option->use_default_delegates = FALSE;
          else
            ml_logw ("Invalid value for UseDefaultDelegates (%s). Use 'true' or 'false'.", pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "Delegate") == 0) {
          if (g_ascii_strcasecmp (pair[1], "NNAPI") == 0)
            option->delegate = TFLITE_DELEGATE_NNAPI;
          else if (g_ascii_strcasecmp (pair[1], "GPU") == 0)
            option->delegate = TFLITE_DELEGATE_GPU;
          else if (g_ascii_strcasecmp (pair[1], "XNNPACK") == 0)
            option->delegate = TFLITE_DELEGATE_XNNPACK;
          else if (g_ascii_strcasecmp (pair[1], "External") == 0)
            option->delegate = TFLITE_DELEGATE_EXTERNAL;
          else if (g_ascii_strcasecmp (pair[1], "QNN") == 0)
            option->delegate = TFLITE_DELEGATE_QNN;
          else
            ml_logw ("Unknown option to set tensorflow-lite delegate (%s).", pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "ExtDelegateLib") == 0) {
          option->ext_delegate_path = g_strdup (pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "ExtDelegateKeyVal") == 0) {
          gchar **kvpairs;
          guint j, kvnum;
          GHashTable *table = option->ext_delegate_kv_table;

          kvpairs = g_strsplit (pair[1], ";", -1);
          kvnum = g_strv_length (kvpairs);

          for (j = 0; j < kvnum; j++) {
            gchar **kv = g_strsplit (kvpairs[j], "#", -1);

            if (g_strv_length (kv) > 1) {
              g_strstrip (kv[0]);
              g_strstrip (kv[1]);
              if (!table) {
                table = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);
                option->ext_delegate_kv_table = table;
              }
              g_hash_table_insert (table, g_strdup (kv[0]), g_strdup (kv[1]));
            }
            g_strfreev (kv);
          }
          g_strfreev (kvpairs);
        } else if (g_ascii_strcasecmp (pair[0], "QNNBackend") == 0) {
          /* parse QNN Delegate Backend custom options */
          if (g_ascii_strcasecmp (pair[1], "HTP") == 0)
            option->qnn_backend_type = QNN_BACKEND_HTP;
          else if (g_ascii_strcasecmp (pair[1], "DSP") == 0)
            option->qnn_backend_type = QNN_BACKEND_DSP;
          else if (g_ascii_strcasecmp (pair[1], "GPU") == 0)
            option->qnn_backend_type = QNN_BACKEND_GPU;
          else
            ml_logw ("Unknown QNN backend option (%s).", pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "QNNPerformanceMode") == 0) {
          /* parse QNN Delegate Performance mode custom options */
          if (g_ascii_strcasecmp (pair[1], "default") == 0)
            option->qnn_performance_mode = QNN_PERFMODE_Default;
          else if (g_ascii_strcasecmp (pair[1], "highperformance") == 0)
            option->qnn_performance_mode = QNN_PERFMODE_HighPerf;
          else if (g_ascii_strcasecmp (pair[1], "powersaver") == 0)
            option->qnn_performance_mode = QNN_PERFMODE_PowerSaver;
          else
            ml_logw ("Unknown option for QNN perf mode: %s. Please set one of { \"default\", \"highperformance\", \"powersaver\" }",
                pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "CameraAdaptor") == 0) {
          option->camera_adaptor_format = g_strdup (pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "DmaBuf") == 0) {
          if (g_ascii_strcasecmp (pair[1], "true") == 0 || g_ascii_strcasecmp (pair[1], "1") == 0)
            option->dmabuf_enabled = true;
          else if (g_ascii_strcasecmp (pair[1], "false") == 0 || g_ascii_strcasecmp (pair[1], "0") == 0)
            option->dmabuf_enabled = false;
          else
            ml_logw ("Invalid value for DmaBuf (%s). Use 'true' or 'false'.", pair[1]);
        } else {
          ml_logw ("Unknown option (%s).", strv[i]);
        }
      }

      g_strfreev (pair);
    }

    g_strfreev (strv);
  }

  if (option->delegate == TFLITE_DELEGATE_EXTERNAL && option->ext_delegate_path == NULL) {
    ml_logw ("No shared lib for external delegate.");
    option->delegate = TFLITE_DELEGATE_NONE;
  }

  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilterProperties *prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  UNUSED (prop);

  if (!core)
    return;

  core->releaseInputDmaBuf ();
  core->releaseOutputDmaBuf ();
#ifdef HAVE_EDGEFIRST_HAL
  core->releaseHalDmaBuf ();
#endif
  delete core;
  *private_data = NULL;
}

/**
 * @brief Load tensorflow lite modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tflite_loadModelFile (const GstTensorFilterProperties *prop, void **private_data)
{
  int ret = 0;
  TFLiteCore *core;
  tflite_option_s option = {};

  if (tflite_parseCustomOption (prop, &option) != 0) {
    g_printerr ("Failed to parse options to initialize tensorflow-lite model.");
    ret = -1;
    goto done;
  }

  core = static_cast<TFLiteCore *> (*private_data);

  if (core != NULL) {
    if (core->compareModelPath (option.model_file)) {
      ret = 1; /* skipped */
      goto done;
    }

    tflite_close (prop, private_data);
  }

  core = new TFLiteCore (prop);
  if (core->init (&option) != 0) {
    *private_data = NULL;
    delete core;

    g_printerr ("failed to initialize the object: Tensorflow-lite");
    ret = -2;
    goto done;
  }

  *private_data = core;

  /* Only set up DMA-BUF when enabled (default: true, override via DmaBuf:false) */
  nns_logd ("DmaBuf=%s delegate=%s",
      core->isDmaBufEnabled () ? "true" : "false",
      core->getExtDelegatePath () ? core->getExtDelegatePath () : "(none)");
  if (core->isDmaBufEnabled ()) {
    vx_dmabuf_api_load (core->getExtDelegatePath ());
    core->setupOutputDmaBuf ();
    core->setupInputDmaBuf ();

#ifdef HAVE_EDGEFIRST_HAL
    /* Try HAL delegate DMA-BUF if VxDelegate path didn't initialize */
    if (!core->in_dmabuf.initialized) {
      nns_logd ("VxDelegate not available, trying HAL delegate (available=%d)",
          hal_dmabuf_api.available);
      core->setupHalDmaBuf ();
    }
#endif
  }

done:
  g_free ((gpointer) option.ext_delegate_path);
  option.ext_delegate_path = nullptr;

  if (option.ext_delegate_kv_table)
    g_hash_table_unref (option.ext_delegate_kv_table);
  option.ext_delegate_kv_table = nullptr;

  g_free ((gpointer) option.camera_adaptor_format);
  option.camera_adaptor_format = nullptr;

  return ret;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static int
tflite_open (const GstTensorFilterProperties *prop, void **private_data)
{
#if TFLITE_VERSION_MAJOR < 2
  ml_logw ("tensor_filter of TensorFlow Lite version 1.X is deprecated. Please use of TF Lite 2.X");
#endif
  int status = tflite_loadModelFile (prop, private_data);

  return status;
}

/**
 * @brief V2 invoke callback — receives GstMemory* directly without mapping.
 *
 * For DMA-BUF inputs (from camera/ISP), registers the fd with VxDelegate
 * for zero-copy NPU inference. For system memory, falls back to mapping
 * and pointer assignment (like V0). Output is always allocated as system
 * memory and copied from TFLite's output tensors.
 *
 * @param[in] prop read-only property values
 * @param[in/out] private_data Sub-plugin's private data (TFLiteCore*)
 * @param[in] input Array of void* (opaque GstMemory*) for input tensors
 * @param[in/out] output Array of void* (opaque GstMemory*) for output tensors.
 *   With allocate_in_invoke=TRUE, these start as NULL and are allocated here.
 * @param[in] num_input Number of input tensors
 * @param[in] num_output Number of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_invoke_v2 (const GstTensorFilterProperties *prop, void **private_data,
    void **input, void **output, unsigned int num_input, unsigned int num_output)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);
  UNUSED (prop);

  tflite::Interpreter *tfl_interp = core->getTfLiteInterpreter ();
  g_return_val_if_fail (tfl_interp != NULL, -EINVAL);

  /* Get the inner VxDelegate pointer for DMA-BUF API calls */
  TfLiteDelegate *delegate = vx_dmabuf_api.GetInstance
      ? vx_dmabuf_api.GetInstance () : core->getDelegate ();

  GstMapInfo in_maps[NNS_TENSOR_SIZE_LIMIT];
  gboolean in_mapped[NNS_TENSOR_SIZE_LIMIT] = { FALSE, };
  TfLiteBufferHandle in_handles[NNS_TENSOR_SIZE_LIMIT];
  gboolean in_dmabuf[NNS_TENSOR_SIZE_LIMIT] = { FALSE, };
  TfLiteStatus status;
  int ret = -1;

  /* Initialize handles to null */
  for (unsigned int i = 0; i < num_input; i++)
    in_handles[i] = kTfLiteNullBufferHandle;

  /* Check if DMA-BUF zero-copy is available (VxDelegate or HAL delegate) */
  gboolean use_dmabuf = FALSE;
  if (delegate && vx_dmabuf_api.available && vx_dmabuf_api.IsDmaBufSupported) {
    use_dmabuf = vx_dmabuf_api.IsDmaBufSupported (delegate);
  }
#ifdef HAVE_EDGEFIRST_HAL
  if (!use_dmabuf && core->hal_dmabuf.initialized)
    use_dmabuf = TRUE;
#endif

  int64_t start_time = g_get_monotonic_time ();

  /* 1. Set up input tensors */
  for (unsigned int i = 0; i < num_input; i++) {
    GstMemory *mem = (GstMemory *) input[i];
    TfLiteTensor *tensor = core->getInputTensorPtr (i);

    if (G_UNLIKELY (!tensor)) {
      ml_loge ("tflite_invoke_v2: input tensor %u not cached", i);
      goto cleanup;
    }

    if (use_dmabuf && gst_is_dmabuf_memory (mem)) {
      /* Zero-copy DMA-BUF path: delegates publish their own pre-allocated
       * input DMA-BUF upstream via the allocation pool.  edgefirstcameraadaptor
       * acquires from that pool and writes the converted frame into it.
       * The only valid input fd is the delegate-owned one below. */
      int fd = gst_dmabuf_memory_get_fd (mem);

      if (core->in_dmabuf.initialized && fd == core->in_dmabuf.fd) {
        /* VxDelegate: buffer is the pre-allocated input DMA-BUF — already bound */
        in_dmabuf[i] = TRUE;
        in_handles[i] = kTfLiteNullBufferHandle;
        continue;
      }
#ifdef HAVE_EDGEFIRST_HAL
      if (core->hal_dmabuf.initialized && fd == core->hal_dmabuf.input_fd) {
        /* HAL/Neutron delegate: edgefirstcameraadaptor wrote into the
         * delegate-owned input DMA-BUF at the negotiated offset — already bound */
        in_dmabuf[i] = TRUE;
        in_handles[i] = kTfLiteNullBufferHandle;
        continue;
      }
#endif

      /* Unexpected DMA-BUF fd: VX and Neutron/HAL delegates only export their
       * own pre-allocated input buffer — they do not import external fds
       * (HAL API has no import path; per-frame VX rebinding requires a full
       * graph rebuild).  Upstream element did not write into the delegate-
       * proposed pool buffer. Fall through to CPU memcpy path. */
      static gboolean unexpected_fd_warned = FALSE;
      if (G_UNLIKELY (!unexpected_fd_warned)) {
        unexpected_fd_warned = TRUE;
        nns_logw ("Input tensor %u: unexpected DMA-BUF fd=%d "
            "(expected delegate fd=%d). Upstream element did not write into "
            "the proposed pool — zero-copy is broken.", i, fd,
            core->in_dmabuf.initialized ? core->in_dmabuf.fd
                                        : core->hal_dmabuf.input_fd);
      } else {
        nns_logd ("Input tensor %u: unexpected DMA-BUF fd=%d, memcpy path", i, fd);
      }
    }

    /* CPU fallback: map GstMemory and set tensor data pointer */
    if (!gst_memory_map (mem, &in_maps[i], GST_MAP_READ)) {
      ml_loge ("tflite_invoke_v2: failed to map input memory %u", i);
      goto cleanup;
    }
    in_mapped[i] = TRUE;

    if (i == 0 && core->in_dmabuf.initialized && core->in_dmabuf.map_ptr) {
      /* VxDelegate memcpy fallback: the delegate-owned input DMA-BUF is
       * persistently bound to the input tensor from setupInputDmaBuf(), but
       * upstream did not write into it via the proposed pool.  Copy the
       * upstream frame into the delegate's DMA-BUF so the NPU reads the
       * correct data.  Must not happen in a correctly configured pipeline. */
      gsize copy_size = MIN (in_maps[i].size, core->in_dmabuf.size);
      memcpy (core->in_dmabuf.map_ptr, in_maps[i].data, copy_size);

      /* Flush CPU caches so NPU reads the data we just wrote.
       * The delegate-owned DMA-BUF has an active NPU device attachment,
       * so DMA_BUF_IOCTL_SYNC (called internally by SyncForDevice)
       * performs actual cache maintenance. Without this, the NPU would
       * read stale data from cache on cached CMA heap buffers. */
      if (vx_dmabuf_api.SyncForDevice && delegate
          && core->in_dmabuf.handle != kTfLiteNullBufferHandle) {
        vx_dmabuf_api.SyncForDevice (delegate, core->in_dmabuf.handle);
      }

      static gboolean input_copy_warned = FALSE;
      if (G_UNLIKELY (!input_copy_warned)) {
        input_copy_warned = TRUE;
        nns_logw ("Memcpy into delegate input DMA-BUF fd=%d (%zu bytes): "
            "upstream element did not write into the proposed pool — "
            "zero-copy is broken.",
            core->in_dmabuf.fd, copy_size);
      } else {
        nns_logd ("Memcpy into delegate input DMA-BUF fd=%d (%zu bytes)",
            core->in_dmabuf.fd, copy_size);
      }

      /* The DMA-BUF is already bound to tensor 0 from setupInputDmaBuf(),
       * so no registration needed — just mark it and continue. */
      in_dmabuf[i] = TRUE;
      in_handles[i] = kTfLiteNullBufferHandle;
      continue;
    }

    if (core->hasCameraAdaptor () && i == 0) {
      /* CameraAdaptor active but no delegate-owned DMA-BUF available —
       * the CPU path cannot handle 4ch→3ch conversion. */
      ml_loge ("CameraAdaptor active but no input DMA-BUF for memcpy fallback. "
          "Inference will likely fail.");
    }

    tensor->data.raw = (char *) in_maps[i].data;
  }

  /* For non-DMA-BUF outputs, set tensor pointers to pre-allocated memory.
   * Note: with allocate_in_invoke=TRUE, we don't set output pointers —
   * TFLite uses its internal buffers, and we copy after invoke. */

  {
    int64_t stop_time = g_get_monotonic_time ();
    tflite_internal_stats.total_overhead_latency += stop_time - start_time;
  }

  /* First-invoke DMA-BUF status logging (one-time) */
  {
    static gboolean first_invoke_logged = FALSE;
    if (G_UNLIKELY (!first_invoke_logged)) {
      first_invoke_logged = TRUE;
      nns_logi ("tflite_invoke_v2 first frame: input_dmabuf=%s output_dmabuf=%s "
          "input_pool=%s delegate=%p", use_dmabuf ? "yes" : "no",
          core->out_dmabuf.initialized ? "yes" : "no",
          core->in_dmabuf.initialized ? "yes" : "no", delegate);
    }
  }

  /* 2. Invoke the model */
#ifdef HAVE_EDGEFIRST_HAL
  /* Only flush CPU caches if we did a CPU memcpy into the HAL DMA-BUF.
   * The zero-copy pool path uses hardware (G2D/OpenGL) via the HAL to write
   * the converted frame — no CPU involvement — so the implicit DMABuf fence
   * between G2D and the Neutron DMA engine provides the required
   * synchronization.  Calling DMA_BUF_SYNC_END|WRITE here would interfere
   * with that fence and cause the Neutron driver to time out. */
  if (core->hal_dmabuf.initialized && hal_dmabuf_api.sync_for_device
      && in_mapped[0]) {
    hal_dmabuf_api.sync_for_device (core->hal_dmabuf.delegate_handle,
        core->hal_dmabuf.input_tensor_idx);
  }
#endif

  start_time = g_get_monotonic_time ();
  status = tfl_interp->Invoke ();
  {
    int64_t stop_time = g_get_monotonic_time ();
    tflite_internal_stats.total_invoke_latency += stop_time - start_time;
    tflite_internal_stats.total_invoke_num += 1;
  }

  if (status != kTfLiteOk) {
    ml_loge ("tflite_invoke_v2: TFLite Invoke() failed");
    goto cleanup;
  }

  /* NOTE: Do NOT call hal_dmabuf_api.sync_for_cpu() here.
   *
   * All Neutron input and output tensors share a single DMA-BUF allocation.
   * sync_for_cpu() issues DMA_BUF_IOCTL_SYNC(START|READ) on the output
   * tensor fd, which marks the *entire* shared buffer as CPU-busy.
   *
   * In the zero-copy pool path (G2D/GL → Neutron DMA-BUF), sync_for_device()
   * is not called before the next Invoke() because there is no CPU write to
   * flush — the buffer ownership is never returned to the NPU.  On the second
   * frame the Neutron DMA engine finds the buffer still marked CPU-busy and
   * times out (error 383307 DRIVER/TIMEOUT).
   *
   * The Neutron delegate copies output tensor data to TFLite's internal CPU
   * buffers during Invoke(); tensor->data.raw is already CPU-accessible after
   * Invoke() returns.  No explicit DMA-BUF sync is needed to read outputs. */

  /* 3. Handle output tensors */
  if (core->out_dmabuf.initialized) {
    /* DMA-BUF output path: wrap delegate-owned fd as GstDmaBufMemory */
    static GstAllocator *dmabuf_allocator = NULL;
    if (G_UNLIKELY (!dmabuf_allocator))
      dmabuf_allocator = gst_dmabuf_allocator_new ();

    for (unsigned int i = 0; i < num_output; i++) {
      GstMemory *out_mem = gst_dmabuf_allocator_alloc_with_flags (
          dmabuf_allocator, core->out_dmabuf.fds[i],
          core->out_dmabuf.sizes[i], GST_FD_MEMORY_FLAG_DONT_CLOSE);
      if (G_UNLIKELY (!out_mem)) {
        ml_loge ("tflite_invoke_v2: failed to wrap output DMA-BUF fd=%d",
            core->out_dmabuf.fds[i]);
        goto cleanup_output;
      }
      output[i] = out_mem;
    }
  } else {
    /* CPU fallback: allocate system memory + memcpy (Phase 1 path) */
    for (unsigned int i = 0; i < num_output; i++) {
      TfLiteTensor *tensor = core->getOutputTensorPtr (i);
      if (G_UNLIKELY (!tensor)) {
        ml_loge ("tflite_invoke_v2: output tensor %u not cached", i);
        goto cleanup_output;
      }

      GstMemory *out_mem = gst_allocator_alloc (NULL, tensor->bytes, NULL);
      if (G_UNLIKELY (!out_mem)) {
        ml_loge ("tflite_invoke_v2: failed to allocate output %u (%zu bytes)",
            i, tensor->bytes);
        goto cleanup_output;
      }

      GstMapInfo out_map;
      if (!gst_memory_map (out_mem, &out_map, GST_MAP_WRITE)) {
        gst_memory_unref (out_mem);
        ml_loge ("tflite_invoke_v2: failed to map output memory %u", i);
        goto cleanup_output;
      }

      memcpy (out_map.data, tensor->data.raw, tensor->bytes);
      gst_memory_unmap (out_mem, &out_map);
      output[i] = out_mem;
    }
  }

  ret = 0;
  goto cleanup;

cleanup_output:
  /* Free any output memories allocated so far on error */
  for (unsigned int i = 0; i < num_output; i++) {
    if (output[i]) {
      gst_memory_unref ((GstMemory *) output[i]);
      output[i] = NULL;
    }
  }

cleanup:
  /* Unmap CPU-mapped inputs and unregister DMA-BUF handles */
  for (unsigned int i = 0; i < num_input; i++) {
    if (in_mapped[i])
      gst_memory_unmap ((GstMemory *) input[i], &in_maps[i]);
    if (in_dmabuf[i] && in_handles[i] != kTfLiteNullBufferHandle)
      vx_dmabuf_api.UnregisterDmaBuf (delegate, in_handles[i]);
  }

  return ret;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimensions and types of input tensors
 */
static int
tflite_getInputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);
  UNUSED (prop);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimensions and types of output tensors
 */
static int
tflite_getOutputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);
  UNUSED (prop);

  return core->getOutputTensorDim (info);
}

#define tryRecovery(failedAt, status, location, exp) \
  do {                                               \
    status = (exp);                                  \
    if (status != 0) {                               \
      failedAt = location;                           \
      goto recovery_fail;                            \
    }                                                \
  } while (0)

/**
 * @brief A fallback function to recover input tensor dimensions
 */
static void
tflite_setInputDim_recovery (
    TFLiteCore *core, GstTensorsInfo *cur_in_info, const char *reason, int mode)
{
  int failedAt, status;

  tryRecovery (failedAt, status, __LINE__, core->setInputTensorDim (cur_in_info));
  if (mode >= 1)
    tryRecovery (failedAt, status, __LINE__, core->setInputTensorProp ());
  if (mode >= 2)
    tryRecovery (failedAt, status, __LINE__, core->setOutputTensorProp ());
  tryRecovery (failedAt, status, __LINE__, core->cacheInOutTensorPtr ());

  return;

recovery_fail:
  ml_logf ("Tensorflow-lite's setInputDim failed (%s) and its recovery failed (at %d line with error %d), too. "
           "The behavior will be unstable.\n",
      reason, failedAt, status);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param in_info The dimensions and types of input tensors
 * @param[out] out_info The dimensions and types of output tensors
 * @detail Output Tensor info is recalculated based on the set Input Tensor Info
 */
static int
tflite_setInputDim (const GstTensorFilterProperties *prop, void **private_data,
    const GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  GstTensorsInfo cur_in_info;
  int status;
  UNUSED (prop);

  g_return_val_if_fail (core, -EINVAL);
  g_return_val_if_fail (in_info, -EINVAL);
  g_return_val_if_fail (out_info, -EINVAL);

  /** get current input tensor info for resetting */
  status = core->getInputTensorDim (&cur_in_info);
  if (status != 0)
    goto exit;

  /** set new input tensor info */
  status = core->setInputTensorDim (in_info);
  if (status != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info, "while setting input tensor info", 0);
    goto exit;
  }

  /** update input tensor info */
  if ((status = core->setInputTensorProp ()) != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info, "while updating input tensor info", 1);
    goto exit;
  }

  /** update output tensor info */
  if ((status = core->setOutputTensorProp ()) != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info, "while updating output tensor info", 2);
    goto exit;
  }

  /** update the input and output tensor cache */
  status = core->cacheInOutTensorPtr ();
  if (status != 0) {
    tflite_setInputDim_recovery (
        core, &cur_in_info, "while updating input and output tensor cache", 2);
    goto exit;
  }

  /** get output tensor info to be returned */
  status = core->getOutputTensorDim (out_info);
  if (status != 0) {
    tflite_setInputDim_recovery (
        core, &cur_in_info, "while retrieving update output tensor info", 2);
    goto exit;
  }

exit:
  gst_tensors_info_free (&cur_in_info);

  return status;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_reloadModel (const GstTensorFilterProperties *prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core, -EINVAL);

  if (prop->num_models != 1)
    return -1;

  return core->reloadModel (prop->model_files[0]);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] hw backend accelerator hardware
 * @return 0 if supported. -errno if not supported.
 */
static int
tflite_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (tflite_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

/**
 * @brief V2 propose_allocation callback for TFLite sub-plugin.
 *
 * Adds the delegate-owned input DMA-BUF pool to the allocation query so
 * upstream elements (e.g., imxvideoconvert_g2d) can write directly into
 * the pre-bound buffer.
 */
static int
tflite_propose_allocation_v2 (const GstTensorFilterProperties *prop,
    void **private_data, void *query_ptr)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  GstQuery *query = (GstQuery *) query_ptr;
  UNUSED (prop);

  if (!core || !query || !core->isDmaBufEnabled ())
    return 0;

  /* Add delegate-owned input DMA-BUF pool if available (VxDelegate) */
  if (core->in_dmabuf.initialized && core->in_dmabuf.pool) {
    gst_query_add_allocation_pool (
        query, core->in_dmabuf.pool, core->in_dmabuf.size, 1, 1);
    nns_logi ("Proposed input DMA-BUF pool: fd=%d size=%zu",
        core->in_dmabuf.fd, core->in_dmabuf.size);
  }

#ifdef HAVE_EDGEFIRST_HAL
  /* Add HAL delegate input DMA-BUF pool if available */
  if (core->hal_dmabuf.initialized && core->hal_dmabuf.input_pool) {
    gst_query_add_allocation_pool (
        query, core->hal_dmabuf.input_pool, core->hal_dmabuf.input_size, 1, 1);
    nns_logi ("Proposed HAL DMA-BUF input pool: fd=%d offset=%zu size=%zu",
        core->hal_dmabuf.input_fd, core->hal_dmabuf.input_offset,
        core->hal_dmabuf.input_size);
  }
#endif

  /* Add DMA-BUF allocator to allocation params */
  GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new ();
  if (dmabuf_alloc) {
    gst_query_add_allocation_param (query, dmabuf_alloc, NULL);
    gst_object_unref (dmabuf_alloc);
  }

  return 0;
}

/**
 * @brief Map TfLiteType to NNStreamer tensor_type (standalone helper).
 */
static tensor_type
tflite_map_type (TfLiteType t)
{
  switch (t) {
    case kTfLiteFloat32: return _NNS_FLOAT32;
    case kTfLiteUInt8:   return _NNS_UINT8;
    case kTfLiteInt32:   return _NNS_INT32;
    case kTfLiteInt64:   return _NNS_INT64;
#ifdef TFLITE_INT8
    case kTfLiteInt8:    return _NNS_INT8;
#endif
#ifdef TFLITE_INT16
    case kTfLiteInt16:   return _NNS_INT16;
#endif
#if defined(TFLITE_FLOAT16) && defined(FLOAT16_SUPPORT)
    case kTfLiteFloat16: return _NNS_FLOAT16;
#endif
    default:             return _NNS_END;
  }
}

/**
 * @brief V2 get_output_quantization callback.
 *
 * Reads quantization parameters from TfLiteTensor for each output.
 * Supports both per-tensor and per-channel (per-axis) quantization.
 */
static int
tflite_get_output_quantization (const GstTensorFilterProperties *prop,
    void **private_data, void *quant_ptr, unsigned int num_outputs)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  NnsTensorQuantInfo *quant = static_cast<NnsTensorQuantInfo *> (quant_ptr);
  UNUSED (prop);

  if (!core)
    return -1;

  tflite::Interpreter *tfl = core->getTfLiteInterpreter ();
  if (!tfl)
    return -1;

  const std::vector<int> &outputs = tfl->outputs ();
  for (unsigned int i = 0; i < num_outputs && i < outputs.size (); i++) {
    TfLiteTensor *t = tfl->tensor (outputs[i]);
    if (!t)
      continue;

    tensor_type dtype = tflite_map_type (t->type);

    /* Check for per-channel (per-axis) quantization */
    if (t->quantization.type == kTfLiteAffineQuantization
        && t->quantization.params) {
      TfLiteAffineQuantization *aq =
          static_cast<TfLiteAffineQuantization *> (t->quantization.params);

      if (aq->scale && aq->scale->size > 1) {
        /* Per-channel quantization */
        gdouble *scales = (gdouble *) g_new (gdouble, aq->scale->size);
        gint64 *zps = (gint64 *) g_new (gint64, aq->scale->size);
        gboolean is_symmetric = TRUE;

        for (int j = 0; j < aq->scale->size; j++) {
          scales[j] = aq->scale->data[j];
          zps[j] = (aq->zero_point && j < aq->zero_point->size)
              ? aq->zero_point->data[j] : 0;
          if (zps[j] != 0)
            is_symmetric = FALSE;
        }

        NnsTensorQuantScheme scheme = is_symmetric
            ? NNS_QUANT_SYMMETRIC_PER_CHANNEL : NNS_QUANT_AFFINE_PER_CHANNEL;
        nns_tensor_quant_info_set_per_channel (&quant[i], scheme, dtype,
            aq->scale->size, aq->quantized_dimension, scales, zps);
        g_free (scales);
        g_free (zps);
        continue;
      }
    }

    /* Per-tensor quantization (common case for output activations) */
    if (t->params.scale != 0.0f) {
      if (t->params.zero_point == 0)
        nns_tensor_quant_info_set_symmetric (&quant[i], dtype,
            (gdouble) t->params.scale);
      else
        nns_tensor_quant_info_set_affine (&quant[i], dtype,
            (gdouble) t->params.scale, (gint64) t->params.zero_point);
    }
  }

  return 0;
}

static gchar filter_subplugin_tensorflow_lite[] = TFLITE_SUBPLUGIN_NAME;

static GstTensorFilterFramework NNS_support_tensorflow_lite
    = { .version = GST_TENSOR_FILTER_FRAMEWORK_V2,
        .open = tflite_open,
        .close = tflite_close,
        { .v2 = {
              .name = filter_subplugin_tensorflow_lite,
              .allow_in_place = FALSE,
              .allocate_in_invoke = TRUE, /**< V2: sub-plugin allocates output GstMemory* */
              .run_without_model = FALSE,
              .verify_model_path = TRUE,
              .statistics = &tflite_internal_stats,
              .invoke_v2 = tflite_invoke_v2,
              .getInputDimension = tflite_getInputDim,
              .getOutputDimension = tflite_getOutputDim,
              .setInputDimension = tflite_setInputDim,
              .destroyNotify = nullptr,
              .reloadModel = tflite_reloadModel,
              .handleEvent = nullptr,
              .checkAvailability = tflite_checkAvailability,
              .allocateInInvoke = nullptr,
              .propose_allocation = tflite_propose_allocation_v2,
              .get_model_metadata = NULL,
              .get_model_labels = NULL,
              .get_output_quantization = tflite_get_output_quantization,
          } } };

/**
 * @brief Internal function to register the filter.
 */
static void
_nns_filter_register_tflite (void)
{
  nnstreamer_filter_probe (&NNS_support_tensorflow_lite);
  nnstreamer_filter_set_custom_property_desc (NNS_support_tensorflow_lite.v2.name,
      "NumThreads", "Number of threads. Set 0 for default behaviors.",
      "UseDefaultDelegates", "Whether to use default delegates in resolver. Set 'true' or 'false'. Default is 'false'.",
      "Delegate", "TF-Lite delegation options: {'NNAPI', 'GPU', 'XNNPACK', 'External', 'QNN'}."
      " Do not specify to disable delegation.",
      "ExtDelegateLib", "Path to external delegate shared library", "ExtDelegateKeyVal",
      "key/values pairs optional parameters for delegate."
      " Format ExtDelegateKeyVal=key1#value1;key2#value2...",
      NULL);
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_tflite (void)
{
  _nns_filter_register_tflite ();
}

/** @brief Destruct the subplugin */
void
fini_filter_tflite (void)
{
  nnstreamer_filter_exit (NNS_support_tensorflow_lite.v2.name);
}
