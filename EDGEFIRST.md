# EdgeFirst NNStreamer Branch

This document describes the `edgefirst` branch, which is maintained by [EdgeFirst AI](https://github.com/EdgeFirstAI) for development and testing on NXP i.MX platforms.

## Upstream Source

The `edgefirst` branch is based on the NNStreamer production branch used by NXP in their Yocto BSP:

| Property | Value |
|----------|-------|
| **Upstream Repository** | https://github.com/nnstreamer/nnstreamer |
| **Base Branch** | `prod/tizen-9.0` |
| **Base Commit** | `9cf11e0768892636622480fe1b5fc042b5a224ad` |
| **Version** | 2.4.2 |

## Applied Patches

The following patches from the NXP Yocto BSP (`meta-imx/meta-imx-ml/recipes-nnstreamer/nnstreamer/nnstreamer/`) are applied in order:

### 1. RGB888 Video Meta Stride Support
- **File**: `0001-rgb888_support_nnstreamer.patch`
- **Author**: Elliot Chen (NXP)
- **Status**: i.MX specific
- **Description**: Uses video meta stride for padding removal instead of hardcoded 4-byte alignment calculation. Required for RGB888 support with `imxvideoconvert_g2d`.

### 2. GRAY8 Frame Padding Removal
- **File**: `0001-gray8_padding_removal.patch`
- **Author**: Alexandru Firuti (NXP)
- **Status**: i.MX specific
- **Description**: Adds frame padding removal for GRAY8 format with i.MX-specific 16-pixel and 4096-byte alignment requirements.

### 3. EthosU Delegate Memory Copy
- **File**: `0001-AIR-11938-tensor-filter-use-memcpy-ethosu-delegate.patch`
- **Author**: Nicolas Goueslain (NXP)
- **Status**: i.MX specific
- **Description**: Enables memcpy for EthosU NPU delegate in TFLite filter, similar to XNNPACK delegate handling. Required for proper buffer management when using NXP EthosU acceleration.

### 4. Default Delegates Option
- **File**: `0001-Fix-to-provide-default-delegates.patch`
- **Author**: Nolann Chobert (NXP)
- **Status**: Pending upstream
- **Description**: Adds `UseDefaultDelegates` custom option to the TFLite filter. Allows external delegates to use XNNPACK for operations they don't support.

### 5. NumPy Include Path Fix
- **File**: `0001-meson.build-Fix-include-path-for-numpy-YOCIMX-8735.patch`
- **Author**: Nicolas Goueslain (NXP)
- **Status**: Pending upstream
- **Description**: Fixes numpy include path from `numpy/core/include` to `numpy/_core/include` for compatibility with newer numpy versions.

### 6. Custom Filter Path Fix
- **File**: `0001-Fix-libnnstreamer_customfilter_passthrough.so-path.patch`
- **Author**: Nolann Chobert (NXP)
- **Status**: i.MX specific (Yocto build)
- **Description**: Fixes custom filter path for unittests to use installed location rather than build directory path.

## Build Instructions

### Prerequisites

You need a Yocto SDK for your target platform. The SDK provides the cross-compilation toolchain, sysroot, and required dependencies.

> **Note**: The SDK paths shown below (`/opt/yocto-sdk-*`) are specific to our local development setup. Your SDK installation path will vary depending on where you installed it.

### Building for i.MX 8M Plus

```bash
# Source the SDK environment
source /opt/yocto-sdk-imx8mp-frdm-6.12.49-2.2.0/environment-setup-armv8a-poky-linux

# Configure with meson
cd /path/to/nnstreamer
git checkout edgefirst
meson setup build --prefix=/usr \
    -Dtflite2-support=enabled \
    -Dwerror=false

# Build
ninja -C build
```

### Building for i.MX 95

```bash
# Source the SDK environment
source /opt/yocto-sdk-imx95-frdm-6.12.49-2.2.0/environment-setup-armv8a-poky-linux

# Configure with meson
cd /path/to/nnstreamer
git checkout edgefirst
meson setup build --prefix=/usr \
    -Dtflite2-support=enabled \
    -Dwerror=false

# Build
ninja -C build
```

### Common Build Options

| Option | Description |
|--------|-------------|
| `-Dtflite2-support=enabled` | Enable TensorFlow Lite 2.x inference support |
| `-Dpython3-support=enabled` | Enable Python 3 filter/converter/decoder support |
| `-Dprotobuf-support=enabled` | Enable Protocol Buffers support |
| `-Dnnstreamer-edge-support=enabled` | Enable NNStreamer Edge for distributed inference |
| `-Denable-test=true` | Build unit tests |
| `-Dwerror=false` | Disable warnings-as-errors (recommended for SDK builds) |

### Installing to Target

After building, you can install to a staging directory and copy to target:

```bash
# Install to staging directory
DESTDIR=/tmp/nnstreamer-install ninja -C build install

# Copy to target (example)
scp -r /tmp/nnstreamer-install/usr/* root@<target-ip>:/usr/
```

Or install directly via NFS/SSH mount.

## EdgeFirst Additions

This section documents EdgeFirst-specific features and modifications added on top of the NXP patches.

### V2 Invoke API with DMA-BUF Zero-Copy

The `edgefirst` branch extends the `tensor_filter` framework with a V2 invoke API that enables true DMA-BUF zero-copy between GStreamer elements and NPU hardware. This eliminates per-frame CPU memory copies on the inference hot path.

Key features:
- **DMA-BUF output allocation**: `tensor_filter` advertises DMA-BUF memory to downstream elements
- **Delegate-owned input DMA-BUF**: NPU delegates allocate input buffers that upstream elements (e.g. G2D) can write into directly
- **Allocation proxy**: `tensor_converter` forwards allocation queries so hardware video converters see the NPU's buffer pool
- **`propose_allocation` callback**: V2 sub-plugins can add custom buffer pools to GStreamer's allocation negotiation

### Ara-2 NPU Sub-Plugin

The `ara2` tensor_filter sub-plugin provides V2 invoke support for the Kinara Ara-2 NPU with full DMA-BUF zero-copy. All runtime symbols are loaded via `dlopen`/`dlsym` so the sub-plugin degrades gracefully when the Ara-2 runtime is unavailable.

Build with `-Dara2-support=enabled` (auto-detected from `dvapi.h` availability).

### Model Metadata Properties

The V2 framework exposes two read-only GObject properties on `tensor_filter` that allow applications to query model metadata at runtime. Both properties are populated by the framework-level ZIP reader (`edgefirst_metadata.cc`), which extracts `edgefirst.json` and `labels.txt` from the ZIP archive appended to model files (DVM, TFLite, etc.) at model open time. V2 sub-plugins may also implement `get_model_metadata` and `get_model_labels` callbacks as a fallback, but currently no sub-plugin does — both Ara-2 and TFLite set these to `NULL`.

#### Property: `model-labels`

Returns the model's class labels as an ordered string array (`GStrv` / `G_TYPE_STRV`). The label at index `i` corresponds to class index `i` in the model's output tensor.

**Data sources** (in priority order):
1. `labels.txt` from the model file's embedded ZIP archive (one label per line)
2. `NULL` if `labels.txt` is not present in the ZIP

#### Property: `model-metadata`

Returns the full model metadata as a JSON string conforming to the [EdgeFirst metadata schema](https://docs.edgefirst.ai/models/metadata/). Applications can parse this with any JSON library to access traceability information, output tensor specifications, quantization parameters, decoder configuration, and more.

**Data sources** (in priority order):
1. Verbatim `edgefirst.json` from the model file's embedded ZIP archive (returned as-is)
2. `NULL` if `edgefirst.json` is not present in the ZIP

#### Querying Properties from C/C++ Applications

Both properties are available after the pipeline reaches `PAUSED` state (i.e., the model has been loaded by the sub-plugin). Query them using standard GObject property access on the `tensor_filter` element:

```c
#include <gst/gst.h>

/* Assume 'filter' is a GstElement* for tensor_filter */

/* --- Class Labels --- */
GStrv labels = NULL;
g_object_get (filter, "model-labels", &labels, NULL);
if (labels) {
  for (int i = 0; labels[i]; i++)
    g_print ("  class %d: %s\n", i, labels[i]);
  g_strfreev (labels);  /* caller owns the copy */
}

/* --- Full Metadata JSON --- */
gchar *json = NULL;
g_object_get (filter, "model-metadata", &json, NULL);
if (json) {
  /* Parse with json-glib, cJSON, or any JSON library */
  g_print ("Metadata: %s\n", json);
  g_free (json);  /* caller owns the copy */
}
```

#### Querying Properties from GStreamer Pipeline Applications

For C++ applications using GStreamer pipelines (the typical NNStreamer use case), you can query metadata after setting the pipeline to `PAUSED`:

```cpp
#include <gst/gst.h>
#include <json-glib/json-glib.h>

/* After gst_element_set_state(pipeline, GST_STATE_PAUSED) and
 * gst_element_get_state() confirms PAUSED: */

GstElement *filter = gst_bin_get_by_name (GST_BIN (pipeline), "nn");

/* Get class labels */
GStrv labels = NULL;
g_object_get (filter, "model-labels", &labels, NULL);
if (labels) {
  int num_classes = g_strv_length (labels);
  g_print ("Model has %d classes\n", num_classes);

  /* Use labels for post-processing (e.g., annotating detections) */
  for (int i = 0; labels[i]; i++)
    g_print ("  [%d] %s\n", i, labels[i]);

  g_strfreev (labels);
}

/* Get full metadata for output quantization parameters */
gchar *metadata_json = NULL;
g_object_get (filter, "model-metadata", &metadata_json, NULL);
if (metadata_json) {
  JsonParser *parser = json_parser_new ();
  if (json_parser_load_from_data (parser, metadata_json, -1, NULL)) {
    JsonNode *root = json_parser_get_root (parser);
    JsonObject *obj = json_node_get_object (root);

    /* Access output tensor specifications */
    if (json_object_has_member (obj, "outputs")) {
      JsonArray *outputs = json_object_get_array_member (obj, "outputs");
      for (guint i = 0; i < json_array_get_length (outputs); i++) {
        JsonObject *out = json_array_get_object_element (outputs, i);
        const gchar *name = json_object_get_string_member (out, "name");
        const gchar *dtype = json_object_get_string_member (out, "dtype");

        g_print ("Output[%u]: name=%s dtype=%s\n", i,
            name ? name : "unnamed", dtype ? dtype : "unknown");

        /* Get quantization [scale, zero_point] */
        if (json_object_has_member (out, "quantization")) {
          JsonArray *q = json_object_get_array_member (out, "quantization");
          double scale = json_array_get_double_element (q, 0);
          gint64 zp = json_array_get_int_element (q, 1);
          g_print ("  quantization: scale=%f zero_point=%ld\n", scale, zp);
        }
      }
    }

    /* Access decoder version (Ultralytics models) */
    if (json_object_has_member (obj, "decoder_version"))
      g_print ("Decoder: %s\n",
          json_object_get_string_member (obj, "decoder_version"));

    /* Access NMS configuration */
    if (json_object_has_member (obj, "validation")) {
      JsonObject *val = json_object_get_object_member (obj, "validation");
      if (json_object_has_member (val, "iou"))
        g_print ("NMS IoU threshold: %f\n",
            json_object_get_double_member (val, "iou"));
      if (json_object_has_member (val, "score"))
        g_print ("NMS score threshold: %f\n",
            json_object_get_double_member (val, "score"));
    }
  }
  g_object_unref (parser);
  g_free (metadata_json);
}

gst_object_unref (filter);
```

#### Querying Properties from Python

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import json

Gst.init(None)

pipeline = Gst.parse_launch(
    'filesrc location=video.mp4 ! qtdemux ! h264parse ! v4l2h264dec ! '
    'imxvideoconvert_g2d ! video/x-raw,format=RGBA,width=640,height=640 ! '
    'videobox ! tensor_converter ! '
    'tensor_filter framework=ara2 model=yolov8n.dvm name=nn ! '
    'tensor_sink'
)

# Model must be loaded (PAUSED state) before querying
pipeline.set_state(Gst.State.PAUSED)
pipeline.get_state(Gst.CLOCK_TIME_NONE)

nn = pipeline.get_by_name('nn')

# Get class labels
labels = nn.get_property('model-labels')
if labels:
    print(f'Model has {len(labels)} classes:')
    for i, label in enumerate(labels):
        print(f'  [{i}] {label}')

# Get full metadata
metadata_str = nn.get_property('model-metadata')
if metadata_str:
    metadata = json.loads(metadata_str)

    # Output tensor specifications
    for out in metadata.get('outputs', []):
        print(f"Output '{out.get('name')}': "
              f"dtype={out.get('dtype')}, "
              f"shape={out.get('shape')}")
        q = out.get('quantization')
        if q:
            print(f'  scale={q[0]}, zero_point={q[1]}')

    # Traceability (if edgefirst.json was embedded)
    host = metadata.get('host', {})
    if host.get('studio_server'):
        print(f"Trained on: {host['studio_server']}")
        print(f"Session: {host.get('session')}")

    # Dataset info
    ds = metadata.get('dataset', {})
    if ds.get('name'):
        print(f"Dataset: {ds['name']}")

pipeline.set_state(Gst.State.NULL)
```

#### Example: Self-Configuring Post-Processor

The metadata properties enable applications to auto-configure their post-processing pipeline from the model file itself, without hardcoded assumptions:

```cpp
/* Auto-configure YOLOv8 post-processing from model metadata */
struct PostProcConfig {
  int num_classes;
  std::vector<std::string> class_names;
  std::vector<double> output_scales;
  std::vector<int64_t> output_zero_points;
  double nms_iou_threshold;
  double nms_score_threshold;
};

PostProcConfig
configure_from_model (GstElement *filter)
{
  PostProcConfig config = {};
  config.nms_iou_threshold = 0.45;   /* defaults */
  config.nms_score_threshold = 0.25;

  /* Labels → class names and count */
  GStrv labels = NULL;
  g_object_get (filter, "model-labels", &labels, NULL);
  if (labels) {
    for (int i = 0; labels[i]; i++)
      config.class_names.push_back (labels[i]);
    config.num_classes = config.class_names.size ();
    g_strfreev (labels);
  }

  /* Metadata → quantization params and thresholds */
  gchar *json_str = NULL;
  g_object_get (filter, "model-metadata", &json_str, NULL);
  if (json_str) {
    JsonParser *parser = json_parser_new ();
    if (json_parser_load_from_data (parser, json_str, -1, NULL)) {
      JsonObject *root = json_node_get_object (
          json_parser_get_root (parser));

      /* Per-output quantization */
      if (json_object_has_member (root, "outputs")) {
        JsonArray *outputs = json_object_get_array_member (root, "outputs");
        for (guint i = 0; i < json_array_get_length (outputs); i++) {
          JsonObject *out = json_array_get_object_element (outputs, i);
          if (json_object_has_member (out, "quantization")) {
            JsonArray *q = json_object_get_array_member (out, "quantization");
            config.output_scales.push_back (
                json_array_get_double_element (q, 0));
            config.output_zero_points.push_back (
                json_array_get_int_element (q, 1));
          }
        }
      }

      /* NMS thresholds from validation config */
      if (json_object_has_member (root, "validation")) {
        JsonObject *val = json_object_get_object_member (root, "validation");
        if (json_object_has_member (val, "iou"))
          config.nms_iou_threshold =
              json_object_get_double_member (val, "iou");
        if (json_object_has_member (val, "score"))
          config.nms_score_threshold =
              json_object_get_double_member (val, "score");
      }
    }
    g_object_unref (parser);
    g_free (json_str);
  }

  return config;
}
```

#### Metadata JSON Schema Reference

When `edgefirst.json` is embedded in the model file, the `model-metadata` property returns it verbatim. The full schema is documented in the [EdgeFirst Model Metadata Specification](https://docs.edgefirst.ai/models/metadata/).

When no `edgefirst.json` is present, a minimal JSON is built from the model's runtime parameters:

```json
{
  "outputs": [
    {
      "index": 0,
      "name": "output_scores",
      "dtype": "uint8",
      "shape": [80, 1, 8400],
      "quantization": [0.003906, 0]
    },
    {
      "index": 1,
      "name": "output_boxes",
      "dtype": "int16",
      "shape": [4, 1, 8400],
      "quantization": [0.019824, 0]
    }
  ]
}
```

Key schema sections (all optional, present only when `edgefirst.json` exists):

| Section | Description |
|---------|-------------|
| `host` | Traceability: EdgeFirst Studio server, project ID, training session |
| `dataset` | Dataset name, ID, and `classes` (ordered label list) |
| `name`, `description`, `author` | Model identification |
| `input` | Input shape, CameraAdaptor format, channel configuration |
| `model` | Architecture config (backbone, detection/segmentation flags) |
| `outputs[]` | Per-tensor: shape, dtype, type, decoder, quantization, stride |
| `decoder_version` | YOLO architecture: `yolov5`, `yolov8`, `yolo11`, `yolo26` |
| `nms` | NMS mode: `class_agnostic` or `class_aware` |
| `validation` | NMS thresholds (iou, score), preprocessing method |

---

## References

- [NNStreamer GitHub](https://github.com/nnstreamer/nnstreamer)
- [NNStreamer Documentation](https://nnstreamer.github.io/)
- [NXP i.MX Machine Learning User's Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)
