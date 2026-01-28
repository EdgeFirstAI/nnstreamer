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

*This section documents EdgeFirst-specific features and modifications added on top of the NXP patches.*

<!-- TODO: Document EdgeFirst additions here as they are developed -->

---

## References

- [NNStreamer GitHub](https://github.com/nnstreamer/nnstreamer)
- [NNStreamer Documentation](https://nnstreamer.github.io/)
- [NXP i.MX Machine Learning User's Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)
