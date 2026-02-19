/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * EdgeFirst Model Metadata Utilities
 * Copyright (C) 2026 Au-Zone Technologies <www.au-zone.com>
 */
/**
 * @file   edgefirst_metadata.h
 * @date   19 Feb 2026
 * @brief  Read edgefirst.json and labels.txt from ZIP archives in model files
 * @author Sébastien Taylor <sebastien@au-zone.com>
 *
 * Model files (DVM, TFLite, etc.) can embed an edgefirst.json metadata file
 * and/or a labels.txt class-label file as a ZIP archive appended to the model
 * binary. ZIP reads from the back (EOCD) while the model reads from the front,
 * so they coexist. These utilities extract those embedded files.
 *
 * Only the ZIP tail (EOCD), central directory, and target file entries are
 * read into memory — the model binary itself is never loaded.
 *
 * This header is designed for dual C/C++ use:
 *   - C code sees only the extern "C" wrappers (GLib-allocated types)
 *   - C++ code sees both the C++ API and the C wrappers
 */
#ifndef __EDGEFIRST_METADATA_H__
#define __EDGEFIRST_METADATA_H__

#ifdef __cplusplus
#include <string>
#include <vector>

/**
 * @brief Read a named file from the ZIP archive appended to a model file.
 * @param model_path Path to the model file (e.g. .dvm, .tflite).
 * @param filename Name of the file within the ZIP (e.g. "edgefirst.json").
 * @return File contents as a string, or empty string if not found.
 */
std::string edgefirst_zip_read_file (const char *model_path,
    const char *filename);

/**
 * @brief Read labels.txt from a model file's ZIP archive.
 * @param model_path Path to the model file.
 * @return Ordered label vector (one per line, empty lines skipped).
 *         Empty vector if labels.txt is not present.
 */
std::vector<std::string> edgefirst_read_labels (const char *model_path);

/**
 * @brief Read edgefirst.json from a model file's ZIP archive.
 * @param model_path Path to the model file.
 * @return Raw JSON string, or empty string if not present.
 */
std::string edgefirst_read_metadata_json (const char *model_path);

extern "C" {
#endif

/**
 * @brief Read edgefirst.json from the model file's ZIP archive (C wrapper).
 * @param model_path Path to the model file.
 * @return g_malloc'd JSON string (caller must g_free), or NULL if not found.
 */
char *edgefirst_zip_read_metadata_c (const char *model_path);

/**
 * @brief Read labels from the model file's ZIP archive (C wrapper).
 *
 * Reads labels.txt from the ZIP archive appended to the model file.
 * Returns a NULL-terminated GStrv (caller must g_strfreev).
 *
 * @param model_path Path to the model file.
 * @param[out] num_labels Number of labels returned (may be NULL).
 * @return GStrv of labels, or NULL if no labels found.
 */
char **edgefirst_zip_read_labels_c (const char *model_path,
    unsigned int *num_labels);

#ifdef __cplusplus
}
#endif

#endif /* __EDGEFIRST_METADATA_H__ */
