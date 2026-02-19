/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * EdgeFirst Model Metadata Utilities
 * Copyright (C) 2026 Au-Zone Technologies <www.au-zone.com>
 */
/**
 * @file   edgefirst_metadata.cc
 * @date   19 Feb 2026
 * @brief  ZIP EOCD parser + edgefirst.json / labels.txt extraction
 * @author Sébastien Taylor <sebastien@au-zone.com>
 *
 * Reads files from a ZIP archive appended to model files (DVM, TFLite).
 * The ZIP End-of-Central-Directory (EOCD) is scanned backwards from the
 * end of the file, then the central directory is parsed to locate entries.
 * Supports stored (method 0) and deflated (method 8) entries via zlib.
 *
 * Only the EOCD tail, central directory, and target file entry are read
 * into memory — the model binary itself is never loaded.
 *
 * Dependencies: zlib (always available — GLib depends on it via GStreamer).
 */

#include "include/edgefirst_metadata.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include <glib.h>
#include <zlib.h>

/* ZIP format constants */
#define ZIP_EOCD_SIGNATURE      0x06054b50
#define ZIP_CENTRAL_DIR_SIG     0x02014b50
#define ZIP_EOCD_MIN_SIZE       22
#define ZIP_EOCD_MAX_COMMENT    65535
#define ZIP_METHOD_STORED       0
#define ZIP_METHOD_DEFLATED     8

/* Maximum bytes to read when searching for EOCD (22 + 64K comment) */
#define ZIP_EOCD_SEARCH_SIZE    (ZIP_EOCD_MIN_SIZE + ZIP_EOCD_MAX_COMMENT)

/**
 * @brief Read a 16-bit little-endian value from a byte buffer.
 */
static inline uint16_t
read_le16 (const uint8_t *p)
{
  return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

/**
 * @brief Read a 32-bit little-endian value from a byte buffer.
 */
static inline uint32_t
read_le32 (const uint8_t *p)
{
  return (uint32_t) p[0] | ((uint32_t) p[1] << 8)
      | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

/**
 * @brief Read a named file from the ZIP archive appended to a model file.
 *
 * Uses seek-based I/O: reads only the EOCD tail (~64KB max), the central
 * directory (typically a few KB), and the target file entry. The model
 * binary itself is never loaded into memory.
 *
 * Supports stored (method 0) and deflated (method 8) ZIP entries.
 */
std::string
edgefirst_zip_read_file (const char *model_path, const char *filename)
{
  if (!model_path || !filename)
    return {};

  FILE *fp = fopen (model_path, "rb");
  if (!fp)
    return {};

  /* Get file size */
  if (fseek (fp, 0, SEEK_END) != 0) {
    fclose (fp);
    return {};
  }
  long file_size = ftell (fp);
  if (file_size < ZIP_EOCD_MIN_SIZE) {
    fclose (fp);
    return {};
  }

  /* Read the tail of the file to search for EOCD signature.
   * The EOCD is at least 22 bytes and may be followed by a comment
   * up to 65535 bytes, so we read at most 22 + 65535 bytes from the end. */
  size_t tail_size = (size_t) file_size;
  if (tail_size > ZIP_EOCD_SEARCH_SIZE)
    tail_size = ZIP_EOCD_SEARCH_SIZE;

  long tail_offset = file_size - (long) tail_size;
  std::vector<uint8_t> tail (tail_size);

  if (fseek (fp, tail_offset, SEEK_SET) != 0) {
    fclose (fp);
    return {};
  }
  if (fread (tail.data (), 1, tail_size, fp) != tail_size) {
    fclose (fp);
    return {};
  }

  /* Scan backwards for EOCD signature (PK\x05\x06) */
  size_t eocd_rel = 0;
  bool found_eocd = false;

  for (size_t i = tail_size - ZIP_EOCD_MIN_SIZE; ; i--) {
    if (read_le32 (&tail[i]) == ZIP_EOCD_SIGNATURE) {
      eocd_rel = i;
      found_eocd = true;
      break;
    }
    if (i == 0)
      break;
  }

  if (!found_eocd) {
    fclose (fp);
    return {};
  }

  /* Parse EOCD */
  const uint8_t *eocd = &tail[eocd_rel];
  size_t eocd_abs = (size_t) tail_offset + eocd_rel;

  if (eocd_rel + ZIP_EOCD_MIN_SIZE > tail_size) {
    fclose (fp);
    return {};
  }

  uint16_t num_entries = read_le16 (eocd + 10);
  uint32_t cd_size = read_le32 (eocd + 12);
  uint32_t cd_offset = read_le32 (eocd + 16);

  /* When the ZIP is appended to another file (e.g. DVM + ZIP), the internal
   * offsets are relative to the start of the ZIP archive, not the combined
   * file. Compute the prefix offset: the actual central directory is at
   * eocd_abs - cd_size, but the EOCD says it's at cd_offset. */
  size_t actual_cd_pos = eocd_abs - cd_size;
  size_t prefix_offset = actual_cd_pos - cd_offset;

  /* Read central directory into memory */
  std::vector<uint8_t> cd (cd_size);
  if (fseek (fp, (long) actual_cd_pos, SEEK_SET) != 0) {
    fclose (fp);
    return {};
  }
  if (fread (cd.data (), 1, cd_size, fp) != cd_size) {
    fclose (fp);
    return {};
  }

  size_t target_len = strlen (filename);
  std::string result;

  /* Iterate central directory entries */
  size_t pos = 0;
  for (uint16_t e = 0; e < num_entries; e++) {
    if (pos + 46 > cd_size)
      break;
    if (read_le32 (&cd[pos]) != ZIP_CENTRAL_DIR_SIG)
      break;

    uint16_t method = read_le16 (&cd[pos + 10]);
    uint32_t compressed_size = read_le32 (&cd[pos + 20]);
    uint32_t uncompressed_size = read_le32 (&cd[pos + 24]);
    uint16_t name_len = read_le16 (&cd[pos + 28]);
    uint16_t extra_len = read_le16 (&cd[pos + 30]);
    uint16_t comment_len = read_le16 (&cd[pos + 32]);
    uint32_t local_offset_raw = read_le32 (&cd[pos + 42]);
    size_t local_offset = local_offset_raw + prefix_offset;

    /* Check if this is the file we want */
    if (name_len == target_len && pos + 46 + name_len <= cd_size
        && memcmp (&cd[pos + 46], filename, name_len) == 0) {

      /* Read local file header (30 bytes) to find data start */
      uint8_t local_hdr[30];
      if (fseek (fp, (long) local_offset, SEEK_SET) != 0)
        break;
      if (fread (local_hdr, 1, 30, fp) != 30)
        break;

      uint16_t local_name_len = read_le16 (local_hdr + 26);
      uint16_t local_extra_len = read_le16 (local_hdr + 28);
      size_t data_start = local_offset + 30 + local_name_len + local_extra_len;

      /* Read the compressed (or stored) file data */
      std::vector<uint8_t> entry_data (compressed_size);
      if (fseek (fp, (long) data_start, SEEK_SET) != 0)
        break;
      if (fread (entry_data.data (), 1, compressed_size, fp) != compressed_size)
        break;

      if (method == ZIP_METHOD_STORED) {
        result.assign (reinterpret_cast<const char *> (entry_data.data ()),
            compressed_size);
      } else if (method == ZIP_METHOD_DEFLATED) {
        /* ZIP uses raw deflate (no zlib header). Use inflate with
         * negative windowBits (-MAX_WBITS) for raw deflate stream. */
        result.resize (uncompressed_size);
        z_stream strm = {};
        strm.next_in = entry_data.data ();
        strm.avail_in = compressed_size;
        strm.next_out = reinterpret_cast<Bytef *> (&result[0]);
        strm.avail_out = uncompressed_size;

        if (inflateInit2 (&strm, -MAX_WBITS) != Z_OK) {
          result.clear ();
          break;
        }

        int zret = inflate (&strm, Z_FINISH);
        inflateEnd (&strm);

        if (zret != Z_STREAM_END) {
          result.clear ();
          break;
        }
        result.resize (strm.total_out);
      }
      /* else: unsupported method, result stays empty */
      break;
    }

    pos += 46 + name_len + extra_len + comment_len;
  }

  fclose (fp);
  return result;
}

/**
 * @brief Read edgefirst.json from the model file's ZIP archive.
 */
std::string
edgefirst_read_metadata_json (const char *model_path)
{
  return edgefirst_zip_read_file (model_path, "edgefirst.json");
}

/**
 * @brief Read labels.txt from the model file's ZIP archive.
 *
 * Splits on newlines, trims whitespace, skips empty lines.
 */
std::vector<std::string>
edgefirst_read_labels (const char *model_path)
{
  std::string content = edgefirst_zip_read_file (model_path, "labels.txt");
  if (content.empty ())
    return {};

  std::vector<std::string> labels;
  std::istringstream stream (content);
  std::string line;

  while (std::getline (stream, line)) {
    /* Trim trailing \r (Windows line endings) and whitespace */
    while (!line.empty ()
        && (line.back () == '\r' || line.back () == ' ' || line.back () == '\t'))
      line.pop_back ();

    if (!line.empty ())
      labels.push_back (line);
  }

  return labels;
}

/* ========== C wrappers for framework-level use ========== */

extern "C" {

/**
 * @brief Read edgefirst.json from model ZIP (C wrapper).
 * @param model_path Path to the model file.
 * @return g_strdup'd string or NULL. Caller must g_free().
 */
char *
edgefirst_zip_read_metadata_c (const char *model_path)
{
  std::string json = edgefirst_read_metadata_json (model_path);
  if (json.empty ())
    return NULL;
  return g_strdup (json.c_str ());
}

/**
 * @brief Read labels from model ZIP (C wrapper).
 *
 * Reads labels.txt from the ZIP archive appended to the model file.
 * Returns a NULL-terminated GStrv (caller must g_strfreev).
 *
 * @param model_path Path to the model file.
 * @param[out] num_labels Number of labels returned (may be NULL).
 * @return GStrv of labels, or NULL if no labels found.
 */
char **
edgefirst_zip_read_labels_c (const char *model_path,
    unsigned int *num_labels)
{
  if (num_labels)
    *num_labels = 0;

  std::vector<std::string> labels = edgefirst_read_labels (model_path);

  if (labels.empty ())
    return NULL;

  char **strv = g_new0 (char *, labels.size () + 1);
  for (size_t i = 0; i < labels.size (); i++)
    strv[i] = g_strdup (labels[i].c_str ());
  strv[labels.size ()] = NULL;

  if (num_labels)
    *num_labels = (unsigned int) labels.size ();

  return strv;
}

} /* extern "C" */
