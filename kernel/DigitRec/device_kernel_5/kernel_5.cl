#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
__kernel void default_function(const uint64_t test_image, __global uint64_t* restrict train_images, __global uint8_t* restrict knn_mat) {
  uint64_t _top;
  for (int32_t x = 0; x < 10; ++x) {
    for (int32_t y = 0; y < 3; ++y) {
      knn_mat[(y + (x * 3))] = (uint8_t)50;
    }
  }
  uint64_t knn_update;
  #pragma ivdep
  for (int32_t y1 = 0; y1 < 1800; ++y1) {
    //#pragma ii 1
    for (int32_t x1 = 0; x1 < 10; ++x1) {
      uint8_t dist;
      uint64_t diff;
      diff = (train_images[(y1 + (x1 * 1800))] ^ test_image);
      uint8_t out;
      for (int32_t i = 0; i < 49; ++i) {
        out = ((uint8_t)(((uint64_t)out) + ((uint64_t)((diff & (1L << i)) >> i))));
      }
      dist = out;
      uint64_t max_id = 0;
      //for (int32_t i1 = 0; i1 < 3; ++i1) {
      //  if (knn_mat[(max_id + ((uint64_t)(x1 * 3)))] < knn_mat[(i1 + (x1 * 3))]) {
      //    max_id = ((uint64_t)i1);
      //  }
      //}
      //if (dist < knn_mat[(max_id + ((uint64_t)(x1 * 3)))]) {
      knn_mat[(max_id + ((uint64_t)(x1 * 3)))] = dist;
      //}
    }
  }
}

