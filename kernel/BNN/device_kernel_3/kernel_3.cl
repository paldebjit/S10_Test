#include "ihc_apint.h"
__kernel void defualt_function(__global float* restrict input_image, __global float* restrict fc){

float avgpool_LB[64];
for (int32_t i21 = 0; i21 < 1; ++i21) {
  for (int32_t c2 = 0; c2 < 64; ++c2) {
    for (int32_t h2 = 0; h2 < 1; ++h2) {
      for (int32_t w2 = 0; w2 < 1; ++w2) {
        float avgpool_val;
        for (int32_t avgpool_LB_i = 0; avgpool_LB_i < 8; ++avgpool_LB_i) {
          for (int32_t avgpool_LB_j = 0; avgpool_LB_j < 8; ++avgpool_LB_j) {
            avgpool_LB[(avgpool_LB_j + (avgpool_LB_i * 8))] = input_image[(((avgpool_LB_j + (((h2 * 8) + avgpool_LB_i) * 8)) + (c2 * 64)) + (i21 * 4096))];
          }
        }
        for (int32_t avgpool_rr = 0; avgpool_rr < 8; ++avgpool_rr) {
          for (int32_t avgpool_cc = 0; avgpool_cc < 8; ++avgpool_cc) {
            avgpool_val = ((float)(((float)avgpool_val) + ((float)avgpool_LB[(((w2 * 8) + avgpool_cc) + (avgpool_rr * 8))])));
          }
        }
        fc[(((w2 + h2) + c2) + (i21 * 64))] = ((float)(((float)avgpool_val) / (float)64));
      }
    }
  }
}
}
