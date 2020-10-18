// HASH:587118235
#include "ihc_apint.h"
__kernel void test(__global float* restrict input_image, __global float* restrict conv1_weight, __global float* restrict bn1_running_mean, __global float* restrict bn1_running_var, __global float* restrict bn1_weight, __global float* restrict bn1_bias, __global float* restrict layer1_0_binarize1_shift_x_bias, __global bool* restrict layer1_0_conv1_weight, __global int8_t* restrict conv2) {
    float conv1_pad[3468];
    for (int32_t indices = 0; indices < 1; ++indices) {
      for (int32_t not_zero = 0; not_zero < 3; ++not_zero) {
        for (int32_t index_tuple = 0; index_tuple < 34; ++index_tuple) {
          for (int32_t i = 0; i < 34; ++i) {
            conv1_pad[(((i + (index_tuple * 34)) + (not_zero * 1156)) + (indices * 3468))] = (float)(((((1 <= index_tuple) && (index_tuple < 33)) && (1 <= i)) && (i < 33)) ? ((float)input_image[((((i + (index_tuple * 32)) + (not_zero * 1024)) + (indices * 3072)) + -33)]) : ((float)0));
          }
        }
      }
    }
    float conv1[16384];
    for (int32_t nn = 0; nn < 1; ++nn) {
      for (int32_t ff = 0; ff < 16; ++ff) {
        for (int32_t yy = 0; yy < 32; ++yy) {
          for (int32_t xx = 0; xx < 32; ++xx) {
            float sum;
            for (int32_t rc = 0; rc < 3; ++rc) {
              for (int32_t ry = 0; ry < 3; ++ry) {
                for (int32_t rx = 0; rx < 3; ++rx) {
                  sum = ((float)(((float)(((float)conv1_pad[((((xx + rx) + ((yy + ry) * 34)) + (rc * 1156)) + (nn * 3468))]) * ((float)conv1_weight[(((rx + (ry * 3)) + (rc * 9)) + (ff * 27))]))) + ((float)sum)));
                }
              }
            }
            conv1[(((xx + (yy * 32)) + (ff * 1024)) + (nn * 16384))] = sum;
          }
        }
      }
    }
    float bn1[16384];
    for (int32_t x = 0; x < 1; ++x) {
      for (int32_t args0 = 0; args0 < 16; ++args0) {
        for (int32_t args1 = 0; args1 < 32; ++args1) {
          for (int32_t args2 = 0; args2 < 32; ++args2) {
            bn1[(((args2 + (args1 * 32)) + (args0 * 1024)) + (x * 16384))] = ((float)((((float)(((float)conv1[(((args2 + (args1 * 32)) + (args0 * 1024)) + (x * 16384))]))) * ((float)bn1_weight[args0])) + ((float)bn1_bias[args0])));
          }
        }
      }
    }
    bool rsign1[16384];
    for (int32_t nn1 = 0; nn1 < 1; ++nn1) {
      for (int32_t cc = 0; cc < 16; ++cc) {
        for (int32_t ww = 0; ww < 32; ++ww) {
          for (int32_t hh = 0; hh < 32; ++hh) {
            rsign1[(((hh + (ww * 32)) + (cc * 1024)) + (nn1 * 16384))] = ((bool)(int32_t)(((float)0 < ((float)(((float)bn1[(((hh + (ww * 32)) + (cc * 1024)) + (nn1 * 16384))]) + ((float)layer1_0_binarize1_shift_x_bias[cc])))) ? ((int32_t)1) : ((int32_t)0)));
          }
        }
      }
    }
    bool conv2_pad[18496];
    for (int32_t indices1 = 0; indices1 < 1; ++indices1) {
      for (int32_t not_zero1 = 0; not_zero1 < 16; ++not_zero1) {
        for (int32_t index_tuple1 = 0; index_tuple1 < 34; ++index_tuple1) {
          for (int32_t i1 = 0; i1 < 34; ++i1) {
            conv2_pad[(((i1 + (index_tuple1 * 34)) + (not_zero1 * 1156)) + (indices1 * 18496))] = (bool)(((((1 <= index_tuple1) && (index_tuple1 < 33)) && (1 <= i1)) && (i1 < 33)) ? ((bool)rsign1[((((i1 + (index_tuple1 * 32)) + (not_zero1 * 1024)) + (indices1 * 16384)) + -33)]) : ((bool)0));
          }
        }
      }
    }
    for (int32_t nn2 = 0; nn2 < 1; ++nn2) {
      for (int32_t ff1 = 0; ff1 < 16; ++ff1) {
        for (int32_t yy1 = 0; yy1 < 32; ++yy1) {
          for (int32_t xx1 = 0; xx1 < 32; ++xx1) {
            int8_t conv2_sum;
            for (int32_t rc1 = 0; rc1 < 16; ++rc1) {
              for (int32_t ry1 = 0; ry1 < 3; ++ry1) {
                for (int32_t rx1 = 0; rx1 < 3; ++rx1) {
                  conv2_sum = ((int8_t)(((int64_t)(uint32_t)(((((((int64_t)1 - ((int64_t)rx1)) <= ((int64_t)xx1)) && (((int64_t)xx1) < ((int64_t)33 - ((int64_t)rx1)))) && (((int64_t)1 - ((int64_t)ry1)) <= ((int64_t)yy1))) && (((int64_t)yy1) < ((int64_t)33 - ((int64_t)ry1)))) ? ((uint32_t)((((1U - ((uint32_t)conv2_pad[((((xx1 + rx1) + ((yy1 + ry1) * 34)) + (rc1 * 1156)) + (nn2 * 18496))])) ^ layer1_0_conv1_weight[(((rx1 + (ry1 * 3)) + (rc1 * 9)) + (ff1 * 144))]) << 1) - 1U)) : ((uint32_t)0U))) + ((int64_t)conv2_sum)));
                }
              }
            }
            conv2[(((xx1 + (yy1 * 32)) + (ff1 * 1024)) + (nn2 * 16384))] = conv2_sum;
          }
        }
      }
    }
}

