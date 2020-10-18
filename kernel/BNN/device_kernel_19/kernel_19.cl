// HASH:1637752357
#include "ihc_apint.h"
__kernel void test(__global float* restrict input_image, __global float* restrict layer1_0_binarize1_shift_x_bias, __global float* restrict conv1_weight, __global float* restrict bn1, __global float* restrict bn1_running_mean, __global float* restrict bn1_running_var, __global float* restrict bn1_weight, __global float* restrict bn1_bias) {
    bool rsign1[3072];
    for (int32_t nn = 0; nn < 1; ++nn) {
      for (int32_t cc = 0; cc < 3; ++cc) {
        for (int32_t ww = 0; ww < 32; ++ww) {
          for (int32_t hh = 0; hh < 32; ++hh) {
            rsign1[(((hh + (ww * 32)) + (cc * 1024)) + (nn * 3072))] = ((bool)(int32_t)(((float)0 < ((float)(((float)input_image[(((hh + (ww * 32)) + (cc * 1024)) + (nn * 3072))]) + ((float)layer1_0_binarize1_shift_x_bias[cc])))) ? ((int32_t)1) : ((int32_t)0)));
          }
        }
      }
    }
    bool conv1_pad[3468];
    for (int32_t indices = 0; indices < 1; ++indices) {
      for (int32_t not_zero = 0; not_zero < 3; ++not_zero) {
        for (int32_t index_tuple = 0; index_tuple < 34; ++index_tuple) {
          for (int32_t i = 0; i < 34; ++i) {
            conv1_pad[(((i + (index_tuple * 34)) + (not_zero * 1156)) + (indices * 3468))] = (bool)(((((1 <= index_tuple) && (index_tuple < 33)) && (1 <= i)) && (i < 33)) ? ((bool)rsign1[((((i + (index_tuple * 32)) + (not_zero * 1024)) + (indices * 3072)) + -33)]) : ((bool)0));
          }
        }
      }
    }
    int8_t conv1[16384];
    for (int32_t nn1 = 0; nn1 < 1; ++nn1) {
      for (int32_t ff = 0; ff < 16; ++ff) {
        for (int32_t yy = 0; yy < 32; ++yy) {
          for (int32_t xx = 0; xx < 32; ++xx) {
            int8_t conv1_sum;
            for (int32_t rc = 0; rc < 3; ++rc) {
              for (int32_t ry = 0; ry < 3; ++ry) {
                for (int32_t rx = 0; rx < 3; ++rx) {
                  conv1_sum = ((int8_t)(((int64_t)(uint32_t)(((((((int64_t)1 - ((int64_t)rx)) <= ((int64_t)xx)) && (((int64_t)xx) < ((int64_t)33 - ((int64_t)rx)))) && (((int64_t)1 - ((int64_t)ry)) <= ((int64_t)yy))) && (((int64_t)yy) < ((int64_t)33 - ((int64_t)ry)))) ? ((uint32_t)((((1U - ((uint32_t)conv1_pad[((((xx + rx) + ((yy + ry) * 34)) + (rc * 1156)) + (nn1 * 3468))])) ^ conv1_weight[(((rx + (ry * 3)) + (rc * 9)) + (ff * 27))]) << 1) - 1U)) : ((uint32_t)0U))) + ((int64_t)conv1_sum)));
                }
              }
            }
            conv1[(((xx + (yy * 32)) + (ff * 1024)) + (nn1 * 16384))] = conv1_sum;
          }
        }
      }
    }
    for (int32_t x = 0; x < 1; ++x) {
      for (int32_t args0 = 0; args0 < 16; ++args0) {
        for (int32_t args1 = 0; args1 < 32; ++args1) {
          #pragma unroll 1
          for (int32_t args2 = 0; args2 < 32; ++args2) {
            bn1[(((args2 + (args1 * 32)) + (args0 * 1024)) + (x * 16384))] = ((float)((((float)(((float)conv1[(((args2 + (args1 * 32)) + (args0 * 1024)) + (x * 16384))]))) * ((float)bn1_weight[args0])) + ((float)bn1_bias[args0])));
          }
        }
      }
    }
}

