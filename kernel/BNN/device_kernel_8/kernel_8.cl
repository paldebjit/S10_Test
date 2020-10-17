// HASH:635185065
#include "ihc_apint.h"
__kernel void test(__global float* restrict input_image, __global float* restrict conv1_weight, __global float* restrict bn1, __global float* restrict bn1_running_mean, __global float* restrict bn1_running_var, __global float* restrict bn1_weight, __global float* restrict bn1_bias) {
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
    for (int32_t x = 0; x < 1; ++x) {
      for (int32_t args0 = 0; args0 < 16; ++args0) {
        for (int32_t args1 = 0; args1 < 32; ++args1) {
          for (int32_t args2 = 0; args2 < 32; ++args2) {
            bn1[(((args2 + (args1 * 32)) + (args0 * 1024)) + (x * 16384))] = ((float)(((((float)(((float)conv1[(((args2 + (args1 * 32)) + (args0 * 1024)) + (x * 16384))]) - ((float)bn1_running_mean[args0]))) / sqrtf((((float)bn1_running_var[args0]) + 1.000000e-07f))) * ((float)bn1_weight[args0])) + ((float)bn1_bias[args0])));
          }
        }
      }
    }
}

