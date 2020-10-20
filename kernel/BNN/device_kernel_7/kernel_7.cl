#include "ihc_apint.h"
__kernel void defualt_function(__global float* restrict input_image, __global float* restrict fc){

 //float input_image_[1][64][8][8];
 float input_image_[8][64][8][1];
 
 for (int32_t i = 0; i < 1; ++i) {
   for (int32_t j = 0; j < 8; ++j) {
     for (int32_t k = 0; k < 64; ++k) {
       for (int32_t l = 0; l < 8; ++l) {
         input_image_[l][k][j][i] = input_image[(((l + (k * 64)) + (j * 8)) + (i * 4096))];
       }
     }
   }
 }
 
 float avgpool_LB[4][4];
 #pragma unroll 1
 for (int32_t i21 = 0; i21 < 1; ++i21) {
   for (int32_t c2 = 0; c2 < 8; ++c2) {
     for (int32_t h2 = 0; h2 < 1; ++h2) {
       for (int32_t w2 = 0; w2 < 1; ++w2) {
         float avgpool_val;
         for (int32_t avgpool_LB_i = 0; avgpool_LB_i < 4; ++avgpool_LB_i) {
           for (int32_t avgpool_LB_j = 0; avgpool_LB_j < 4; ++avgpool_LB_j) {
             int32_t idx = (((h2<<3) + avgpool_LB_i)<<3);
             avgpool_LB[avgpool_LB_j][avgpool_LB_i] = input_image_[avgpool_LB_j][idx][c2][i21];
             //fc[avgpool_LB_j + avgpool_LB_i * 8] = input_image_[avgpool_LB_j][idx][c2][i21];
           }
         }

         for (int32_t avgpool_rr = 0; avgpool_rr < 4; ++avgpool_rr) {
           for (int32_t avgpool_cc = 0; avgpool_cc < 4; ++avgpool_cc) {
             avgpool_val = ((float)(((float)avgpool_val) + ((float)avgpool_LB[avgpool_cc][avgpool_rr])));
             //fc[0] = ((float)(((float)fc[0]) + ((float)avgpool_LB[avgpool_cc][avgpool_rr])));
           }
          }
         fc[(((w2 + h2) + c2) + (i21 * 64))] = ((float)(((float)avgpool_val) / (float)64));

       }
     }
   }
 }
}
