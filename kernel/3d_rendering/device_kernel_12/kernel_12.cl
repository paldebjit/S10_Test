#include "ihc_apint.h"
#include "3d_rendering.h"

#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
__kernel void default_function(
                               __global coord_t* restrict triangle_3d, 
                               const int angle, 
                               __global frame_t* restrict outp
                              ) {

  int _top;
  int z_buffer[65536];
  int frame_buffer[65536];
  for (int x = 0; x < 256; ++x) {
    for (int y = 0; y < 256; ++y) {
      z_buffer[(y + (x * 256))] = 255;
    }
  }
  int main_body;
  for (int m = 0; m < 3191; ++m) {
    coord_t coord = triangle_3d[m];
    int triangle_3d_[9];
    #pragma unroll
    for (int x1 = 0; x1 < 9; ++x1) {
      triangle_3d_[x1] = coord.val[x1];
    }
  }
}
