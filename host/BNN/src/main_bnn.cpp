
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// rapidjson headers
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
using namespace rapidjson;

#include "CL/opencl.h"
#pragma message ("* Compiling for ALTERA CL")
#define AOCX_FILE "AOCX/bnn.aocx"



#define CHECK(status) 							\
    if (status != CL_SUCCESS)						\
{									\
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
    exit(1);								\
}									\

void* acl_aligned_malloc (size_t size) {
  void *result = NULL;
  posix_memalign (&result, 64, size);
  return result;
}


int main(int argc, char ** argv) {
  std::cout << "[INFO] Initialize input buffers...\n";

  FILE *f = fopen("inputs.json", "r");
  char readBuffer[65536];
  FileReadStream is(f, readBuffer, sizeof(readBuffer));

  Document document;
  document.ParseStream(is);
  fclose(f);
  assert(document.HasMember("input_image"));
  const Value& input_image_d = document["input_image"];
  assert(input_image_d.IsArray());
  auto input_image = new float[1][3][32][32];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      for (size_t i2 = 0; i2 < 32; i2++) {
        for (size_t i3 = 0; i3 < 32; i3++) {
          input_image[i0][i1][i2][i3] = (input_image_d[i3 + i2*32 + i1*1024 + i0*3072].GetFloat()) / 1000.0;
        }
      }
    }
  }

  assert(document.HasMember("conv1_weight"));
  const Value& conv1_weight_d = document["conv1_weight"];
  assert(conv1_weight_d.IsArray());
  auto conv1_weight = new float[16][3][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          conv1_weight[i0][i1][i2][i3] = (conv1_weight_d[i3 + i2*3 + i1*9 + i0*27].GetFloat()) / 1000.0;
        }
      }
    }
  }

  assert(document.HasMember("bn1_weight"));
  const Value& bn1_weight_d = document["bn1_weight"];
  assert(bn1_weight_d.IsArray());
  auto bn1_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_weight[i0] = (bn1_weight_d[i0].GetFloat()) / 1000.00;
  }

  assert(document.HasMember("bn1_bias"));
  const Value& bn1_bias_d = document["bn1_bias"];
  assert(bn1_bias_d.IsArray());
  auto bn1_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_bias[i0] = (bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("bn1_running_mean"));
  const Value& bn1_running_mean_d = document["bn1_running_mean"];
  assert(bn1_running_mean_d.IsArray());
  auto bn1_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_running_mean[i0] = (bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("bn1_running_var"));
  const Value& bn1_running_var_d = document["bn1_running_var"];
  assert(bn1_running_var_d.IsArray());
  auto bn1_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_running_var[i0] = (bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_rprelu1_shift_x_bias"));
  const Value& layer1_0_rprelu1_shift_x_bias_d = document["layer1_0_rprelu1_shift_x_bias"];
  assert(layer1_0_rprelu1_shift_x_bias_d.IsArray());
  auto layer1_0_rprelu1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_rprelu1_shift_x_bias[i0] = (layer1_0_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_rprelu1_shift_y_bias"));
  const Value& layer1_0_rprelu1_shift_y_bias_d = document["layer1_0_rprelu1_shift_y_bias"];
  assert(layer1_0_rprelu1_shift_y_bias_d.IsArray());
  auto layer1_0_rprelu1_shift_y_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_rprelu1_shift_y_bias[i0] = (layer1_0_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_rprelu1_prelu_weight"));
  const Value& layer1_0_rprelu1_prelu_weight_d = document["layer1_0_rprelu1_prelu_weight"];
  assert(layer1_0_rprelu1_prelu_weight_d.IsArray());
  auto layer1_0_rprelu1_prelu_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_rprelu1_prelu_weight[i0] = (layer1_0_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_rprelu2_shift_x_bias"));
  const Value& layer1_0_rprelu2_shift_x_bias_d = document["layer1_0_rprelu2_shift_x_bias"];
  assert(layer1_0_rprelu2_shift_x_bias_d.IsArray());
  auto layer1_0_rprelu2_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_rprelu2_shift_x_bias[i0] = (layer1_0_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_rprelu2_shift_y_bias"));
  const Value& layer1_0_rprelu2_shift_y_bias_d = document["layer1_0_rprelu2_shift_y_bias"];
  assert(layer1_0_rprelu2_shift_y_bias_d.IsArray());
  auto layer1_0_rprelu2_shift_y_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_rprelu2_shift_y_bias[i0] = (layer1_0_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_rprelu2_prelu_weight"));
  const Value& layer1_0_rprelu2_prelu_weight_d = document["layer1_0_rprelu2_prelu_weight"];
  assert(layer1_0_rprelu2_prelu_weight_d.IsArray());
  auto layer1_0_rprelu2_prelu_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_rprelu2_prelu_weight[i0] = (layer1_0_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_binarize1_shift_x_bias"));
  const Value& layer1_0_binarize1_shift_x_bias_d = document["layer1_0_binarize1_shift_x_bias"];
  assert(layer1_0_binarize1_shift_x_bias_d.IsArray());
  auto layer1_0_binarize1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_binarize1_shift_x_bias[i0] = (layer1_0_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_binarize2_shift_x_bias"));
  const Value& layer1_0_binarize2_shift_x_bias_d = document["layer1_0_binarize2_shift_x_bias"];
  assert(layer1_0_binarize2_shift_x_bias_d.IsArray());
  auto layer1_0_binarize2_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_binarize2_shift_x_bias[i0] = (layer1_0_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_conv1_weight"));
  const Value& layer1_0_conv1_weight_d = document["layer1_0_conv1_weight"];
  assert(layer1_0_conv1_weight_d.IsArray());
  auto layer1_0_conv1_weight = new uint8_t[16][16][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer1_0_conv1_weight[i0][i1][i2][i3] = (layer1_0_conv1_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer1_0_bn1_weight"));
  const Value& layer1_0_bn1_weight_d = document["layer1_0_bn1_weight"];
  assert(layer1_0_bn1_weight_d.IsArray());
  auto layer1_0_bn1_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn1_weight[i0] = (layer1_0_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_bn1_bias"));
  const Value& layer1_0_bn1_bias_d = document["layer1_0_bn1_bias"];
  assert(layer1_0_bn1_bias_d.IsArray());
  auto layer1_0_bn1_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn1_bias[i0] = (layer1_0_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_bn1_running_mean"));
  const Value& layer1_0_bn1_running_mean_d = document["layer1_0_bn1_running_mean"];
  assert(layer1_0_bn1_running_mean_d.IsArray());
  auto layer1_0_bn1_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn1_running_mean[i0] = (layer1_0_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_bn1_running_var"));
  const Value& layer1_0_bn1_running_var_d = document["layer1_0_bn1_running_var"];
  assert(layer1_0_bn1_running_var_d.IsArray());
  auto layer1_0_bn1_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn1_running_var[i0] = (layer1_0_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_conv2_weight"));
  const Value& layer1_0_conv2_weight_d = document["layer1_0_conv2_weight"];
  assert(layer1_0_conv2_weight_d.IsArray());
  auto layer1_0_conv2_weight = new uint8_t[16][16][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer1_0_conv2_weight[i0][i1][i2][i3] = (layer1_0_conv2_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer1_0_bn2_weight"));
  const Value& layer1_0_bn2_weight_d = document["layer1_0_bn2_weight"];
  assert(layer1_0_bn2_weight_d.IsArray());
  auto layer1_0_bn2_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn2_weight[i0] = (layer1_0_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_bn2_bias"));
  const Value& layer1_0_bn2_bias_d = document["layer1_0_bn2_bias"];
  assert(layer1_0_bn2_bias_d.IsArray());
  auto layer1_0_bn2_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn2_bias[i0] = (layer1_0_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_bn2_running_mean"));
  const Value& layer1_0_bn2_running_mean_d = document["layer1_0_bn2_running_mean"];
  assert(layer1_0_bn2_running_mean_d.IsArray());
  auto layer1_0_bn2_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn2_running_mean[i0] = (layer1_0_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_0_bn2_running_var"));
  const Value& layer1_0_bn2_running_var_d = document["layer1_0_bn2_running_var"];
  assert(layer1_0_bn2_running_var_d.IsArray());
  auto layer1_0_bn2_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_0_bn2_running_var[i0] = (layer1_0_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_rprelu1_shift_x_bias"));
  const Value& layer1_1_rprelu1_shift_x_bias_d = document["layer1_1_rprelu1_shift_x_bias"];
  assert(layer1_1_rprelu1_shift_x_bias_d.IsArray());
  auto layer1_1_rprelu1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_rprelu1_shift_x_bias[i0] = (layer1_1_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_rprelu1_shift_y_bias"));
  const Value& layer1_1_rprelu1_shift_y_bias_d = document["layer1_1_rprelu1_shift_y_bias"];
  assert(layer1_1_rprelu1_shift_y_bias_d.IsArray());
  auto layer1_1_rprelu1_shift_y_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_rprelu1_shift_y_bias[i0] = (layer1_1_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_rprelu1_prelu_weight"));
  const Value& layer1_1_rprelu1_prelu_weight_d = document["layer1_1_rprelu1_prelu_weight"];
  assert(layer1_1_rprelu1_prelu_weight_d.IsArray());
  auto layer1_1_rprelu1_prelu_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_rprelu1_prelu_weight[i0] = (layer1_1_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_rprelu2_shift_x_bias"));
  const Value& layer1_1_rprelu2_shift_x_bias_d = document["layer1_1_rprelu2_shift_x_bias"];
  assert(layer1_1_rprelu2_shift_x_bias_d.IsArray());
  auto layer1_1_rprelu2_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_rprelu2_shift_x_bias[i0] = (layer1_1_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_rprelu2_shift_y_bias"));
  const Value& layer1_1_rprelu2_shift_y_bias_d = document["layer1_1_rprelu2_shift_y_bias"];
  assert(layer1_1_rprelu2_shift_y_bias_d.IsArray());
  auto layer1_1_rprelu2_shift_y_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_rprelu2_shift_y_bias[i0] = (layer1_1_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_rprelu2_prelu_weight"));
  const Value& layer1_1_rprelu2_prelu_weight_d = document["layer1_1_rprelu2_prelu_weight"];
  assert(layer1_1_rprelu2_prelu_weight_d.IsArray());
  auto layer1_1_rprelu2_prelu_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_rprelu2_prelu_weight[i0] = (layer1_1_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_binarize1_shift_x_bias"));
  const Value& layer1_1_binarize1_shift_x_bias_d = document["layer1_1_binarize1_shift_x_bias"];
  assert(layer1_1_binarize1_shift_x_bias_d.IsArray());
  auto layer1_1_binarize1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_binarize1_shift_x_bias[i0] = (layer1_1_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_binarize2_shift_x_bias"));
  const Value& layer1_1_binarize2_shift_x_bias_d = document["layer1_1_binarize2_shift_x_bias"];
  assert(layer1_1_binarize2_shift_x_bias_d.IsArray());
  auto layer1_1_binarize2_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_binarize2_shift_x_bias[i0] = (layer1_1_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_conv1_weight"));
  const Value& layer1_1_conv1_weight_d = document["layer1_1_conv1_weight"];
  assert(layer1_1_conv1_weight_d.IsArray());
  auto layer1_1_conv1_weight = new uint8_t[16][16][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer1_1_conv1_weight[i0][i1][i2][i3] = (layer1_1_conv1_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer1_1_bn1_weight"));
  const Value& layer1_1_bn1_weight_d = document["layer1_1_bn1_weight"];
  assert(layer1_1_bn1_weight_d.IsArray());
  auto layer1_1_bn1_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn1_weight[i0] = (layer1_1_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_bn1_bias"));
  const Value& layer1_1_bn1_bias_d = document["layer1_1_bn1_bias"];
  assert(layer1_1_bn1_bias_d.IsArray());
  auto layer1_1_bn1_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn1_bias[i0] = (layer1_1_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_bn1_running_mean"));
  const Value& layer1_1_bn1_running_mean_d = document["layer1_1_bn1_running_mean"];
  assert(layer1_1_bn1_running_mean_d.IsArray());
  auto layer1_1_bn1_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn1_running_mean[i0] = (layer1_1_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_bn1_running_var"));
  const Value& layer1_1_bn1_running_var_d = document["layer1_1_bn1_running_var"];
  assert(layer1_1_bn1_running_var_d.IsArray());
  auto layer1_1_bn1_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn1_running_var[i0] = (layer1_1_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_conv2_weight"));
  const Value& layer1_1_conv2_weight_d = document["layer1_1_conv2_weight"];
  assert(layer1_1_conv2_weight_d.IsArray());
  auto layer1_1_conv2_weight = new uint8_t[16][16][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer1_1_conv2_weight[i0][i1][i2][i3] = (layer1_1_conv2_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer1_1_bn2_weight"));
  const Value& layer1_1_bn2_weight_d = document["layer1_1_bn2_weight"];
  assert(layer1_1_bn2_weight_d.IsArray());
  auto layer1_1_bn2_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn2_weight[i0] = (layer1_1_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_bn2_bias"));
  const Value& layer1_1_bn2_bias_d = document["layer1_1_bn2_bias"];
  assert(layer1_1_bn2_bias_d.IsArray());
  auto layer1_1_bn2_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn2_bias[i0] = (layer1_1_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_bn2_running_mean"));
  const Value& layer1_1_bn2_running_mean_d = document["layer1_1_bn2_running_mean"];
  assert(layer1_1_bn2_running_mean_d.IsArray());
  auto layer1_1_bn2_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn2_running_mean[i0] = (layer1_1_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_1_bn2_running_var"));
  const Value& layer1_1_bn2_running_var_d = document["layer1_1_bn2_running_var"];
  assert(layer1_1_bn2_running_var_d.IsArray());
  auto layer1_1_bn2_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_1_bn2_running_var[i0] = (layer1_1_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_rprelu1_shift_x_bias"));
  const Value& layer1_2_rprelu1_shift_x_bias_d = document["layer1_2_rprelu1_shift_x_bias"];
  assert(layer1_2_rprelu1_shift_x_bias_d.IsArray());
  auto layer1_2_rprelu1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_rprelu1_shift_x_bias[i0] = (layer1_2_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_rprelu1_shift_y_bias"));
  const Value& layer1_2_rprelu1_shift_y_bias_d = document["layer1_2_rprelu1_shift_y_bias"];
  assert(layer1_2_rprelu1_shift_y_bias_d.IsArray());
  auto layer1_2_rprelu1_shift_y_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_rprelu1_shift_y_bias[i0] = (layer1_2_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_rprelu1_prelu_weight"));
  const Value& layer1_2_rprelu1_prelu_weight_d = document["layer1_2_rprelu1_prelu_weight"];
  assert(layer1_2_rprelu1_prelu_weight_d.IsArray());
  auto layer1_2_rprelu1_prelu_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_rprelu1_prelu_weight[i0] = (layer1_2_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_rprelu2_shift_x_bias"));
  const Value& layer1_2_rprelu2_shift_x_bias_d = document["layer1_2_rprelu2_shift_x_bias"];
  assert(layer1_2_rprelu2_shift_x_bias_d.IsArray());
  auto layer1_2_rprelu2_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_rprelu2_shift_x_bias[i0] = (layer1_2_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_rprelu2_shift_y_bias"));
  const Value& layer1_2_rprelu2_shift_y_bias_d = document["layer1_2_rprelu2_shift_y_bias"];
  assert(layer1_2_rprelu2_shift_y_bias_d.IsArray());
  auto layer1_2_rprelu2_shift_y_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_rprelu2_shift_y_bias[i0] = (layer1_2_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_rprelu2_prelu_weight"));
  const Value& layer1_2_rprelu2_prelu_weight_d = document["layer1_2_rprelu2_prelu_weight"];
  assert(layer1_2_rprelu2_prelu_weight_d.IsArray());
  auto layer1_2_rprelu2_prelu_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_rprelu2_prelu_weight[i0] = (layer1_2_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_binarize1_shift_x_bias"));
  const Value& layer1_2_binarize1_shift_x_bias_d = document["layer1_2_binarize1_shift_x_bias"];
  assert(layer1_2_binarize1_shift_x_bias_d.IsArray());
  auto layer1_2_binarize1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_binarize1_shift_x_bias[i0] = (layer1_2_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_binarize2_shift_x_bias"));
  const Value& layer1_2_binarize2_shift_x_bias_d = document["layer1_2_binarize2_shift_x_bias"];
  assert(layer1_2_binarize2_shift_x_bias_d.IsArray());
  auto layer1_2_binarize2_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_binarize2_shift_x_bias[i0] = (layer1_2_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_conv1_weight"));
  const Value& layer1_2_conv1_weight_d = document["layer1_2_conv1_weight"];
  assert(layer1_2_conv1_weight_d.IsArray());
  auto layer1_2_conv1_weight = new uint8_t[16][16][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer1_2_conv1_weight[i0][i1][i2][i3] = (layer1_2_conv1_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer1_2_bn1_weight"));
  const Value& layer1_2_bn1_weight_d = document["layer1_2_bn1_weight"];
  assert(layer1_2_bn1_weight_d.IsArray());
  auto layer1_2_bn1_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn1_weight[i0] = (layer1_2_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_bn1_bias"));
  const Value& layer1_2_bn1_bias_d = document["layer1_2_bn1_bias"];
  assert(layer1_2_bn1_bias_d.IsArray());
  auto layer1_2_bn1_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn1_bias[i0] = (layer1_2_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_bn1_running_mean"));
  const Value& layer1_2_bn1_running_mean_d = document["layer1_2_bn1_running_mean"];
  assert(layer1_2_bn1_running_mean_d.IsArray());
  auto layer1_2_bn1_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn1_running_mean[i0] = (layer1_2_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_bn1_running_var"));
  const Value& layer1_2_bn1_running_var_d = document["layer1_2_bn1_running_var"];
  assert(layer1_2_bn1_running_var_d.IsArray());
  auto layer1_2_bn1_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn1_running_var[i0] = (layer1_2_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_conv2_weight"));
  const Value& layer1_2_conv2_weight_d = document["layer1_2_conv2_weight"];
  assert(layer1_2_conv2_weight_d.IsArray());
  auto layer1_2_conv2_weight = new uint8_t[16][16][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer1_2_conv2_weight[i0][i1][i2][i3] = (layer1_2_conv2_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer1_2_bn2_weight"));
  const Value& layer1_2_bn2_weight_d = document["layer1_2_bn2_weight"];
  assert(layer1_2_bn2_weight_d.IsArray());
  auto layer1_2_bn2_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn2_weight[i0] = (layer1_2_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_bn2_bias"));
  const Value& layer1_2_bn2_bias_d = document["layer1_2_bn2_bias"];
  assert(layer1_2_bn2_bias_d.IsArray());
  auto layer1_2_bn2_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn2_bias[i0] = (layer1_2_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_bn2_running_mean"));
  const Value& layer1_2_bn2_running_mean_d = document["layer1_2_bn2_running_mean"];
  assert(layer1_2_bn2_running_mean_d.IsArray());
  auto layer1_2_bn2_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn2_running_mean[i0] = (layer1_2_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer1_2_bn2_running_var"));
  const Value& layer1_2_bn2_running_var_d = document["layer1_2_bn2_running_var"];
  assert(layer1_2_bn2_running_var_d.IsArray());
  auto layer1_2_bn2_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer1_2_bn2_running_var[i0] = (layer1_2_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_rprelu1_shift_x_bias"));
  const Value& layer2_0_rprelu1_shift_x_bias_d = document["layer2_0_rprelu1_shift_x_bias"];
  assert(layer2_0_rprelu1_shift_x_bias_d.IsArray());
  auto layer2_0_rprelu1_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_rprelu1_shift_x_bias[i0] = (layer2_0_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_rprelu1_shift_y_bias"));
  const Value& layer2_0_rprelu1_shift_y_bias_d = document["layer2_0_rprelu1_shift_y_bias"];
  assert(layer2_0_rprelu1_shift_y_bias_d.IsArray());
  auto layer2_0_rprelu1_shift_y_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_rprelu1_shift_y_bias[i0] = (layer2_0_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_rprelu1_prelu_weight"));
  const Value& layer2_0_rprelu1_prelu_weight_d = document["layer2_0_rprelu1_prelu_weight"];
  assert(layer2_0_rprelu1_prelu_weight_d.IsArray());
  auto layer2_0_rprelu1_prelu_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_rprelu1_prelu_weight[i0] = (layer2_0_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_rprelu2_shift_x_bias"));
  const Value& layer2_0_rprelu2_shift_x_bias_d = document["layer2_0_rprelu2_shift_x_bias"];
  assert(layer2_0_rprelu2_shift_x_bias_d.IsArray());
  auto layer2_0_rprelu2_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_rprelu2_shift_x_bias[i0] = (layer2_0_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_rprelu2_shift_y_bias"));
  const Value& layer2_0_rprelu2_shift_y_bias_d = document["layer2_0_rprelu2_shift_y_bias"];
  assert(layer2_0_rprelu2_shift_y_bias_d.IsArray());
  auto layer2_0_rprelu2_shift_y_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_rprelu2_shift_y_bias[i0] = (layer2_0_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_rprelu2_prelu_weight"));
  const Value& layer2_0_rprelu2_prelu_weight_d = document["layer2_0_rprelu2_prelu_weight"];
  assert(layer2_0_rprelu2_prelu_weight_d.IsArray());
  auto layer2_0_rprelu2_prelu_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_rprelu2_prelu_weight[i0] = (layer2_0_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_binarize1_shift_x_bias"));
  const Value& layer2_0_binarize1_shift_x_bias_d = document["layer2_0_binarize1_shift_x_bias"];
  assert(layer2_0_binarize1_shift_x_bias_d.IsArray());
  auto layer2_0_binarize1_shift_x_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    layer2_0_binarize1_shift_x_bias[i0] = (layer2_0_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_binarize2_shift_x_bias"));
  const Value& layer2_0_binarize2_shift_x_bias_d = document["layer2_0_binarize2_shift_x_bias"];
  assert(layer2_0_binarize2_shift_x_bias_d.IsArray());
  auto layer2_0_binarize2_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_binarize2_shift_x_bias[i0] = (layer2_0_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_conv1_weight"));
  const Value& layer2_0_conv1_weight_d = document["layer2_0_conv1_weight"];
  assert(layer2_0_conv1_weight_d.IsArray());
  auto layer2_0_conv1_weight = new uint8_t[32][16][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer2_0_conv1_weight[i0][i1][i2][i3] = (layer2_0_conv1_weight_d[i3 + i2*3 + i1*9 + i0*144].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer2_0_bn1_weight"));
  const Value& layer2_0_bn1_weight_d = document["layer2_0_bn1_weight"];
  assert(layer2_0_bn1_weight_d.IsArray());
  auto layer2_0_bn1_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn1_weight[i0] = (layer2_0_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_bn1_bias"));
  const Value& layer2_0_bn1_bias_d = document["layer2_0_bn1_bias"];
  assert(layer2_0_bn1_bias_d.IsArray());
  auto layer2_0_bn1_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn1_bias[i0] = (layer2_0_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_bn1_running_mean"));
  const Value& layer2_0_bn1_running_mean_d = document["layer2_0_bn1_running_mean"];
  assert(layer2_0_bn1_running_mean_d.IsArray());
  auto layer2_0_bn1_running_mean = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn1_running_mean[i0] = (layer2_0_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_bn1_running_var"));
  const Value& layer2_0_bn1_running_var_d = document["layer2_0_bn1_running_var"];
  assert(layer2_0_bn1_running_var_d.IsArray());
  auto layer2_0_bn1_running_var = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn1_running_var[i0] = (layer2_0_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_conv2_weight"));
  const Value& layer2_0_conv2_weight_d = document["layer2_0_conv2_weight"];
  assert(layer2_0_conv2_weight_d.IsArray());
  auto layer2_0_conv2_weight = new uint8_t[32][32][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer2_0_conv2_weight[i0][i1][i2][i3] = (layer2_0_conv2_weight_d[i3 + i2*3 + i1*9 + i0*288].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer2_0_bn2_weight"));
  const Value& layer2_0_bn2_weight_d = document["layer2_0_bn2_weight"];
  assert(layer2_0_bn2_weight_d.IsArray());
  auto layer2_0_bn2_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn2_weight[i0] = (layer2_0_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_bn2_bias"));
  const Value& layer2_0_bn2_bias_d = document["layer2_0_bn2_bias"];
  assert(layer2_0_bn2_bias_d.IsArray());
  auto layer2_0_bn2_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn2_bias[i0] = (layer2_0_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_bn2_running_mean"));
  const Value& layer2_0_bn2_running_mean_d = document["layer2_0_bn2_running_mean"];
  assert(layer2_0_bn2_running_mean_d.IsArray());
  auto layer2_0_bn2_running_mean = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn2_running_mean[i0] = (layer2_0_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_0_bn2_running_var"));
  const Value& layer2_0_bn2_running_var_d = document["layer2_0_bn2_running_var"];
  assert(layer2_0_bn2_running_var_d.IsArray());
  auto layer2_0_bn2_running_var = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_0_bn2_running_var[i0] = (layer2_0_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_rprelu1_shift_x_bias"));
  const Value& layer2_1_rprelu1_shift_x_bias_d = document["layer2_1_rprelu1_shift_x_bias"];
  assert(layer2_1_rprelu1_shift_x_bias_d.IsArray());
  auto layer2_1_rprelu1_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_rprelu1_shift_x_bias[i0] = (layer2_1_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_rprelu1_shift_y_bias"));
  const Value& layer2_1_rprelu1_shift_y_bias_d = document["layer2_1_rprelu1_shift_y_bias"];
  assert(layer2_1_rprelu1_shift_y_bias_d.IsArray());
  auto layer2_1_rprelu1_shift_y_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_rprelu1_shift_y_bias[i0] = (layer2_1_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_rprelu1_prelu_weight"));
  const Value& layer2_1_rprelu1_prelu_weight_d = document["layer2_1_rprelu1_prelu_weight"];
  assert(layer2_1_rprelu1_prelu_weight_d.IsArray());
  auto layer2_1_rprelu1_prelu_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_rprelu1_prelu_weight[i0] = (layer2_1_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_rprelu2_shift_x_bias"));
  const Value& layer2_1_rprelu2_shift_x_bias_d = document["layer2_1_rprelu2_shift_x_bias"];
  assert(layer2_1_rprelu2_shift_x_bias_d.IsArray());
  auto layer2_1_rprelu2_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_rprelu2_shift_x_bias[i0] = (layer2_1_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_rprelu2_shift_y_bias"));
  const Value& layer2_1_rprelu2_shift_y_bias_d = document["layer2_1_rprelu2_shift_y_bias"];
  assert(layer2_1_rprelu2_shift_y_bias_d.IsArray());
  auto layer2_1_rprelu2_shift_y_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_rprelu2_shift_y_bias[i0] = (layer2_1_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_rprelu2_prelu_weight"));
  const Value& layer2_1_rprelu2_prelu_weight_d = document["layer2_1_rprelu2_prelu_weight"];
  assert(layer2_1_rprelu2_prelu_weight_d.IsArray());
  auto layer2_1_rprelu2_prelu_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_rprelu2_prelu_weight[i0] = (layer2_1_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_binarize1_shift_x_bias"));
  const Value& layer2_1_binarize1_shift_x_bias_d = document["layer2_1_binarize1_shift_x_bias"];
  assert(layer2_1_binarize1_shift_x_bias_d.IsArray());
  auto layer2_1_binarize1_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_binarize1_shift_x_bias[i0] = (layer2_1_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_binarize2_shift_x_bias"));
  const Value& layer2_1_binarize2_shift_x_bias_d = document["layer2_1_binarize2_shift_x_bias"];
  assert(layer2_1_binarize2_shift_x_bias_d.IsArray());
  auto layer2_1_binarize2_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_binarize2_shift_x_bias[i0] = (layer2_1_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_conv1_weight"));
  const Value& layer2_1_conv1_weight_d = document["layer2_1_conv1_weight"];
  assert(layer2_1_conv1_weight_d.IsArray());
  auto layer2_1_conv1_weight = new uint8_t[32][32][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer2_1_conv1_weight[i0][i1][i2][i3] = (layer2_1_conv1_weight_d[i3 + i2*3 + i1*9 + i0*288].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer2_1_bn1_weight"));
  const Value& layer2_1_bn1_weight_d = document["layer2_1_bn1_weight"];
  assert(layer2_1_bn1_weight_d.IsArray());
  auto layer2_1_bn1_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn1_weight[i0] = (layer2_1_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_bn1_bias"));
  const Value& layer2_1_bn1_bias_d = document["layer2_1_bn1_bias"];
  assert(layer2_1_bn1_bias_d.IsArray());
  auto layer2_1_bn1_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn1_bias[i0] = (layer2_1_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_bn1_running_mean"));
  const Value& layer2_1_bn1_running_mean_d = document["layer2_1_bn1_running_mean"];
  assert(layer2_1_bn1_running_mean_d.IsArray());
  auto layer2_1_bn1_running_mean = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn1_running_mean[i0] = (layer2_1_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_bn1_running_var"));
  const Value& layer2_1_bn1_running_var_d = document["layer2_1_bn1_running_var"];
  assert(layer2_1_bn1_running_var_d.IsArray());
  auto layer2_1_bn1_running_var = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn1_running_var[i0] = (layer2_1_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_conv2_weight"));
  const Value& layer2_1_conv2_weight_d = document["layer2_1_conv2_weight"];
  assert(layer2_1_conv2_weight_d.IsArray());
  auto layer2_1_conv2_weight = new uint8_t[32][32][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer2_1_conv2_weight[i0][i1][i2][i3] = (layer2_1_conv2_weight_d[i3 + i2*3 + i1*9 + i0*288].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer2_1_bn2_weight"));
  const Value& layer2_1_bn2_weight_d = document["layer2_1_bn2_weight"];
  assert(layer2_1_bn2_weight_d.IsArray());
  auto layer2_1_bn2_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn2_weight[i0] = (layer2_1_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_bn2_bias"));
  const Value& layer2_1_bn2_bias_d = document["layer2_1_bn2_bias"];
  assert(layer2_1_bn2_bias_d.IsArray());
  auto layer2_1_bn2_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn2_bias[i0] = (layer2_1_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_bn2_running_mean"));
  const Value& layer2_1_bn2_running_mean_d = document["layer2_1_bn2_running_mean"];
  assert(layer2_1_bn2_running_mean_d.IsArray());
  auto layer2_1_bn2_running_mean = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn2_running_mean[i0] = (layer2_1_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_1_bn2_running_var"));
  const Value& layer2_1_bn2_running_var_d = document["layer2_1_bn2_running_var"];
  assert(layer2_1_bn2_running_var_d.IsArray());
  auto layer2_1_bn2_running_var = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_1_bn2_running_var[i0] = (layer2_1_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_rprelu1_shift_x_bias"));
  const Value& layer2_2_rprelu1_shift_x_bias_d = document["layer2_2_rprelu1_shift_x_bias"];
  assert(layer2_2_rprelu1_shift_x_bias_d.IsArray());
  auto layer2_2_rprelu1_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_rprelu1_shift_x_bias[i0] = (layer2_2_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_rprelu1_shift_y_bias"));
  const Value& layer2_2_rprelu1_shift_y_bias_d = document["layer2_2_rprelu1_shift_y_bias"];
  assert(layer2_2_rprelu1_shift_y_bias_d.IsArray());
  auto layer2_2_rprelu1_shift_y_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_rprelu1_shift_y_bias[i0] = (layer2_2_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_rprelu1_prelu_weight"));
  const Value& layer2_2_rprelu1_prelu_weight_d = document["layer2_2_rprelu1_prelu_weight"];
  assert(layer2_2_rprelu1_prelu_weight_d.IsArray());
  auto layer2_2_rprelu1_prelu_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_rprelu1_prelu_weight[i0] = (layer2_2_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_rprelu2_shift_x_bias"));
  const Value& layer2_2_rprelu2_shift_x_bias_d = document["layer2_2_rprelu2_shift_x_bias"];
  assert(layer2_2_rprelu2_shift_x_bias_d.IsArray());
  auto layer2_2_rprelu2_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_rprelu2_shift_x_bias[i0] = (layer2_2_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_rprelu2_shift_y_bias"));
  const Value& layer2_2_rprelu2_shift_y_bias_d = document["layer2_2_rprelu2_shift_y_bias"];
  assert(layer2_2_rprelu2_shift_y_bias_d.IsArray());
  auto layer2_2_rprelu2_shift_y_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_rprelu2_shift_y_bias[i0] = (layer2_2_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_rprelu2_prelu_weight"));
  const Value& layer2_2_rprelu2_prelu_weight_d = document["layer2_2_rprelu2_prelu_weight"];
  assert(layer2_2_rprelu2_prelu_weight_d.IsArray());
  auto layer2_2_rprelu2_prelu_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_rprelu2_prelu_weight[i0] = (layer2_2_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_binarize1_shift_x_bias"));
  const Value& layer2_2_binarize1_shift_x_bias_d = document["layer2_2_binarize1_shift_x_bias"];
  assert(layer2_2_binarize1_shift_x_bias_d.IsArray());
  auto layer2_2_binarize1_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_binarize1_shift_x_bias[i0] = (layer2_2_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_binarize2_shift_x_bias"));
  const Value& layer2_2_binarize2_shift_x_bias_d = document["layer2_2_binarize2_shift_x_bias"];
  assert(layer2_2_binarize2_shift_x_bias_d.IsArray());
  auto layer2_2_binarize2_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_binarize2_shift_x_bias[i0] = (layer2_2_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_conv1_weight"));
  const Value& layer2_2_conv1_weight_d = document["layer2_2_conv1_weight"];
  assert(layer2_2_conv1_weight_d.IsArray());
  auto layer2_2_conv1_weight = new uint8_t[32][32][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer2_2_conv1_weight[i0][i1][i2][i3] = (layer2_2_conv1_weight_d[i3 + i2*3 + i1*9 + i0*288].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer2_2_bn1_weight"));
  const Value& layer2_2_bn1_weight_d = document["layer2_2_bn1_weight"];
  assert(layer2_2_bn1_weight_d.IsArray());
  auto layer2_2_bn1_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn1_weight[i0] = (layer2_2_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_bn1_bias"));
  const Value& layer2_2_bn1_bias_d = document["layer2_2_bn1_bias"];
  assert(layer2_2_bn1_bias_d.IsArray());
  auto layer2_2_bn1_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn1_bias[i0] = (layer2_2_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_bn1_running_mean"));
  const Value& layer2_2_bn1_running_mean_d = document["layer2_2_bn1_running_mean"];
  assert(layer2_2_bn1_running_mean_d.IsArray());
  auto layer2_2_bn1_running_mean = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn1_running_mean[i0] = (layer2_2_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_bn1_running_var"));
  const Value& layer2_2_bn1_running_var_d = document["layer2_2_bn1_running_var"];
  assert(layer2_2_bn1_running_var_d.IsArray());
  auto layer2_2_bn1_running_var = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn1_running_var[i0] = (layer2_2_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_conv2_weight"));
  const Value& layer2_2_conv2_weight_d = document["layer2_2_conv2_weight"];
  assert(layer2_2_conv2_weight_d.IsArray());
  auto layer2_2_conv2_weight = new uint8_t[32][32][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer2_2_conv2_weight[i0][i1][i2][i3] = (layer2_2_conv2_weight_d[i3 + i2*3 + i1*9 + i0*288].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer2_2_bn2_weight"));
  const Value& layer2_2_bn2_weight_d = document["layer2_2_bn2_weight"];
  assert(layer2_2_bn2_weight_d.IsArray());
  auto layer2_2_bn2_weight = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn2_weight[i0] = (layer2_2_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_bn2_bias"));
  const Value& layer2_2_bn2_bias_d = document["layer2_2_bn2_bias"];
  assert(layer2_2_bn2_bias_d.IsArray());
  auto layer2_2_bn2_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn2_bias[i0] = (layer2_2_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_bn2_running_mean"));
  const Value& layer2_2_bn2_running_mean_d = document["layer2_2_bn2_running_mean"];
  assert(layer2_2_bn2_running_mean_d.IsArray());
  auto layer2_2_bn2_running_mean = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn2_running_mean[i0] = (layer2_2_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer2_2_bn2_running_var"));
  const Value& layer2_2_bn2_running_var_d = document["layer2_2_bn2_running_var"];
  assert(layer2_2_bn2_running_var_d.IsArray());
  auto layer2_2_bn2_running_var = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer2_2_bn2_running_var[i0] = (layer2_2_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_rprelu1_shift_x_bias"));
  const Value& layer3_0_rprelu1_shift_x_bias_d = document["layer3_0_rprelu1_shift_x_bias"];
  assert(layer3_0_rprelu1_shift_x_bias_d.IsArray());
  auto layer3_0_rprelu1_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_rprelu1_shift_x_bias[i0] = (layer3_0_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_rprelu1_shift_y_bias"));
  const Value& layer3_0_rprelu1_shift_y_bias_d = document["layer3_0_rprelu1_shift_y_bias"];
  assert(layer3_0_rprelu1_shift_y_bias_d.IsArray());
  auto layer3_0_rprelu1_shift_y_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_rprelu1_shift_y_bias[i0] = (layer3_0_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_rprelu1_prelu_weight"));
  const Value& layer3_0_rprelu1_prelu_weight_d = document["layer3_0_rprelu1_prelu_weight"];
  assert(layer3_0_rprelu1_prelu_weight_d.IsArray());
  auto layer3_0_rprelu1_prelu_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_rprelu1_prelu_weight[i0] = (layer3_0_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_rprelu2_shift_x_bias"));
  const Value& layer3_0_rprelu2_shift_x_bias_d = document["layer3_0_rprelu2_shift_x_bias"];
  assert(layer3_0_rprelu2_shift_x_bias_d.IsArray());
  auto layer3_0_rprelu2_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_rprelu2_shift_x_bias[i0] = (layer3_0_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_rprelu2_shift_y_bias"));
  const Value& layer3_0_rprelu2_shift_y_bias_d = document["layer3_0_rprelu2_shift_y_bias"];
  assert(layer3_0_rprelu2_shift_y_bias_d.IsArray());
  auto layer3_0_rprelu2_shift_y_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_rprelu2_shift_y_bias[i0] = (layer3_0_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_rprelu2_prelu_weight"));
  const Value& layer3_0_rprelu2_prelu_weight_d = document["layer3_0_rprelu2_prelu_weight"];
  assert(layer3_0_rprelu2_prelu_weight_d.IsArray());
  auto layer3_0_rprelu2_prelu_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_rprelu2_prelu_weight[i0] = (layer3_0_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_binarize1_shift_x_bias"));
  const Value& layer3_0_binarize1_shift_x_bias_d = document["layer3_0_binarize1_shift_x_bias"];
  assert(layer3_0_binarize1_shift_x_bias_d.IsArray());
  auto layer3_0_binarize1_shift_x_bias = new float[32];
  for (size_t i0 = 0; i0 < 32; i0++) {
    layer3_0_binarize1_shift_x_bias[i0] = (layer3_0_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_binarize2_shift_x_bias"));
  const Value& layer3_0_binarize2_shift_x_bias_d = document["layer3_0_binarize2_shift_x_bias"];
  assert(layer3_0_binarize2_shift_x_bias_d.IsArray());
  auto layer3_0_binarize2_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_binarize2_shift_x_bias[i0] = (layer3_0_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_conv1_weight"));
  const Value& layer3_0_conv1_weight_d = document["layer3_0_conv1_weight"];
  assert(layer3_0_conv1_weight_d.IsArray());
  auto layer3_0_conv1_weight = new uint8_t[64][32][3][3];
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer3_0_conv1_weight[i0][i1][i2][i3] = (layer3_0_conv1_weight_d[i3 + i2*3 + i1*9 + i0*288].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer3_0_bn1_weight"));
  const Value& layer3_0_bn1_weight_d = document["layer3_0_bn1_weight"];
  assert(layer3_0_bn1_weight_d.IsArray());
  auto layer3_0_bn1_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn1_weight[i0] = (layer3_0_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_bn1_bias"));
  const Value& layer3_0_bn1_bias_d = document["layer3_0_bn1_bias"];
  assert(layer3_0_bn1_bias_d.IsArray());
  auto layer3_0_bn1_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn1_bias[i0] = (layer3_0_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_bn1_running_mean"));
  const Value& layer3_0_bn1_running_mean_d = document["layer3_0_bn1_running_mean"];
  assert(layer3_0_bn1_running_mean_d.IsArray());
  auto layer3_0_bn1_running_mean = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn1_running_mean[i0] = (layer3_0_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_bn1_running_var"));
  const Value& layer3_0_bn1_running_var_d = document["layer3_0_bn1_running_var"];
  assert(layer3_0_bn1_running_var_d.IsArray());
  auto layer3_0_bn1_running_var = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn1_running_var[i0] = (layer3_0_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_conv2_weight"));
  const Value& layer3_0_conv2_weight_d = document["layer3_0_conv2_weight"];
  assert(layer3_0_conv2_weight_d.IsArray());
  auto layer3_0_conv2_weight = new uint8_t[64][64][3][3];
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer3_0_conv2_weight[i0][i1][i2][i3] = (layer3_0_conv2_weight_d[i3 + i2*3 + i1*9 + i0*576].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer3_0_bn2_weight"));
  const Value& layer3_0_bn2_weight_d = document["layer3_0_bn2_weight"];
  assert(layer3_0_bn2_weight_d.IsArray());
  auto layer3_0_bn2_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn2_weight[i0] = (layer3_0_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_bn2_bias"));
  const Value& layer3_0_bn2_bias_d = document["layer3_0_bn2_bias"];
  assert(layer3_0_bn2_bias_d.IsArray());
  auto layer3_0_bn2_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn2_bias[i0] = (layer3_0_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_bn2_running_mean"));
  const Value& layer3_0_bn2_running_mean_d = document["layer3_0_bn2_running_mean"];
  assert(layer3_0_bn2_running_mean_d.IsArray());
  auto layer3_0_bn2_running_mean = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn2_running_mean[i0] = (layer3_0_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_0_bn2_running_var"));
  const Value& layer3_0_bn2_running_var_d = document["layer3_0_bn2_running_var"];
  assert(layer3_0_bn2_running_var_d.IsArray());
  auto layer3_0_bn2_running_var = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_0_bn2_running_var[i0] = (layer3_0_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_rprelu1_shift_x_bias"));
  const Value& layer3_1_rprelu1_shift_x_bias_d = document["layer3_1_rprelu1_shift_x_bias"];
  assert(layer3_1_rprelu1_shift_x_bias_d.IsArray());
  auto layer3_1_rprelu1_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_rprelu1_shift_x_bias[i0] = (layer3_1_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_rprelu1_shift_y_bias"));
  const Value& layer3_1_rprelu1_shift_y_bias_d = document["layer3_1_rprelu1_shift_y_bias"];
  assert(layer3_1_rprelu1_shift_y_bias_d.IsArray());
  auto layer3_1_rprelu1_shift_y_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_rprelu1_shift_y_bias[i0] = (layer3_1_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_rprelu1_prelu_weight"));
  const Value& layer3_1_rprelu1_prelu_weight_d = document["layer3_1_rprelu1_prelu_weight"];
  assert(layer3_1_rprelu1_prelu_weight_d.IsArray());
  auto layer3_1_rprelu1_prelu_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_rprelu1_prelu_weight[i0] = (layer3_1_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_rprelu2_shift_x_bias"));
  const Value& layer3_1_rprelu2_shift_x_bias_d = document["layer3_1_rprelu2_shift_x_bias"];
  assert(layer3_1_rprelu2_shift_x_bias_d.IsArray());
  auto layer3_1_rprelu2_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_rprelu2_shift_x_bias[i0] = (layer3_1_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_rprelu2_shift_y_bias"));
  const Value& layer3_1_rprelu2_shift_y_bias_d = document["layer3_1_rprelu2_shift_y_bias"];
  assert(layer3_1_rprelu2_shift_y_bias_d.IsArray());
  auto layer3_1_rprelu2_shift_y_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_rprelu2_shift_y_bias[i0] = (layer3_1_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_rprelu2_prelu_weight"));
  const Value& layer3_1_rprelu2_prelu_weight_d = document["layer3_1_rprelu2_prelu_weight"];
  assert(layer3_1_rprelu2_prelu_weight_d.IsArray());
  auto layer3_1_rprelu2_prelu_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_rprelu2_prelu_weight[i0] = (layer3_1_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_binarize1_shift_x_bias"));
  const Value& layer3_1_binarize1_shift_x_bias_d = document["layer3_1_binarize1_shift_x_bias"];
  assert(layer3_1_binarize1_shift_x_bias_d.IsArray());
  auto layer3_1_binarize1_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_binarize1_shift_x_bias[i0] = (layer3_1_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_binarize2_shift_x_bias"));
  const Value& layer3_1_binarize2_shift_x_bias_d = document["layer3_1_binarize2_shift_x_bias"];
  assert(layer3_1_binarize2_shift_x_bias_d.IsArray());
  auto layer3_1_binarize2_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_binarize2_shift_x_bias[i0] = (layer3_1_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_conv1_weight"));
  const Value& layer3_1_conv1_weight_d = document["layer3_1_conv1_weight"];
  assert(layer3_1_conv1_weight_d.IsArray());
  auto layer3_1_conv1_weight = new uint8_t[64][64][3][3];
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer3_1_conv1_weight[i0][i1][i2][i3] = (layer3_1_conv1_weight_d[i3 + i2*3 + i1*9 + i0*576].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer3_1_bn1_weight"));
  const Value& layer3_1_bn1_weight_d = document["layer3_1_bn1_weight"];
  assert(layer3_1_bn1_weight_d.IsArray());
  auto layer3_1_bn1_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn1_weight[i0] = (layer3_1_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_bn1_bias"));
  const Value& layer3_1_bn1_bias_d = document["layer3_1_bn1_bias"];
  assert(layer3_1_bn1_bias_d.IsArray());
  auto layer3_1_bn1_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn1_bias[i0] = (layer3_1_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_bn1_running_mean"));
  const Value& layer3_1_bn1_running_mean_d = document["layer3_1_bn1_running_mean"];
  assert(layer3_1_bn1_running_mean_d.IsArray());
  auto layer3_1_bn1_running_mean = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn1_running_mean[i0] = (layer3_1_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_bn1_running_var"));
  const Value& layer3_1_bn1_running_var_d = document["layer3_1_bn1_running_var"];
  assert(layer3_1_bn1_running_var_d.IsArray());
  auto layer3_1_bn1_running_var = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn1_running_var[i0] = (layer3_1_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_conv2_weight"));
  const Value& layer3_1_conv2_weight_d = document["layer3_1_conv2_weight"];
  assert(layer3_1_conv2_weight_d.IsArray());
  auto layer3_1_conv2_weight = new uint8_t[64][64][3][3];
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer3_1_conv2_weight[i0][i1][i2][i3] = (layer3_1_conv2_weight_d[i3 + i2*3 + i1*9 + i0*576].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer3_1_bn2_weight"));
  const Value& layer3_1_bn2_weight_d = document["layer3_1_bn2_weight"];
  assert(layer3_1_bn2_weight_d.IsArray());
  auto layer3_1_bn2_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn2_weight[i0] = (layer3_1_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_bn2_bias"));
  const Value& layer3_1_bn2_bias_d = document["layer3_1_bn2_bias"];
  assert(layer3_1_bn2_bias_d.IsArray());
  auto layer3_1_bn2_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn2_bias[i0] = (layer3_1_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_bn2_running_mean"));
  const Value& layer3_1_bn2_running_mean_d = document["layer3_1_bn2_running_mean"];
  assert(layer3_1_bn2_running_mean_d.IsArray());
  auto layer3_1_bn2_running_mean = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn2_running_mean[i0] = (layer3_1_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_1_bn2_running_var"));
  const Value& layer3_1_bn2_running_var_d = document["layer3_1_bn2_running_var"];
  assert(layer3_1_bn2_running_var_d.IsArray());
  auto layer3_1_bn2_running_var = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_1_bn2_running_var[i0] = (layer3_1_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_rprelu1_shift_x_bias"));
  const Value& layer3_2_rprelu1_shift_x_bias_d = document["layer3_2_rprelu1_shift_x_bias"];
  assert(layer3_2_rprelu1_shift_x_bias_d.IsArray());
  auto layer3_2_rprelu1_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_rprelu1_shift_x_bias[i0] = (layer3_2_rprelu1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_rprelu1_shift_y_bias"));
  const Value& layer3_2_rprelu1_shift_y_bias_d = document["layer3_2_rprelu1_shift_y_bias"];
  assert(layer3_2_rprelu1_shift_y_bias_d.IsArray());
  auto layer3_2_rprelu1_shift_y_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_rprelu1_shift_y_bias[i0] = (layer3_2_rprelu1_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_rprelu1_prelu_weight"));
  const Value& layer3_2_rprelu1_prelu_weight_d = document["layer3_2_rprelu1_prelu_weight"];
  assert(layer3_2_rprelu1_prelu_weight_d.IsArray());
  auto layer3_2_rprelu1_prelu_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_rprelu1_prelu_weight[i0] = (layer3_2_rprelu1_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_rprelu2_shift_x_bias"));
  const Value& layer3_2_rprelu2_shift_x_bias_d = document["layer3_2_rprelu2_shift_x_bias"];
  assert(layer3_2_rprelu2_shift_x_bias_d.IsArray());
  auto layer3_2_rprelu2_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_rprelu2_shift_x_bias[i0] = (layer3_2_rprelu2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_rprelu2_shift_y_bias"));
  const Value& layer3_2_rprelu2_shift_y_bias_d = document["layer3_2_rprelu2_shift_y_bias"];
  assert(layer3_2_rprelu2_shift_y_bias_d.IsArray());
  auto layer3_2_rprelu2_shift_y_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_rprelu2_shift_y_bias[i0] = (layer3_2_rprelu2_shift_y_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_rprelu2_prelu_weight"));
  const Value& layer3_2_rprelu2_prelu_weight_d = document["layer3_2_rprelu2_prelu_weight"];
  assert(layer3_2_rprelu2_prelu_weight_d.IsArray());
  auto layer3_2_rprelu2_prelu_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_rprelu2_prelu_weight[i0] = (layer3_2_rprelu2_prelu_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_binarize1_shift_x_bias"));
  const Value& layer3_2_binarize1_shift_x_bias_d = document["layer3_2_binarize1_shift_x_bias"];
  assert(layer3_2_binarize1_shift_x_bias_d.IsArray());
  auto layer3_2_binarize1_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_binarize1_shift_x_bias[i0] = (layer3_2_binarize1_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_binarize2_shift_x_bias"));
  const Value& layer3_2_binarize2_shift_x_bias_d = document["layer3_2_binarize2_shift_x_bias"];
  assert(layer3_2_binarize2_shift_x_bias_d.IsArray());
  auto layer3_2_binarize2_shift_x_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_binarize2_shift_x_bias[i0] = (layer3_2_binarize2_shift_x_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_conv1_weight"));
  const Value& layer3_2_conv1_weight_d = document["layer3_2_conv1_weight"];
  assert(layer3_2_conv1_weight_d.IsArray());
  auto layer3_2_conv1_weight = new uint8_t[64][64][3][3];
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer3_2_conv1_weight[i0][i1][i2][i3] = (layer3_2_conv1_weight_d[i3 + i2*3 + i1*9 + i0*576].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer3_2_bn1_weight"));
  const Value& layer3_2_bn1_weight_d = document["layer3_2_bn1_weight"];
  assert(layer3_2_bn1_weight_d.IsArray());
  auto layer3_2_bn1_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn1_weight[i0] = (layer3_2_bn1_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_bn1_bias"));
  const Value& layer3_2_bn1_bias_d = document["layer3_2_bn1_bias"];
  assert(layer3_2_bn1_bias_d.IsArray());
  auto layer3_2_bn1_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn1_bias[i0] = (layer3_2_bn1_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_bn1_running_mean"));
  const Value& layer3_2_bn1_running_mean_d = document["layer3_2_bn1_running_mean"];
  assert(layer3_2_bn1_running_mean_d.IsArray());
  auto layer3_2_bn1_running_mean = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn1_running_mean[i0] = (layer3_2_bn1_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_bn1_running_var"));
  const Value& layer3_2_bn1_running_var_d = document["layer3_2_bn1_running_var"];
  assert(layer3_2_bn1_running_var_d.IsArray());
  auto layer3_2_bn1_running_var = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn1_running_var[i0] = (layer3_2_bn1_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_conv2_weight"));
  const Value& layer3_2_conv2_weight_d = document["layer3_2_conv2_weight"];
  assert(layer3_2_conv2_weight_d.IsArray());
  auto layer3_2_conv2_weight = new uint8_t[64][64][3][3];
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          layer3_2_conv2_weight[i0][i1][i2][i3] = (layer3_2_conv2_weight_d[i3 + i2*3 + i1*9 + i0*576].GetInt());
        }
      }
    }
  }

  assert(document.HasMember("layer3_2_bn2_weight"));
  const Value& layer3_2_bn2_weight_d = document["layer3_2_bn2_weight"];
  assert(layer3_2_bn2_weight_d.IsArray());
  auto layer3_2_bn2_weight = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn2_weight[i0] = (layer3_2_bn2_weight_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_bn2_bias"));
  const Value& layer3_2_bn2_bias_d = document["layer3_2_bn2_bias"];
  assert(layer3_2_bn2_bias_d.IsArray());
  auto layer3_2_bn2_bias = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn2_bias[i0] = (layer3_2_bn2_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_bn2_running_mean"));
  const Value& layer3_2_bn2_running_mean_d = document["layer3_2_bn2_running_mean"];
  assert(layer3_2_bn2_running_mean_d.IsArray());
  auto layer3_2_bn2_running_mean = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn2_running_mean[i0] = (layer3_2_bn2_running_mean_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("layer3_2_bn2_running_var"));
  const Value& layer3_2_bn2_running_var_d = document["layer3_2_bn2_running_var"];
  assert(layer3_2_bn2_running_var_d.IsArray());
  auto layer3_2_bn2_running_var = new float[64];
  for (size_t i0 = 0; i0 < 64; i0++) {
    layer3_2_bn2_running_var[i0] = (layer3_2_bn2_running_var_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("linear_weight"));
  const Value& linear_weight_d = document["linear_weight"];
  assert(linear_weight_d.IsArray());
  auto linear_weight = new float[10][64];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      linear_weight[i0][i1] = (linear_weight_d[i1 + i0*64].GetFloat()) / 1000.0;
    }
  }

  assert(document.HasMember("linear_bias"));
  const Value& linear_bias_d = document["linear_bias"];
  assert(linear_bias_d.IsArray());
  auto linear_bias = new float[10];
  for (size_t i0 = 0; i0 < 10; i0++) {
    linear_bias[i0] = (linear_bias_d[i0].GetFloat()) / 1000.0;
  }

  assert(document.HasMember("fc"));
  const Value& fc_d = document["fc"];
  assert(fc_d.IsArray());
  auto fc = new float[1][10];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      fc[i0][i1] = (fc_d[i1 + i0*10].GetFloat()) / 1000.0;
    }
  }

  std::cout << "[INFO] Initialize RTE...\n";


  cl_int status;
  cl_uint numDevices = 0;
  cl_uint numPlatforms = 0;
  cl_platform_id* platforms = NULL;
  const cl_uint maxDevices = 4;
  cl_device_id devices[maxDevices];
  cl_event kernel_exec_event;

  std::cout << "Just before global and local worksizes\n";
  // global and local worksize
  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};

  std::cout << "Just before platform and device information\n";
  // get platform and device information 
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  platforms = (cl_platform_id*) acl_aligned_malloc (numPlatforms * sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,
      maxDevices, devices, &numDevices); CHECK(status);

  std::cout << "Just before context and command queue\n";
  // create contex and command queue 
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
  CHECK(status);
  cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 
      CL_QUEUE_PROFILING_ENABLE, &status);
  CHECK(status);

  std::cout << "Just before reading AOCL binary\n";
  // read aocx and create binary
  FILE *fp = fopen(AOCX_FILE, "rb");
  fseek(fp, 0, SEEK_END);
  size_t  binary_length = ftell(fp);

  std::cout << "Just before creating program binary\n";
  // create program from binary 
  const unsigned char *binary;
  binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
  assert(binary && "Malloc failed"); rewind(fp);
  if (fread((void*)binary, binary_length, 1, fp) == 0) {
    printf("Failed to read from the AOCX file (fread).\n");
    return -1;
  }
  fclose(fp);
  cl_program program = clCreateProgramWithBinary(context, 1, devices,
      &binary_length, (const unsigned char **)&binary, &status, NULL);

  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  CHECK(status);
    
  printf("Just before Kernel Call\n");

  // Compute and kernel call from host
  int _top;

  cl_kernel kernel = clCreateKernel(program, "test", &status);
  cl_mem buffer_input_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1*3*32*32, NULL, &status); CHECK(status);
  cl_mem buffer_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16*3*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_binarize1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_rprelu1_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_rprelu1_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_rprelu1_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_binarize2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn2_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn2_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_rprelu2_shift_x_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_rprelu2_prelu_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_rprelu2_shift_y_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*64, NULL, &status); CHECK(status);
  cl_mem buffer_linear_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*10*64, NULL, &status); CHECK(status);
  cl_mem buffer_fc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1*10, NULL, &status); CHECK(status);
  cl_mem buffer_linear_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*10, NULL, &status); CHECK(status);

  // write buffers to device
  status = clEnqueueWriteBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(cl_float)*1*3*32*32, input_image, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_conv1_weight, CL_TRUE, 0, sizeof(cl_float)*16*3*3*3, conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer2_0_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*16*3*3, layer2_0_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_0_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer3_0_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*32*3*3, layer3_0_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_0_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_binarize1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu1_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu1_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu1_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_binarize2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu2_shift_x_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu2_prelu_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu2_shift_y_bias, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_linear_weight, CL_TRUE, 0, sizeof(cl_float)*10*64, linear_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_fc, CL_TRUE, 0, sizeof(cl_float)*1*10, fc, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_linear_bias, CL_TRUE, 0, sizeof(cl_float)*10, linear_bias, 0, NULL, NULL); CHECK(status);

  // set device kernel buffer
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_input_image); CHECK(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&buffer_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&buffer_layer1_0_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&buffer_layer1_0_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&buffer_layer1_0_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&buffer_layer1_0_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&buffer_layer1_0_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&buffer_layer1_0_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&buffer_layer1_0_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&buffer_layer1_0_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&buffer_layer1_0_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&buffer_layer1_0_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 16, sizeof(cl_mem), (void*)&buffer_layer1_0_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 17, sizeof(cl_mem), (void*)&buffer_layer1_0_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 18, sizeof(cl_mem), (void*)&buffer_layer1_0_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 19, sizeof(cl_mem), (void*)&buffer_layer1_0_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 20, sizeof(cl_mem), (void*)&buffer_layer1_0_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 21, sizeof(cl_mem), (void*)&buffer_layer1_0_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 22, sizeof(cl_mem), (void*)&buffer_layer1_0_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 23, sizeof(cl_mem), (void*)&buffer_layer1_0_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 24, sizeof(cl_mem), (void*)&buffer_layer1_1_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 25, sizeof(cl_mem), (void*)&buffer_layer1_1_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 26, sizeof(cl_mem), (void*)&buffer_layer1_1_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 27, sizeof(cl_mem), (void*)&buffer_layer1_1_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 28, sizeof(cl_mem), (void*)&buffer_layer1_1_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 29, sizeof(cl_mem), (void*)&buffer_layer1_1_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 30, sizeof(cl_mem), (void*)&buffer_layer1_1_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 31, sizeof(cl_mem), (void*)&buffer_layer1_1_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 32, sizeof(cl_mem), (void*)&buffer_layer1_1_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 33, sizeof(cl_mem), (void*)&buffer_layer1_1_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 34, sizeof(cl_mem), (void*)&buffer_layer1_1_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 35, sizeof(cl_mem), (void*)&buffer_layer1_1_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 36, sizeof(cl_mem), (void*)&buffer_layer1_1_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 37, sizeof(cl_mem), (void*)&buffer_layer1_1_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 38, sizeof(cl_mem), (void*)&buffer_layer1_1_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 39, sizeof(cl_mem), (void*)&buffer_layer1_1_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 40, sizeof(cl_mem), (void*)&buffer_layer1_1_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 41, sizeof(cl_mem), (void*)&buffer_layer1_1_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 42, sizeof(cl_mem), (void*)&buffer_layer1_2_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 43, sizeof(cl_mem), (void*)&buffer_layer1_2_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 44, sizeof(cl_mem), (void*)&buffer_layer1_2_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 45, sizeof(cl_mem), (void*)&buffer_layer1_2_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 46, sizeof(cl_mem), (void*)&buffer_layer1_2_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 47, sizeof(cl_mem), (void*)&buffer_layer1_2_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 48, sizeof(cl_mem), (void*)&buffer_layer1_2_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 49, sizeof(cl_mem), (void*)&buffer_layer1_2_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 50, sizeof(cl_mem), (void*)&buffer_layer1_2_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 51, sizeof(cl_mem), (void*)&buffer_layer1_2_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 52, sizeof(cl_mem), (void*)&buffer_layer1_2_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 53, sizeof(cl_mem), (void*)&buffer_layer1_2_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 54, sizeof(cl_mem), (void*)&buffer_layer1_2_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 55, sizeof(cl_mem), (void*)&buffer_layer1_2_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 56, sizeof(cl_mem), (void*)&buffer_layer1_2_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 57, sizeof(cl_mem), (void*)&buffer_layer1_2_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 58, sizeof(cl_mem), (void*)&buffer_layer1_2_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 59, sizeof(cl_mem), (void*)&buffer_layer1_2_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 60, sizeof(cl_mem), (void*)&buffer_layer2_0_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 61, sizeof(cl_mem), (void*)&buffer_layer2_0_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 62, sizeof(cl_mem), (void*)&buffer_layer2_0_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 63, sizeof(cl_mem), (void*)&buffer_layer2_0_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 64, sizeof(cl_mem), (void*)&buffer_layer2_0_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 65, sizeof(cl_mem), (void*)&buffer_layer2_0_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 66, sizeof(cl_mem), (void*)&buffer_layer2_0_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 67, sizeof(cl_mem), (void*)&buffer_layer2_0_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 68, sizeof(cl_mem), (void*)&buffer_layer2_0_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 69, sizeof(cl_mem), (void*)&buffer_layer2_0_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 70, sizeof(cl_mem), (void*)&buffer_layer2_0_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 71, sizeof(cl_mem), (void*)&buffer_layer2_0_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 72, sizeof(cl_mem), (void*)&buffer_layer2_0_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 73, sizeof(cl_mem), (void*)&buffer_layer2_0_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 74, sizeof(cl_mem), (void*)&buffer_layer2_0_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 75, sizeof(cl_mem), (void*)&buffer_layer2_0_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 76, sizeof(cl_mem), (void*)&buffer_layer2_0_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 77, sizeof(cl_mem), (void*)&buffer_layer2_0_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 78, sizeof(cl_mem), (void*)&buffer_layer2_1_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 79, sizeof(cl_mem), (void*)&buffer_layer2_1_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 80, sizeof(cl_mem), (void*)&buffer_layer2_1_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 81, sizeof(cl_mem), (void*)&buffer_layer2_1_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 82, sizeof(cl_mem), (void*)&buffer_layer2_1_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 83, sizeof(cl_mem), (void*)&buffer_layer2_1_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 84, sizeof(cl_mem), (void*)&buffer_layer2_1_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 85, sizeof(cl_mem), (void*)&buffer_layer2_1_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 86, sizeof(cl_mem), (void*)&buffer_layer2_1_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 87, sizeof(cl_mem), (void*)&buffer_layer2_1_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 88, sizeof(cl_mem), (void*)&buffer_layer2_1_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 89, sizeof(cl_mem), (void*)&buffer_layer2_1_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 90, sizeof(cl_mem), (void*)&buffer_layer2_1_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 91, sizeof(cl_mem), (void*)&buffer_layer2_1_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 92, sizeof(cl_mem), (void*)&buffer_layer2_1_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 93, sizeof(cl_mem), (void*)&buffer_layer2_1_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 94, sizeof(cl_mem), (void*)&buffer_layer2_1_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 95, sizeof(cl_mem), (void*)&buffer_layer2_1_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 96, sizeof(cl_mem), (void*)&buffer_layer2_2_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 97, sizeof(cl_mem), (void*)&buffer_layer2_2_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 98, sizeof(cl_mem), (void*)&buffer_layer2_2_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 99, sizeof(cl_mem), (void*)&buffer_layer2_2_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 100, sizeof(cl_mem), (void*)&buffer_layer2_2_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 101, sizeof(cl_mem), (void*)&buffer_layer2_2_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 102, sizeof(cl_mem), (void*)&buffer_layer2_2_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 103, sizeof(cl_mem), (void*)&buffer_layer2_2_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 104, sizeof(cl_mem), (void*)&buffer_layer2_2_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 105, sizeof(cl_mem), (void*)&buffer_layer2_2_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 106, sizeof(cl_mem), (void*)&buffer_layer2_2_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 107, sizeof(cl_mem), (void*)&buffer_layer2_2_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 108, sizeof(cl_mem), (void*)&buffer_layer2_2_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 109, sizeof(cl_mem), (void*)&buffer_layer2_2_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 110, sizeof(cl_mem), (void*)&buffer_layer2_2_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 111, sizeof(cl_mem), (void*)&buffer_layer2_2_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 112, sizeof(cl_mem), (void*)&buffer_layer2_2_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 113, sizeof(cl_mem), (void*)&buffer_layer2_2_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 114, sizeof(cl_mem), (void*)&buffer_layer3_0_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 115, sizeof(cl_mem), (void*)&buffer_layer3_0_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 116, sizeof(cl_mem), (void*)&buffer_layer3_0_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 117, sizeof(cl_mem), (void*)&buffer_layer3_0_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 118, sizeof(cl_mem), (void*)&buffer_layer3_0_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 119, sizeof(cl_mem), (void*)&buffer_layer3_0_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 120, sizeof(cl_mem), (void*)&buffer_layer3_0_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 121, sizeof(cl_mem), (void*)&buffer_layer3_0_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 122, sizeof(cl_mem), (void*)&buffer_layer3_0_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 123, sizeof(cl_mem), (void*)&buffer_layer3_0_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 124, sizeof(cl_mem), (void*)&buffer_layer3_0_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 125, sizeof(cl_mem), (void*)&buffer_layer3_0_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 126, sizeof(cl_mem), (void*)&buffer_layer3_0_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 127, sizeof(cl_mem), (void*)&buffer_layer3_0_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 128, sizeof(cl_mem), (void*)&buffer_layer3_0_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 129, sizeof(cl_mem), (void*)&buffer_layer3_0_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 130, sizeof(cl_mem), (void*)&buffer_layer3_0_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 131, sizeof(cl_mem), (void*)&buffer_layer3_0_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 132, sizeof(cl_mem), (void*)&buffer_layer3_1_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 133, sizeof(cl_mem), (void*)&buffer_layer3_1_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 134, sizeof(cl_mem), (void*)&buffer_layer3_1_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 135, sizeof(cl_mem), (void*)&buffer_layer3_1_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 136, sizeof(cl_mem), (void*)&buffer_layer3_1_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 137, sizeof(cl_mem), (void*)&buffer_layer3_1_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 138, sizeof(cl_mem), (void*)&buffer_layer3_1_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 139, sizeof(cl_mem), (void*)&buffer_layer3_1_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 140, sizeof(cl_mem), (void*)&buffer_layer3_1_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 141, sizeof(cl_mem), (void*)&buffer_layer3_1_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 142, sizeof(cl_mem), (void*)&buffer_layer3_1_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 143, sizeof(cl_mem), (void*)&buffer_layer3_1_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 144, sizeof(cl_mem), (void*)&buffer_layer3_1_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 145, sizeof(cl_mem), (void*)&buffer_layer3_1_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 146, sizeof(cl_mem), (void*)&buffer_layer3_1_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 147, sizeof(cl_mem), (void*)&buffer_layer3_1_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 148, sizeof(cl_mem), (void*)&buffer_layer3_1_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 149, sizeof(cl_mem), (void*)&buffer_layer3_1_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 150, sizeof(cl_mem), (void*)&buffer_layer3_2_binarize1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 151, sizeof(cl_mem), (void*)&buffer_layer3_2_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 152, sizeof(cl_mem), (void*)&buffer_layer3_2_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 153, sizeof(cl_mem), (void*)&buffer_layer3_2_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 154, sizeof(cl_mem), (void*)&buffer_layer3_2_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 155, sizeof(cl_mem), (void*)&buffer_layer3_2_bn1_bias); CHECK(status);
  status = clSetKernelArg(kernel, 156, sizeof(cl_mem), (void*)&buffer_layer3_2_rprelu1_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 157, sizeof(cl_mem), (void*)&buffer_layer3_2_rprelu1_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 158, sizeof(cl_mem), (void*)&buffer_layer3_2_rprelu1_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 159, sizeof(cl_mem), (void*)&buffer_layer3_2_binarize2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 160, sizeof(cl_mem), (void*)&buffer_layer3_2_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 161, sizeof(cl_mem), (void*)&buffer_layer3_2_bn2_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 162, sizeof(cl_mem), (void*)&buffer_layer3_2_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 163, sizeof(cl_mem), (void*)&buffer_layer3_2_bn2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 164, sizeof(cl_mem), (void*)&buffer_layer3_2_bn2_bias); CHECK(status);
  status = clSetKernelArg(kernel, 165, sizeof(cl_mem), (void*)&buffer_layer3_2_rprelu2_shift_x_bias); CHECK(status);
  status = clSetKernelArg(kernel, 166, sizeof(cl_mem), (void*)&buffer_layer3_2_rprelu2_prelu_weight); CHECK(status);
  status = clSetKernelArg(kernel, 167, sizeof(cl_mem), (void*)&buffer_layer3_2_rprelu2_shift_y_bias); CHECK(status);
  status = clSetKernelArg(kernel, 168, sizeof(cl_mem), (void*)&buffer_linear_weight); CHECK(status);
  status = clSetKernelArg(kernel, 169, sizeof(cl_mem), (void*)&buffer_fc); CHECK(status);
  status = clSetKernelArg(kernel, 170, sizeof(cl_mem), (void*)&buffer_linear_bias); CHECK(status);
  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_exec_event); CHECK(status);

  // enqueue kernel function
  status = clFlush(cmdQueue); CHECK(status);
  status = clFinish(cmdQueue); CHECK(status);;
  clEnqueueReadBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(cl_float)*1*3*32*32, input_image, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_conv1_weight, CL_TRUE, 0, sizeof(cl_float)*16*3*3*3, conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_0_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_1_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer1_2_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*16, layer2_0_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*16*3*3, layer2_0_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_0_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_0_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_1_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer2_2_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*32, layer3_0_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*32*3*3, layer3_0_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_0_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_0_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_1_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_binarize1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_binarize1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn1_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_rprelu1_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu1_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_rprelu1_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu1_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_rprelu1_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu1_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_binarize2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_binarize2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn2_running_mean, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn2_running_var, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn2_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn2_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_bn2_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_rprelu2_shift_x_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu2_shift_x_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_rprelu2_prelu_weight, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu2_prelu_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_rprelu2_shift_y_bias, CL_TRUE, 0, sizeof(cl_float)*64, layer3_2_rprelu2_shift_y_bias, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_linear_weight, CL_TRUE, 0, sizeof(cl_float)*10*64, linear_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_fc, CL_TRUE, 0, sizeof(cl_float)*1*10, fc, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_linear_bias, CL_TRUE, 0, sizeof(cl_float)*10, linear_bias, 0, NULL, NULL);

  // execution on host 
    
  /*
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  document["input_image"].Clear();
  rapidjson::Value v_input_image(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      for (size_t i2 = 0; i2 < 32; i2++) {
        for (size_t i3 = 0; i3 < 32; i3++) {
          v_input_image.PushBack(rapidjson::Value().SetFloat(input_image[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["input_image"] = v_input_image;
  document["conv1_weight"].Clear();
  rapidjson::Value v_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_conv1_weight.PushBack(rapidjson::Value().SetFloat(conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["conv1_weight"] = v_conv1_weight;
  document["bn1_weight"].Clear();
  rapidjson::Value v_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_bn1_weight.PushBack(rapidjson::Value().SetFloat(bn1_weight[i0]), allocator);
  }
  document["bn1_weight"] = v_bn1_weight;
  document["bn1_bias"].Clear();
  rapidjson::Value v_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_bn1_bias.PushBack(rapidjson::Value().SetFloat(bn1_bias[i0]), allocator);
  }
  document["bn1_bias"] = v_bn1_bias;
  document["bn1_running_mean"].Clear();
  rapidjson::Value v_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(bn1_running_mean[i0]), allocator);
  }
  document["bn1_running_mean"] = v_bn1_running_mean;
  document["bn1_running_var"].Clear();
  rapidjson::Value v_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_bn1_running_var.PushBack(rapidjson::Value().SetFloat(bn1_running_var[i0]), allocator);
  }
  document["bn1_running_var"] = v_bn1_running_var;
  document["layer1_0_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_0_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer1_0_rprelu1_shift_x_bias"] = v_layer1_0_rprelu1_shift_x_bias;
  document["layer1_0_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer1_0_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer1_0_rprelu1_shift_y_bias"] = v_layer1_0_rprelu1_shift_y_bias;
  document["layer1_0_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer1_0_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer1_0_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer1_0_rprelu1_prelu_weight"] = v_layer1_0_rprelu1_prelu_weight;
  document["layer1_0_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_0_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer1_0_rprelu2_shift_x_bias"] = v_layer1_0_rprelu2_shift_x_bias;
  document["layer1_0_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer1_0_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer1_0_rprelu2_shift_y_bias"] = v_layer1_0_rprelu2_shift_y_bias;
  document["layer1_0_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer1_0_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer1_0_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer1_0_rprelu2_prelu_weight"] = v_layer1_0_rprelu2_prelu_weight;
  document["layer1_0_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_0_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer1_0_binarize1_shift_x_bias"] = v_layer1_0_binarize1_shift_x_bias;
  document["layer1_0_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_0_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer1_0_binarize2_shift_x_bias"] = v_layer1_0_binarize2_shift_x_bias;
  document["layer1_0_conv1_weight"].Clear();
  rapidjson::Value v_layer1_0_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer1_0_conv1_weight.PushBack(rapidjson::Value().SetInt(layer1_0_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer1_0_conv1_weight"] = v_layer1_0_conv1_weight;
  document["layer1_0_bn1_weight"].Clear();
  rapidjson::Value v_layer1_0_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer1_0_bn1_weight[i0]), allocator);
  }
  document["layer1_0_bn1_weight"] = v_layer1_0_bn1_weight;
  document["layer1_0_bn1_bias"].Clear();
  rapidjson::Value v_layer1_0_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_bn1_bias[i0]), allocator);
  }
  document["layer1_0_bn1_bias"] = v_layer1_0_bn1_bias;
  document["layer1_0_bn1_running_mean"].Clear();
  rapidjson::Value v_layer1_0_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer1_0_bn1_running_mean[i0]), allocator);
  }
  document["layer1_0_bn1_running_mean"] = v_layer1_0_bn1_running_mean;
  document["layer1_0_bn1_running_var"].Clear();
  rapidjson::Value v_layer1_0_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer1_0_bn1_running_var[i0]), allocator);
  }
  document["layer1_0_bn1_running_var"] = v_layer1_0_bn1_running_var;
  document["layer1_0_conv2_weight"].Clear();
  rapidjson::Value v_layer1_0_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer1_0_conv2_weight.PushBack(rapidjson::Value().SetInt(layer1_0_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer1_0_conv2_weight"] = v_layer1_0_conv2_weight;
  document["layer1_0_bn2_weight"].Clear();
  rapidjson::Value v_layer1_0_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer1_0_bn2_weight[i0]), allocator);
  }
  document["layer1_0_bn2_weight"] = v_layer1_0_bn2_weight;
  document["layer1_0_bn2_bias"].Clear();
  rapidjson::Value v_layer1_0_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer1_0_bn2_bias[i0]), allocator);
  }
  document["layer1_0_bn2_bias"] = v_layer1_0_bn2_bias;
  document["layer1_0_bn2_running_mean"].Clear();
  rapidjson::Value v_layer1_0_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer1_0_bn2_running_mean[i0]), allocator);
  }
  document["layer1_0_bn2_running_mean"] = v_layer1_0_bn2_running_mean;
  document["layer1_0_bn2_running_var"].Clear();
  rapidjson::Value v_layer1_0_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_0_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer1_0_bn2_running_var[i0]), allocator);
  }
  document["layer1_0_bn2_running_var"] = v_layer1_0_bn2_running_var;
  document["layer1_1_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_1_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer1_1_rprelu1_shift_x_bias"] = v_layer1_1_rprelu1_shift_x_bias;
  document["layer1_1_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer1_1_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer1_1_rprelu1_shift_y_bias"] = v_layer1_1_rprelu1_shift_y_bias;
  document["layer1_1_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer1_1_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer1_1_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer1_1_rprelu1_prelu_weight"] = v_layer1_1_rprelu1_prelu_weight;
  document["layer1_1_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_1_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer1_1_rprelu2_shift_x_bias"] = v_layer1_1_rprelu2_shift_x_bias;
  document["layer1_1_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer1_1_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer1_1_rprelu2_shift_y_bias"] = v_layer1_1_rprelu2_shift_y_bias;
  document["layer1_1_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer1_1_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer1_1_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer1_1_rprelu2_prelu_weight"] = v_layer1_1_rprelu2_prelu_weight;
  document["layer1_1_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_1_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer1_1_binarize1_shift_x_bias"] = v_layer1_1_binarize1_shift_x_bias;
  document["layer1_1_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_1_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer1_1_binarize2_shift_x_bias"] = v_layer1_1_binarize2_shift_x_bias;
  document["layer1_1_conv1_weight"].Clear();
  rapidjson::Value v_layer1_1_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer1_1_conv1_weight.PushBack(rapidjson::Value().SetInt(layer1_1_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer1_1_conv1_weight"] = v_layer1_1_conv1_weight;
  document["layer1_1_bn1_weight"].Clear();
  rapidjson::Value v_layer1_1_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer1_1_bn1_weight[i0]), allocator);
  }
  document["layer1_1_bn1_weight"] = v_layer1_1_bn1_weight;
  document["layer1_1_bn1_bias"].Clear();
  rapidjson::Value v_layer1_1_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_bn1_bias[i0]), allocator);
  }
  document["layer1_1_bn1_bias"] = v_layer1_1_bn1_bias;
  document["layer1_1_bn1_running_mean"].Clear();
  rapidjson::Value v_layer1_1_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer1_1_bn1_running_mean[i0]), allocator);
  }
  document["layer1_1_bn1_running_mean"] = v_layer1_1_bn1_running_mean;
  document["layer1_1_bn1_running_var"].Clear();
  rapidjson::Value v_layer1_1_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer1_1_bn1_running_var[i0]), allocator);
  }
  document["layer1_1_bn1_running_var"] = v_layer1_1_bn1_running_var;
  document["layer1_1_conv2_weight"].Clear();
  rapidjson::Value v_layer1_1_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer1_1_conv2_weight.PushBack(rapidjson::Value().SetInt(layer1_1_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer1_1_conv2_weight"] = v_layer1_1_conv2_weight;
  document["layer1_1_bn2_weight"].Clear();
  rapidjson::Value v_layer1_1_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer1_1_bn2_weight[i0]), allocator);
  }
  document["layer1_1_bn2_weight"] = v_layer1_1_bn2_weight;
  document["layer1_1_bn2_bias"].Clear();
  rapidjson::Value v_layer1_1_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer1_1_bn2_bias[i0]), allocator);
  }
  document["layer1_1_bn2_bias"] = v_layer1_1_bn2_bias;
  document["layer1_1_bn2_running_mean"].Clear();
  rapidjson::Value v_layer1_1_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer1_1_bn2_running_mean[i0]), allocator);
  }
  document["layer1_1_bn2_running_mean"] = v_layer1_1_bn2_running_mean;
  document["layer1_1_bn2_running_var"].Clear();
  rapidjson::Value v_layer1_1_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_1_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer1_1_bn2_running_var[i0]), allocator);
  }
  document["layer1_1_bn2_running_var"] = v_layer1_1_bn2_running_var;
  document["layer1_2_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_2_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer1_2_rprelu1_shift_x_bias"] = v_layer1_2_rprelu1_shift_x_bias;
  document["layer1_2_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer1_2_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer1_2_rprelu1_shift_y_bias"] = v_layer1_2_rprelu1_shift_y_bias;
  document["layer1_2_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer1_2_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer1_2_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer1_2_rprelu1_prelu_weight"] = v_layer1_2_rprelu1_prelu_weight;
  document["layer1_2_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_2_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer1_2_rprelu2_shift_x_bias"] = v_layer1_2_rprelu2_shift_x_bias;
  document["layer1_2_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer1_2_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer1_2_rprelu2_shift_y_bias"] = v_layer1_2_rprelu2_shift_y_bias;
  document["layer1_2_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer1_2_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer1_2_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer1_2_rprelu2_prelu_weight"] = v_layer1_2_rprelu2_prelu_weight;
  document["layer1_2_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_2_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer1_2_binarize1_shift_x_bias"] = v_layer1_2_binarize1_shift_x_bias;
  document["layer1_2_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer1_2_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer1_2_binarize2_shift_x_bias"] = v_layer1_2_binarize2_shift_x_bias;
  document["layer1_2_conv1_weight"].Clear();
  rapidjson::Value v_layer1_2_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer1_2_conv1_weight.PushBack(rapidjson::Value().SetInt(layer1_2_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer1_2_conv1_weight"] = v_layer1_2_conv1_weight;
  document["layer1_2_bn1_weight"].Clear();
  rapidjson::Value v_layer1_2_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer1_2_bn1_weight[i0]), allocator);
  }
  document["layer1_2_bn1_weight"] = v_layer1_2_bn1_weight;
  document["layer1_2_bn1_bias"].Clear();
  rapidjson::Value v_layer1_2_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_bn1_bias[i0]), allocator);
  }
  document["layer1_2_bn1_bias"] = v_layer1_2_bn1_bias;
  document["layer1_2_bn1_running_mean"].Clear();
  rapidjson::Value v_layer1_2_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer1_2_bn1_running_mean[i0]), allocator);
  }
  document["layer1_2_bn1_running_mean"] = v_layer1_2_bn1_running_mean;
  document["layer1_2_bn1_running_var"].Clear();
  rapidjson::Value v_layer1_2_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer1_2_bn1_running_var[i0]), allocator);
  }
  document["layer1_2_bn1_running_var"] = v_layer1_2_bn1_running_var;
  document["layer1_2_conv2_weight"].Clear();
  rapidjson::Value v_layer1_2_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer1_2_conv2_weight.PushBack(rapidjson::Value().SetInt(layer1_2_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer1_2_conv2_weight"] = v_layer1_2_conv2_weight;
  document["layer1_2_bn2_weight"].Clear();
  rapidjson::Value v_layer1_2_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer1_2_bn2_weight[i0]), allocator);
  }
  document["layer1_2_bn2_weight"] = v_layer1_2_bn2_weight;
  document["layer1_2_bn2_bias"].Clear();
  rapidjson::Value v_layer1_2_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer1_2_bn2_bias[i0]), allocator);
  }
  document["layer1_2_bn2_bias"] = v_layer1_2_bn2_bias;
  document["layer1_2_bn2_running_mean"].Clear();
  rapidjson::Value v_layer1_2_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer1_2_bn2_running_mean[i0]), allocator);
  }
  document["layer1_2_bn2_running_mean"] = v_layer1_2_bn2_running_mean;
  document["layer1_2_bn2_running_var"].Clear();
  rapidjson::Value v_layer1_2_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer1_2_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer1_2_bn2_running_var[i0]), allocator);
  }
  document["layer1_2_bn2_running_var"] = v_layer1_2_bn2_running_var;
  document["layer2_0_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_0_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer2_0_rprelu1_shift_x_bias"] = v_layer2_0_rprelu1_shift_x_bias;
  document["layer2_0_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer2_0_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer2_0_rprelu1_shift_y_bias"] = v_layer2_0_rprelu1_shift_y_bias;
  document["layer2_0_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer2_0_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer2_0_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer2_0_rprelu1_prelu_weight"] = v_layer2_0_rprelu1_prelu_weight;
  document["layer2_0_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_0_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer2_0_rprelu2_shift_x_bias"] = v_layer2_0_rprelu2_shift_x_bias;
  document["layer2_0_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer2_0_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer2_0_rprelu2_shift_y_bias"] = v_layer2_0_rprelu2_shift_y_bias;
  document["layer2_0_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer2_0_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer2_0_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer2_0_rprelu2_prelu_weight"] = v_layer2_0_rprelu2_prelu_weight;
  document["layer2_0_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_0_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 16; i0++) {
    v_layer2_0_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer2_0_binarize1_shift_x_bias"] = v_layer2_0_binarize1_shift_x_bias;
  document["layer2_0_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_0_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer2_0_binarize2_shift_x_bias"] = v_layer2_0_binarize2_shift_x_bias;
  document["layer2_0_conv1_weight"].Clear();
  rapidjson::Value v_layer2_0_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer2_0_conv1_weight.PushBack(rapidjson::Value().SetInt(layer2_0_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer2_0_conv1_weight"] = v_layer2_0_conv1_weight;
  document["layer2_0_bn1_weight"].Clear();
  rapidjson::Value v_layer2_0_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer2_0_bn1_weight[i0]), allocator);
  }
  document["layer2_0_bn1_weight"] = v_layer2_0_bn1_weight;
  document["layer2_0_bn1_bias"].Clear();
  rapidjson::Value v_layer2_0_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_bn1_bias[i0]), allocator);
  }
  document["layer2_0_bn1_bias"] = v_layer2_0_bn1_bias;
  document["layer2_0_bn1_running_mean"].Clear();
  rapidjson::Value v_layer2_0_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer2_0_bn1_running_mean[i0]), allocator);
  }
  document["layer2_0_bn1_running_mean"] = v_layer2_0_bn1_running_mean;
  document["layer2_0_bn1_running_var"].Clear();
  rapidjson::Value v_layer2_0_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer2_0_bn1_running_var[i0]), allocator);
  }
  document["layer2_0_bn1_running_var"] = v_layer2_0_bn1_running_var;
  document["layer2_0_conv2_weight"].Clear();
  rapidjson::Value v_layer2_0_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer2_0_conv2_weight.PushBack(rapidjson::Value().SetInt(layer2_0_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer2_0_conv2_weight"] = v_layer2_0_conv2_weight;
  document["layer2_0_bn2_weight"].Clear();
  rapidjson::Value v_layer2_0_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer2_0_bn2_weight[i0]), allocator);
  }
  document["layer2_0_bn2_weight"] = v_layer2_0_bn2_weight;
  document["layer2_0_bn2_bias"].Clear();
  rapidjson::Value v_layer2_0_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer2_0_bn2_bias[i0]), allocator);
  }
  document["layer2_0_bn2_bias"] = v_layer2_0_bn2_bias;
  document["layer2_0_bn2_running_mean"].Clear();
  rapidjson::Value v_layer2_0_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer2_0_bn2_running_mean[i0]), allocator);
  }
  document["layer2_0_bn2_running_mean"] = v_layer2_0_bn2_running_mean;
  document["layer2_0_bn2_running_var"].Clear();
  rapidjson::Value v_layer2_0_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_0_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer2_0_bn2_running_var[i0]), allocator);
  }
  document["layer2_0_bn2_running_var"] = v_layer2_0_bn2_running_var;
  document["layer2_1_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_1_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer2_1_rprelu1_shift_x_bias"] = v_layer2_1_rprelu1_shift_x_bias;
  document["layer2_1_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer2_1_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer2_1_rprelu1_shift_y_bias"] = v_layer2_1_rprelu1_shift_y_bias;
  document["layer2_1_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer2_1_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer2_1_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer2_1_rprelu1_prelu_weight"] = v_layer2_1_rprelu1_prelu_weight;
  document["layer2_1_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_1_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer2_1_rprelu2_shift_x_bias"] = v_layer2_1_rprelu2_shift_x_bias;
  document["layer2_1_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer2_1_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer2_1_rprelu2_shift_y_bias"] = v_layer2_1_rprelu2_shift_y_bias;
  document["layer2_1_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer2_1_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer2_1_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer2_1_rprelu2_prelu_weight"] = v_layer2_1_rprelu2_prelu_weight;
  document["layer2_1_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_1_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer2_1_binarize1_shift_x_bias"] = v_layer2_1_binarize1_shift_x_bias;
  document["layer2_1_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_1_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer2_1_binarize2_shift_x_bias"] = v_layer2_1_binarize2_shift_x_bias;
  document["layer2_1_conv1_weight"].Clear();
  rapidjson::Value v_layer2_1_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer2_1_conv1_weight.PushBack(rapidjson::Value().SetInt(layer2_1_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer2_1_conv1_weight"] = v_layer2_1_conv1_weight;
  document["layer2_1_bn1_weight"].Clear();
  rapidjson::Value v_layer2_1_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer2_1_bn1_weight[i0]), allocator);
  }
  document["layer2_1_bn1_weight"] = v_layer2_1_bn1_weight;
  document["layer2_1_bn1_bias"].Clear();
  rapidjson::Value v_layer2_1_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_bn1_bias[i0]), allocator);
  }
  document["layer2_1_bn1_bias"] = v_layer2_1_bn1_bias;
  document["layer2_1_bn1_running_mean"].Clear();
  rapidjson::Value v_layer2_1_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer2_1_bn1_running_mean[i0]), allocator);
  }
  document["layer2_1_bn1_running_mean"] = v_layer2_1_bn1_running_mean;
  document["layer2_1_bn1_running_var"].Clear();
  rapidjson::Value v_layer2_1_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer2_1_bn1_running_var[i0]), allocator);
  }
  document["layer2_1_bn1_running_var"] = v_layer2_1_bn1_running_var;
  document["layer2_1_conv2_weight"].Clear();
  rapidjson::Value v_layer2_1_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer2_1_conv2_weight.PushBack(rapidjson::Value().SetInt(layer2_1_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer2_1_conv2_weight"] = v_layer2_1_conv2_weight;
  document["layer2_1_bn2_weight"].Clear();
  rapidjson::Value v_layer2_1_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer2_1_bn2_weight[i0]), allocator);
  }
  document["layer2_1_bn2_weight"] = v_layer2_1_bn2_weight;
  document["layer2_1_bn2_bias"].Clear();
  rapidjson::Value v_layer2_1_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer2_1_bn2_bias[i0]), allocator);
  }
  document["layer2_1_bn2_bias"] = v_layer2_1_bn2_bias;
  document["layer2_1_bn2_running_mean"].Clear();
  rapidjson::Value v_layer2_1_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer2_1_bn2_running_mean[i0]), allocator);
  }
  document["layer2_1_bn2_running_mean"] = v_layer2_1_bn2_running_mean;
  document["layer2_1_bn2_running_var"].Clear();
  rapidjson::Value v_layer2_1_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_1_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer2_1_bn2_running_var[i0]), allocator);
  }
  document["layer2_1_bn2_running_var"] = v_layer2_1_bn2_running_var;
  document["layer2_2_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_2_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer2_2_rprelu1_shift_x_bias"] = v_layer2_2_rprelu1_shift_x_bias;
  document["layer2_2_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer2_2_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer2_2_rprelu1_shift_y_bias"] = v_layer2_2_rprelu1_shift_y_bias;
  document["layer2_2_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer2_2_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer2_2_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer2_2_rprelu1_prelu_weight"] = v_layer2_2_rprelu1_prelu_weight;
  document["layer2_2_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_2_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer2_2_rprelu2_shift_x_bias"] = v_layer2_2_rprelu2_shift_x_bias;
  document["layer2_2_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer2_2_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer2_2_rprelu2_shift_y_bias"] = v_layer2_2_rprelu2_shift_y_bias;
  document["layer2_2_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer2_2_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer2_2_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer2_2_rprelu2_prelu_weight"] = v_layer2_2_rprelu2_prelu_weight;
  document["layer2_2_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_2_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer2_2_binarize1_shift_x_bias"] = v_layer2_2_binarize1_shift_x_bias;
  document["layer2_2_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer2_2_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer2_2_binarize2_shift_x_bias"] = v_layer2_2_binarize2_shift_x_bias;
  document["layer2_2_conv1_weight"].Clear();
  rapidjson::Value v_layer2_2_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer2_2_conv1_weight.PushBack(rapidjson::Value().SetInt(layer2_2_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer2_2_conv1_weight"] = v_layer2_2_conv1_weight;
  document["layer2_2_bn1_weight"].Clear();
  rapidjson::Value v_layer2_2_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer2_2_bn1_weight[i0]), allocator);
  }
  document["layer2_2_bn1_weight"] = v_layer2_2_bn1_weight;
  document["layer2_2_bn1_bias"].Clear();
  rapidjson::Value v_layer2_2_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_bn1_bias[i0]), allocator);
  }
  document["layer2_2_bn1_bias"] = v_layer2_2_bn1_bias;
  document["layer2_2_bn1_running_mean"].Clear();
  rapidjson::Value v_layer2_2_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer2_2_bn1_running_mean[i0]), allocator);
  }
  document["layer2_2_bn1_running_mean"] = v_layer2_2_bn1_running_mean;
  document["layer2_2_bn1_running_var"].Clear();
  rapidjson::Value v_layer2_2_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer2_2_bn1_running_var[i0]), allocator);
  }
  document["layer2_2_bn1_running_var"] = v_layer2_2_bn1_running_var;
  document["layer2_2_conv2_weight"].Clear();
  rapidjson::Value v_layer2_2_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer2_2_conv2_weight.PushBack(rapidjson::Value().SetInt(layer2_2_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer2_2_conv2_weight"] = v_layer2_2_conv2_weight;
  document["layer2_2_bn2_weight"].Clear();
  rapidjson::Value v_layer2_2_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer2_2_bn2_weight[i0]), allocator);
  }
  document["layer2_2_bn2_weight"] = v_layer2_2_bn2_weight;
  document["layer2_2_bn2_bias"].Clear();
  rapidjson::Value v_layer2_2_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer2_2_bn2_bias[i0]), allocator);
  }
  document["layer2_2_bn2_bias"] = v_layer2_2_bn2_bias;
  document["layer2_2_bn2_running_mean"].Clear();
  rapidjson::Value v_layer2_2_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer2_2_bn2_running_mean[i0]), allocator);
  }
  document["layer2_2_bn2_running_mean"] = v_layer2_2_bn2_running_mean;
  document["layer2_2_bn2_running_var"].Clear();
  rapidjson::Value v_layer2_2_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer2_2_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer2_2_bn2_running_var[i0]), allocator);
  }
  document["layer2_2_bn2_running_var"] = v_layer2_2_bn2_running_var;
  document["layer3_0_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_0_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer3_0_rprelu1_shift_x_bias"] = v_layer3_0_rprelu1_shift_x_bias;
  document["layer3_0_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer3_0_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer3_0_rprelu1_shift_y_bias"] = v_layer3_0_rprelu1_shift_y_bias;
  document["layer3_0_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer3_0_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer3_0_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer3_0_rprelu1_prelu_weight"] = v_layer3_0_rprelu1_prelu_weight;
  document["layer3_0_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_0_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer3_0_rprelu2_shift_x_bias"] = v_layer3_0_rprelu2_shift_x_bias;
  document["layer3_0_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer3_0_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer3_0_rprelu2_shift_y_bias"] = v_layer3_0_rprelu2_shift_y_bias;
  document["layer3_0_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer3_0_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer3_0_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer3_0_rprelu2_prelu_weight"] = v_layer3_0_rprelu2_prelu_weight;
  document["layer3_0_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_0_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 32; i0++) {
    v_layer3_0_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer3_0_binarize1_shift_x_bias"] = v_layer3_0_binarize1_shift_x_bias;
  document["layer3_0_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_0_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer3_0_binarize2_shift_x_bias"] = v_layer3_0_binarize2_shift_x_bias;
  document["layer3_0_conv1_weight"].Clear();
  rapidjson::Value v_layer3_0_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer3_0_conv1_weight.PushBack(rapidjson::Value().SetInt(layer3_0_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer3_0_conv1_weight"] = v_layer3_0_conv1_weight;
  document["layer3_0_bn1_weight"].Clear();
  rapidjson::Value v_layer3_0_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer3_0_bn1_weight[i0]), allocator);
  }
  document["layer3_0_bn1_weight"] = v_layer3_0_bn1_weight;
  document["layer3_0_bn1_bias"].Clear();
  rapidjson::Value v_layer3_0_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_bn1_bias[i0]), allocator);
  }
  document["layer3_0_bn1_bias"] = v_layer3_0_bn1_bias;
  document["layer3_0_bn1_running_mean"].Clear();
  rapidjson::Value v_layer3_0_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer3_0_bn1_running_mean[i0]), allocator);
  }
  document["layer3_0_bn1_running_mean"] = v_layer3_0_bn1_running_mean;
  document["layer3_0_bn1_running_var"].Clear();
  rapidjson::Value v_layer3_0_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer3_0_bn1_running_var[i0]), allocator);
  }
  document["layer3_0_bn1_running_var"] = v_layer3_0_bn1_running_var;
  document["layer3_0_conv2_weight"].Clear();
  rapidjson::Value v_layer3_0_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer3_0_conv2_weight.PushBack(rapidjson::Value().SetInt(layer3_0_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer3_0_conv2_weight"] = v_layer3_0_conv2_weight;
  document["layer3_0_bn2_weight"].Clear();
  rapidjson::Value v_layer3_0_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer3_0_bn2_weight[i0]), allocator);
  }
  document["layer3_0_bn2_weight"] = v_layer3_0_bn2_weight;
  document["layer3_0_bn2_bias"].Clear();
  rapidjson::Value v_layer3_0_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer3_0_bn2_bias[i0]), allocator);
  }
  document["layer3_0_bn2_bias"] = v_layer3_0_bn2_bias;
  document["layer3_0_bn2_running_mean"].Clear();
  rapidjson::Value v_layer3_0_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer3_0_bn2_running_mean[i0]), allocator);
  }
  document["layer3_0_bn2_running_mean"] = v_layer3_0_bn2_running_mean;
  document["layer3_0_bn2_running_var"].Clear();
  rapidjson::Value v_layer3_0_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_0_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer3_0_bn2_running_var[i0]), allocator);
  }
  document["layer3_0_bn2_running_var"] = v_layer3_0_bn2_running_var;
  document["layer3_1_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_1_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer3_1_rprelu1_shift_x_bias"] = v_layer3_1_rprelu1_shift_x_bias;
  document["layer3_1_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer3_1_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer3_1_rprelu1_shift_y_bias"] = v_layer3_1_rprelu1_shift_y_bias;
  document["layer3_1_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer3_1_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer3_1_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer3_1_rprelu1_prelu_weight"] = v_layer3_1_rprelu1_prelu_weight;
  document["layer3_1_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_1_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer3_1_rprelu2_shift_x_bias"] = v_layer3_1_rprelu2_shift_x_bias;
  document["layer3_1_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer3_1_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer3_1_rprelu2_shift_y_bias"] = v_layer3_1_rprelu2_shift_y_bias;
  document["layer3_1_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer3_1_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer3_1_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer3_1_rprelu2_prelu_weight"] = v_layer3_1_rprelu2_prelu_weight;
  document["layer3_1_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_1_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer3_1_binarize1_shift_x_bias"] = v_layer3_1_binarize1_shift_x_bias;
  document["layer3_1_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_1_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer3_1_binarize2_shift_x_bias"] = v_layer3_1_binarize2_shift_x_bias;
  document["layer3_1_conv1_weight"].Clear();
  rapidjson::Value v_layer3_1_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer3_1_conv1_weight.PushBack(rapidjson::Value().SetInt(layer3_1_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer3_1_conv1_weight"] = v_layer3_1_conv1_weight;
  document["layer3_1_bn1_weight"].Clear();
  rapidjson::Value v_layer3_1_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer3_1_bn1_weight[i0]), allocator);
  }
  document["layer3_1_bn1_weight"] = v_layer3_1_bn1_weight;
  document["layer3_1_bn1_bias"].Clear();
  rapidjson::Value v_layer3_1_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_bn1_bias[i0]), allocator);
  }
  document["layer3_1_bn1_bias"] = v_layer3_1_bn1_bias;
  document["layer3_1_bn1_running_mean"].Clear();
  rapidjson::Value v_layer3_1_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer3_1_bn1_running_mean[i0]), allocator);
  }
  document["layer3_1_bn1_running_mean"] = v_layer3_1_bn1_running_mean;
  document["layer3_1_bn1_running_var"].Clear();
  rapidjson::Value v_layer3_1_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer3_1_bn1_running_var[i0]), allocator);
  }
  document["layer3_1_bn1_running_var"] = v_layer3_1_bn1_running_var;
  document["layer3_1_conv2_weight"].Clear();
  rapidjson::Value v_layer3_1_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer3_1_conv2_weight.PushBack(rapidjson::Value().SetInt(layer3_1_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer3_1_conv2_weight"] = v_layer3_1_conv2_weight;
  document["layer3_1_bn2_weight"].Clear();
  rapidjson::Value v_layer3_1_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer3_1_bn2_weight[i0]), allocator);
  }
  document["layer3_1_bn2_weight"] = v_layer3_1_bn2_weight;
  document["layer3_1_bn2_bias"].Clear();
  rapidjson::Value v_layer3_1_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer3_1_bn2_bias[i0]), allocator);
  }
  document["layer3_1_bn2_bias"] = v_layer3_1_bn2_bias;
  document["layer3_1_bn2_running_mean"].Clear();
  rapidjson::Value v_layer3_1_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer3_1_bn2_running_mean[i0]), allocator);
  }
  document["layer3_1_bn2_running_mean"] = v_layer3_1_bn2_running_mean;
  document["layer3_1_bn2_running_var"].Clear();
  rapidjson::Value v_layer3_1_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_1_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer3_1_bn2_running_var[i0]), allocator);
  }
  document["layer3_1_bn2_running_var"] = v_layer3_1_bn2_running_var;
  document["layer3_2_rprelu1_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_2_rprelu1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_rprelu1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_rprelu1_shift_x_bias[i0]), allocator);
  }
  document["layer3_2_rprelu1_shift_x_bias"] = v_layer3_2_rprelu1_shift_x_bias;
  document["layer3_2_rprelu1_shift_y_bias"].Clear();
  rapidjson::Value v_layer3_2_rprelu1_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_rprelu1_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_rprelu1_shift_y_bias[i0]), allocator);
  }
  document["layer3_2_rprelu1_shift_y_bias"] = v_layer3_2_rprelu1_shift_y_bias;
  document["layer3_2_rprelu1_prelu_weight"].Clear();
  rapidjson::Value v_layer3_2_rprelu1_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_rprelu1_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer3_2_rprelu1_prelu_weight[i0]), allocator);
  }
  document["layer3_2_rprelu1_prelu_weight"] = v_layer3_2_rprelu1_prelu_weight;
  document["layer3_2_rprelu2_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_2_rprelu2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_rprelu2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_rprelu2_shift_x_bias[i0]), allocator);
  }
  document["layer3_2_rprelu2_shift_x_bias"] = v_layer3_2_rprelu2_shift_x_bias;
  document["layer3_2_rprelu2_shift_y_bias"].Clear();
  rapidjson::Value v_layer3_2_rprelu2_shift_y_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_rprelu2_shift_y_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_rprelu2_shift_y_bias[i0]), allocator);
  }
  document["layer3_2_rprelu2_shift_y_bias"] = v_layer3_2_rprelu2_shift_y_bias;
  document["layer3_2_rprelu2_prelu_weight"].Clear();
  rapidjson::Value v_layer3_2_rprelu2_prelu_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_rprelu2_prelu_weight.PushBack(rapidjson::Value().SetFloat(layer3_2_rprelu2_prelu_weight[i0]), allocator);
  }
  document["layer3_2_rprelu2_prelu_weight"] = v_layer3_2_rprelu2_prelu_weight;
  document["layer3_2_binarize1_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_2_binarize1_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_binarize1_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_binarize1_shift_x_bias[i0]), allocator);
  }
  document["layer3_2_binarize1_shift_x_bias"] = v_layer3_2_binarize1_shift_x_bias;
  document["layer3_2_binarize2_shift_x_bias"].Clear();
  rapidjson::Value v_layer3_2_binarize2_shift_x_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_binarize2_shift_x_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_binarize2_shift_x_bias[i0]), allocator);
  }
  document["layer3_2_binarize2_shift_x_bias"] = v_layer3_2_binarize2_shift_x_bias;
  document["layer3_2_conv1_weight"].Clear();
  rapidjson::Value v_layer3_2_conv1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer3_2_conv1_weight.PushBack(rapidjson::Value().SetInt(layer3_2_conv1_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer3_2_conv1_weight"] = v_layer3_2_conv1_weight;
  document["layer3_2_bn1_weight"].Clear();
  rapidjson::Value v_layer3_2_bn1_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn1_weight.PushBack(rapidjson::Value().SetFloat(layer3_2_bn1_weight[i0]), allocator);
  }
  document["layer3_2_bn1_weight"] = v_layer3_2_bn1_weight;
  document["layer3_2_bn1_bias"].Clear();
  rapidjson::Value v_layer3_2_bn1_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn1_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_bn1_bias[i0]), allocator);
  }
  document["layer3_2_bn1_bias"] = v_layer3_2_bn1_bias;
  document["layer3_2_bn1_running_mean"].Clear();
  rapidjson::Value v_layer3_2_bn1_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn1_running_mean.PushBack(rapidjson::Value().SetFloat(layer3_2_bn1_running_mean[i0]), allocator);
  }
  document["layer3_2_bn1_running_mean"] = v_layer3_2_bn1_running_mean;
  document["layer3_2_bn1_running_var"].Clear();
  rapidjson::Value v_layer3_2_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn1_running_var.PushBack(rapidjson::Value().SetFloat(layer3_2_bn1_running_var[i0]), allocator);
  }
  document["layer3_2_bn1_running_var"] = v_layer3_2_bn1_running_var;
  document["layer3_2_conv2_weight"].Clear();
  rapidjson::Value v_layer3_2_conv2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          v_layer3_2_conv2_weight.PushBack(rapidjson::Value().SetInt(layer3_2_conv2_weight[i0][i1][i2][i3]), allocator);
        }
      }
    }
  }
  document["layer3_2_conv2_weight"] = v_layer3_2_conv2_weight;
  document["layer3_2_bn2_weight"].Clear();
  rapidjson::Value v_layer3_2_bn2_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn2_weight.PushBack(rapidjson::Value().SetFloat(layer3_2_bn2_weight[i0]), allocator);
  }
  document["layer3_2_bn2_weight"] = v_layer3_2_bn2_weight;
  document["layer3_2_bn2_bias"].Clear();
  rapidjson::Value v_layer3_2_bn2_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn2_bias.PushBack(rapidjson::Value().SetFloat(layer3_2_bn2_bias[i0]), allocator);
  }
  document["layer3_2_bn2_bias"] = v_layer3_2_bn2_bias;
  document["layer3_2_bn2_running_mean"].Clear();
  rapidjson::Value v_layer3_2_bn2_running_mean(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn2_running_mean.PushBack(rapidjson::Value().SetFloat(layer3_2_bn2_running_mean[i0]), allocator);
  }
  document["layer3_2_bn2_running_mean"] = v_layer3_2_bn2_running_mean;
  document["layer3_2_bn2_running_var"].Clear();
  rapidjson::Value v_layer3_2_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 64; i0++) {
    v_layer3_2_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer3_2_bn2_running_var[i0]), allocator);
  }
  document["layer3_2_bn2_running_var"] = v_layer3_2_bn2_running_var;
  document["linear_weight"].Clear();
  rapidjson::Value v_linear_weight(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      v_linear_weight.PushBack(rapidjson::Value().SetFloat(linear_weight[i0][i1]), allocator);
    }
  }
  document["linear_weight"] = v_linear_weight;
  document["linear_bias"].Clear();
  rapidjson::Value v_linear_bias(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 10; i0++) {
    v_linear_bias.PushBack(rapidjson::Value().SetFloat(linear_bias[i0]), allocator);
  }
  document["linear_bias"] = v_linear_bias;
  document["fc"].Clear();
  rapidjson::Value v_fc(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      v_fc.PushBack(rapidjson::Value().SetFloat(fc[i0][i1]), allocator);
    }
  }
  document["fc"] = v_fc;

  FILE* fp1 = fopen("inputs.json", "w"); 
 
  char writeBuffer[65536];
  FileWriteStream os(fp1, writeBuffer, sizeof(writeBuffer));
 
  Writer<FileWriteStream> writer(os);
  document.Accept(writer);
  fclose(fp1);

  */

  }
