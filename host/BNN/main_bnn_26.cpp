
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
#define AOCX_FILE "AOCX/bnn_emu.aocx"



#define CHECK(status) 							\
    if (status != CL_SUCCESS)						\
{									\
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
    exit(1);								\
}									\

double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    start_d = (double)1.0e-9 * start;
    end_d   = (double)1.0e-9 * end;

    return 	(double)1.0e-9 * (end - start); // nanoseconds to seconds
}

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
  auto input_image = new int32_t[1][3][32][32];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      for (size_t i2 = 0; i2 < 32; i2++) {
        for (size_t i3 = 0; i3 < 32; i3++) {
          input_image[i0][i1][i2][i3] = (input_image_d[i3 + i2*32 + i1*1024 + i0*3072].GetFloat());
        }
      }
    }
  }

  assert(document.HasMember("conv1_weight"));
  const Value& conv1_weight_d = document["conv1_weight"];
  assert(conv1_weight_d.IsArray());
  auto conv1_weight = new int32_t[16][3][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          conv1_weight[i0][i1][i2][i3] = (conv1_weight_d[i3 + i2*3 + i1*9 + i0*27].GetFloat());
        }
      }
    }
  }

  assert(document.HasMember("bn1_running_var"));
  const Value& bn1_running_var_d = document["bn1_running_var"];
  assert(bn1_running_var_d.IsArray());
  auto bn1_running_var = new int32_t[4][16];
  for (size_t i0 = 0; i0 < 4; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      bn1_running_var[i0][i1] = (bn1_running_var_d[i1 + i0*16].GetFloat());
    }
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

  assert(document.HasMember("layer1_2_bn2_running_var"));
  const Value& layer1_2_bn2_running_var_d = document["layer1_2_bn2_running_var"];
  assert(layer1_2_bn2_running_var_d.IsArray());
  auto layer1_2_bn2_running_var = new int32_t[48][16];
  for (size_t i0 = 0; i0 < 48; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      layer1_2_bn2_running_var[i0][i1] = (layer1_2_bn2_running_var_d[i1 + i0*16].GetFloat());
    }
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

  assert(document.HasMember("layer2_2_bn2_running_var"));
  const Value& layer2_2_bn2_running_var_d = document["layer2_2_bn2_running_var"];
  assert(layer2_2_bn2_running_var_d.IsArray());
  auto layer2_2_bn2_running_var = new int32_t[48][32];
  for (size_t i0 = 0; i0 < 48; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      layer2_2_bn2_running_var[i0][i1] = (layer2_2_bn2_running_var_d[i1 + i0*32].GetFloat());
    }
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

  assert(document.HasMember("layer3_2_bn2_running_var"));
  const Value& layer3_2_bn2_running_var_d = document["layer3_2_bn2_running_var"];
  assert(layer3_2_bn2_running_var_d.IsArray());
  auto layer3_2_bn2_running_var = new int32_t[48][64];
  for (size_t i0 = 0; i0 < 48; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      layer3_2_bn2_running_var[i0][i1] = (layer3_2_bn2_running_var_d[i1 + i0*64].GetFloat());
    }
  }

  assert(document.HasMember("linear_weight"));
  const Value& linear_weight_d = document["linear_weight"];
  assert(linear_weight_d.IsArray());
  auto linear_weight = new int32_t[10][64];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      linear_weight[i0][i1] = (linear_weight_d[i1 + i0*64].GetFloat());
    }
  }

  assert(document.HasMember("linear_bias"));
  const Value& linear_bias_d = document["linear_bias"];
  assert(linear_bias_d.IsArray());
  auto linear_bias = new int32_t[10];
  for (size_t i0 = 0; i0 < 10; i0++) {
    linear_bias[i0] = (linear_bias_d[i0].GetFloat());
  }

  assert(document.HasMember("fc"));
  const Value& fc_d = document["fc"];
  assert(fc_d.IsArray());
  auto fc = new int32_t[1][10];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      fc[i0][i1] = (fc_d[i1 + i0*10].GetFloat());
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

  // global and local worksize
  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};

  // get platform and device information 
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  platforms = (cl_platform_id*) acl_aligned_malloc (numPlatforms * sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,
      maxDevices, devices, &numDevices); CHECK(status);

  // create contex and command queue 
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
  CHECK(status);
  cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 
      CL_QUEUE_PROFILING_ENABLE, &status);
  CHECK(status);

  // read aocx and create binary
  FILE *fp = fopen(AOCX_FILE, "rb");
  fseek(fp, 0, SEEK_END);
  size_t  binary_length = ftell(fp);

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


  // Compute and kernel call from host
  int __device_scope;

  cl_kernel kernel = clCreateKernel(program, "test", &status);
  cl_mem buffer_input_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*1*3*32*32, NULL, &status); CHECK(status);
  cl_mem buffer_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*16*3*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*4*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*48*16, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_0_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_1_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer1_2_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*16*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*48*32, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*16*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_0_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_1_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer2_2_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*32*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_bn2_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*48*64, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*32*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_0_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_1_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_conv1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_layer3_2_conv2_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint)*64*64*3*3, NULL, &status); CHECK(status);
  cl_mem buffer_linear_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*10*64, NULL, &status); CHECK(status);
  cl_mem buffer_fc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*1*10, NULL, &status); CHECK(status);
  cl_mem buffer_linear_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*10, NULL, &status); CHECK(status);

  // Write buffers to device
  status = clEnqueueWriteBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(int32_t)*1*3*32*32, input_image, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_conv1_weight, CL_TRUE, 0, sizeof(int32_t)*16*3*3*3, conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_running_var, CL_TRUE, 0, sizeof(int32_t)*4*16, bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_bn2_running_var, CL_TRUE, 0, sizeof(int32_t)*48*16, layer1_2_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer1_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_bn2_running_var, CL_TRUE, 0, sizeof(int32_t)*48*32, layer2_2_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*16*3*3, layer2_0_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_0_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer2_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_bn2_running_var, CL_TRUE, 0, sizeof(int32_t)*48*64, layer3_2_bn2_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*32*3*3, layer3_0_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_0_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_layer3_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv2_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_linear_weight, CL_TRUE, 0, sizeof(int32_t)*10*64, linear_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_fc, CL_TRUE, 0, sizeof(int32_t)*1*10, fc, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_linear_bias, CL_TRUE, 0, sizeof(int32_t)*10, linear_bias, 0, NULL, NULL); CHECK(status);

  // set device kernel buffer
  int mode = 0; 
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_input_image); CHECK(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_layer1_2_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_layer1_0_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&buffer_layer1_0_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&buffer_layer1_1_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&buffer_layer1_1_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&buffer_layer1_2_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&buffer_layer1_2_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&buffer_layer2_2_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&buffer_layer2_0_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&buffer_layer2_0_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&buffer_layer2_1_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&buffer_layer2_1_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&buffer_layer2_2_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 16, sizeof(cl_mem), (void*)&buffer_layer2_2_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 17, sizeof(cl_mem), (void*)&buffer_layer3_2_bn2_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 18, sizeof(cl_mem), (void*)&buffer_layer3_0_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 19, sizeof(cl_mem), (void*)&buffer_layer3_0_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 20, sizeof(cl_mem), (void*)&buffer_layer3_1_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 21, sizeof(cl_mem), (void*)&buffer_layer3_1_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 22, sizeof(cl_mem), (void*)&buffer_layer3_2_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 23, sizeof(cl_mem), (void*)&buffer_layer3_2_conv2_weight); CHECK(status);
  status = clSetKernelArg(kernel, 24, sizeof(cl_mem), (void*)&buffer_linear_weight); CHECK(status);
  status = clSetKernelArg(kernel, 25, sizeof(cl_mem), (void*)&buffer_fc); CHECK(status);
  status = clSetKernelArg(kernel, 26, sizeof(cl_mem), (void*)&buffer_linear_bias); CHECK(status);
  status = clSetKernelArg(kernel, 27, sizeof(int), (void*)&mode); CHECK(status);
  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_exec_event); CHECK(status);

  // enqueue kernel function
  status = clFlush(cmdQueue); CHECK(status);
  status = clFinish(cmdQueue); CHECK(status);;
  
  double k_start_time;	
  double k_end_time;
  double k_exec_time;

  k_exec_time = compute_kernel_execution_time(kernel_exec_event, k_start_time, k_end_time);     
  printf("\n\n");

  clEnqueueReadBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(int32_t)*1*3*32*32, input_image, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_conv1_weight, CL_TRUE, 0, sizeof(int32_t)*16*3*3*3, conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_running_var, CL_TRUE, 0, sizeof(int32_t)*4*16, bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_bn2_running_var, CL_TRUE, 0, sizeof(int32_t)*48*16, layer1_2_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_0_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_1_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer1_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*16*16*3*3, layer1_2_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_bn2_running_var, CL_TRUE, 0, sizeof(int32_t)*48*32, layer2_2_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*16*3*3, layer2_0_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_0_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_1_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer2_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*32*32*3*3, layer2_2_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_bn2_running_var, CL_TRUE, 0, sizeof(int32_t)*48*64, layer3_2_bn2_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*32*3*3, layer3_0_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_0_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_0_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_1_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_1_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_conv1_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_layer3_2_conv2_weight, CL_TRUE, 0, sizeof(uint)*64*64*3*3, layer3_2_conv2_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_linear_weight, CL_TRUE, 0, sizeof(int32_t)*10*64, linear_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_fc, CL_TRUE, 0, sizeof(int32_t)*1*10, fc, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_linear_bias, CL_TRUE, 0, sizeof(int32_t)*10, linear_bias, 0, NULL, NULL);

  // execution on host 
  printf("\n===== Reporting measured throughput ======\n\n");
  double k_earliest_start_time = k_start_time;
  double k_latest_end_time     = k_end_time;	
  double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;
  
  printf("\n");
  printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);     
  printf("  Drainer kernels end time\t\t= %.5f s\n", k_latest_end_time);     
  printf("  FPGA MatMult exec time\t\t= %.5f s\n", k_overall_exec_time);     


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
  document["bn1_running_var"].Clear();
  rapidjson::Value v_bn1_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 4; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      v_bn1_running_var.PushBack(rapidjson::Value().SetFloat(bn1_running_var[i0][i1]), allocator);
    }
  }
  document["bn1_running_var"] = v_bn1_running_var;
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
  document["layer1_2_bn2_running_var"].Clear();
  rapidjson::Value v_layer1_2_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 48; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      v_layer1_2_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer1_2_bn2_running_var[i0][i1]), allocator);
    }
  }
  document["layer1_2_bn2_running_var"] = v_layer1_2_bn2_running_var;
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
  document["layer2_2_bn2_running_var"].Clear();
  rapidjson::Value v_layer2_2_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 48; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      v_layer2_2_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer2_2_bn2_running_var[i0][i1]), allocator);
    }
  }
  document["layer2_2_bn2_running_var"] = v_layer2_2_bn2_running_var;
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
  document["layer3_2_bn2_running_var"].Clear();
  rapidjson::Value v_layer3_2_bn2_running_var(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 48; i0++) {
    for (size_t i1 = 0; i1 < 64; i1++) {
      v_layer3_2_bn2_running_var.PushBack(rapidjson::Value().SetFloat(layer3_2_bn2_running_var[i0][i1]), allocator);
    }
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

  FILE* fp1 = fopen("outputs.json", "w"); 
 
  char writeBuffer[65536];
  FileWriteStream os(fp1, writeBuffer, sizeof(writeBuffer));
 
  Writer<FileWriteStream> writer(os);
  document.Accept(writer);
  fclose(fp1);

  

  }
