
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
  
  auto bn1 = new float[16384];
  for (size_t i0 = 0; i0 < 16384; i0++) {
    bn1[i0] = 0.0;
  }

  assert(document.HasMember("bn1_weight"));
  const Value& bn1_weight_d = document["bn1_weight"];
  assert(bn1_weight_d.IsArray());
  auto bn1_weight = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_weight[i0] = (bn1_weight_d[i0].GetFloat()) / 1000.00;
    printf("weight value = %f\n", bn1_weight_d[i0].GetFloat() / 1000.0);
  }

  assert(document.HasMember("bn1_bias"));
  const Value& bn1_bias_d = document["bn1_bias"];
  assert(bn1_bias_d.IsArray());
  auto bn1_bias = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_bias[i0] = (bn1_bias_d[i0].GetFloat()) / 1000.0;
    printf("bias value = %f\n", bn1_bias_d[i0].GetFloat() / 1000.0);
  }

  assert(document.HasMember("bn1_running_mean"));
  const Value& bn1_running_mean_d = document["bn1_running_mean"];
  assert(bn1_running_mean_d.IsArray());
  auto bn1_running_mean = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_running_mean[i0] = (bn1_running_mean_d[i0].GetFloat()) / 1000.0;
    printf("mean value = %f\n", bn1_running_mean_d[i0].GetFloat() / 1000.0);
  }

  assert(document.HasMember("bn1_running_var"));
  const Value& bn1_running_var_d = document["bn1_running_var"];
  assert(bn1_running_var_d.IsArray());
  auto bn1_running_var = new float[16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    bn1_running_var[i0] = (bn1_running_var_d[i0].GetFloat()) / 1000.0;
    printf("var value = %f\n", bn1_running_var_d[i0].GetFloat() / 1000.0);
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
  cl_mem buffer_bn1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16384, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_running_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_running_var = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);
  cl_mem buffer_bn1_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*16, NULL, &status); CHECK(status);

  // write buffers to device
  status = clEnqueueWriteBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(cl_float)*1*3*32*32, input_image, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_conv1_weight, CL_TRUE, 0, sizeof(cl_float)*16*3*3*3, conv1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_mean, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_var, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, bn1_weight, 0, NULL, NULL); CHECK(status);
  status = clEnqueueWriteBuffer(cmdQueue, buffer_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, bn1_bias, 0, NULL, NULL); CHECK(status);

  // set device kernel buffer
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_input_image); CHECK(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_conv1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_bn1); CHECK(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_bn1_running_mean); CHECK(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_bn1_running_var); CHECK(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&buffer_bn1_weight); CHECK(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&buffer_bn1_bias); CHECK(status);
  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_exec_event); CHECK(status);

  // enqueue kernel function
  status = clFlush(cmdQueue); CHECK(status);
  status = clFinish(cmdQueue); CHECK(status);;
  clEnqueueReadBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(cl_float)*1*3*32*32, input_image, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_conv1_weight, CL_TRUE, 0, sizeof(cl_float)*16*3*3*3, conv1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1, CL_TRUE, 0, sizeof(cl_float)*16384, bn1, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_running_mean, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_mean, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_running_var, CL_TRUE, 0, sizeof(cl_float)*16, bn1_running_var, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_weight, CL_TRUE, 0, sizeof(cl_float)*16, bn1_weight, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue, buffer_bn1_bias, CL_TRUE, 0, sizeof(cl_float)*16, bn1_bias, 0, NULL, NULL);

  std::cout << "Done running kernel on FPGA\n";

  for (size_t i0 = 0; i0 < 20; i0++) {
      printf("bn1 value = %f\n", bn1[i0]);
  }

}
