
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
    
  int MAX_RUNS = 1;
  double runtime = 0.0;
  for (int run = 0; run < MAX_RUNS; ++run) {

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
  cl_command_queue cmdQueue[2];
  cmdQueue[0] = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);  CHECK(status);
  cmdQueue[1] = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);  CHECK(status);

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

  std::cout << "[INFO] Just before creating Buffer\n";
  // Compute and kernel call from host
  int __device_scope;

  cl_kernel kernel = clCreateKernel(program, "test", &status); CHECK(status);
  cl_kernel kernel_ret = clCreateKernel(program, "get_output", &status); CHECK(status);
  cl_mem buffer_input_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*1*3*32*32, NULL, &status); CHECK(status);
  cl_mem buffer_fc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t)*1*10, NULL, &status); CHECK(status);
  
  std::cout << "[INFO] Just before writing Buffer\n";
  // Write buffers to device
  status = clEnqueueWriteBuffer(cmdQueue[0], buffer_input_image, CL_TRUE, 0, sizeof(int32_t)*1*3*32*32, input_image, 0, NULL, NULL); CHECK(status);
  //status = clEnqueueWriteBuffer(cmdQueue, buffer_fc, CL_TRUE, 0, sizeof(int32_t)*1*10, fc, 0, NULL, NULL); CHECK(status);

  std::cout << "[INFO] Just before setting kernel argument\n";
  // set device kernel buffer
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_input_image); CHECK(status);
  status = clSetKernelArg(kernel_ret, 0, sizeof(cl_mem), (void*)&buffer_fc); CHECK(status);
  status = clEnqueueNDRangeKernel(cmdQueue[0], kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_exec_event); CHECK(status);
  status = clEnqueueNDRangeKernel(cmdQueue[1], kernel_ret, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_exec_event); CHECK(status);

  // enqueue kernel function
   
  std::cout << "[INFO] Just before starting FPGA execution\n";
  status = clFlush(cmdQueue[0]); CHECK(status);
  status = clFinish(cmdQueue[0]); CHECK(status);
  std::cout << "[INFO] Just finished Queue 1\n";
  status = clFlush(cmdQueue[1]); CHECK(status);
  status = clFinish(cmdQueue[1]); CHECK(status);
  std::cout << "[INFO] Just finished Queue 2\n";

  double k_start_time;	
  double k_end_time;
  double k_exec_time;
  printf("Kernel execution done.\n");
  k_exec_time = compute_kernel_execution_time(kernel_exec_event, k_start_time, k_end_time);     
  printf("\n\n");
 
  /*
  clEnqueueReadBuffer(cmdQueue, buffer_input_image, CL_TRUE, 0, sizeof(int32_t)*1*3*32*32, input_image, 0, NULL, NULL);
  clEnqueueReadBuffer(cmdQueue[1], buffer_fc, CL_TRUE, 0, sizeof(int32_t)*1*10, fc, 0, NULL, NULL);
  */
  // execution on host 
  printf("\n===== Reporting measured throughput ======\n\n");
  double k_earliest_start_time = k_start_time;
  double k_latest_end_time     = k_end_time;	
  double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;
  
  runtime = runtime + k_overall_exec_time;

  printf("\n");
  printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);     
  printf("  Drainer kernels end time\t\t= %.5f s\n", k_latest_end_time);     
  printf("  FPGA MatMult exec time\t\t= %.5f s\n", k_overall_exec_time);     
  }

  printf("Avg runtime for %d runs: %.5f s \n\n", MAX_RUNS, 1.0 * runtime / MAX_RUNS);
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

  */

  }
