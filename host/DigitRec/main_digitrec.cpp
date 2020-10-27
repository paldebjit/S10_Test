// System includes 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "ref/ac_int.h"
#include <chrono>

using namespace std;

#define DPRINTF(...) printf(__VA_ARGS__); fflush(stdout);

typedef chrono::high_resolution_clock Clock;

#ifdef _WIN32
#include <time.h>
#include <windows.h>
#else
#include <sys/time.h>
#endif

#define ALTERA_CL 1

#ifdef ALTERA_CL
#pragma message ("* Compiling for ALTERA CL")
#endif

#ifdef ALTERA_CL
#include "CL/opencl.h"
#endif

#define ACL_ALIGNMENT 64

#include <stdlib.h>
void* acl_aligned_malloc (size_t size) {
    void *result = NULL;
    posix_memalign (&result, ACL_ALIGNMENT, size);
    return result;
}
void acl_aligned_free (void *ptr) {
    free (ptr);
}

//#define EMULATOR
#define COMPUTE_GOLDEN_BLOCKED
//#define COMPUTE_GOLDEN

#define AOCX_FILE "AOCX/digitrec.aocx"

// Check the status returned by the OpenCL API functions
#define CHECK(status) 							                	\
    if (status != CL_SUCCESS)		                				\
{	                                								\
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
    exit(1);						                            	\
}								                                	\

// Check the status returned by the OpenCL API functions, don't exit on error
#define CHECK_NO_EXIT(status) 	        							\
    if (status != CL_SUCCESS)			                			\
{								                                	\
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
}								                                	\


double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    start_d = (double)1.0e-9 * start;
    end_d   = (double)1.0e-9 * end;

    return 	(double)1.0e-9 * (end - start); // nanoseconds to seconds
}

long long int string_to_hex(char a[], int hex_len) {
    long long int x = 0;
    int num = 0;
    for (int i =0; i < hex_len; i++) {
        if (i == 0 || i == 1 || a[i] == '\0')
            continue;
        if(a[i] == '0')
            num = 0;
        else if(a[i] == '1')
            num = 1;
        else if(a[i] == '2')
            num = 2;
        else if(a[i] == '3')
            num = 3;
        else if(a[i] == '4')
            num = 4;
        else if(a[i] == '5')
            num = 5;
        else if(a[i] == '6')
            num = 6;
        else if(a[i] == '7')
            num = 7;
        else if(a[i] == '8')
            num = 8;
        else if(a[i] == '9')
            num = 9;
        else if(a[i] == 'A' || a[i] == 'a')
            num = 10;
        else if(a[i] == 'B' || a[i] == 'b')
            num = 11;
        else if(a[i] == 'C' || a[i] == 'c')
            num = 12;
        else if(a[i] == 'D' || a[i] == 'd')
            num = 13;
        else if(a[i] == 'E' || a[i] == 'e')
            num = 14;
        else if(a[i] == 'F' || a[i] == 'f')
            num = 15;
        x = x + num * pow(16, hex_len - i - 2);
    }
    return x;
}

void populateData(int size, uint64_t* ptr) {
    int ctr = 0;
    FILE *fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    size_t bufferLength = 255;
    char *buffer = (char *)malloc(sizeof(char) * bufferLength);

    fp = fopen("./data/complete_training.dat", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    printf("Reading training data\n");
    while((read = getline(&line, &len, fp)) != -1) {
        if (ctr > size)
            break;
        //printf("Retrieved train image: %s", line);
        ptr[ctr] = string_to_hex(line, read);
        ctr = ctr + 1;
    }
    printf("Reading training data DONE\n");
    fclose(fp);
    if (line) {
        free(line);
    }
}

int main(int argc, const char** argv) {

    printf("%s Starting...\n\n", argv[0]); 
    int MAX_RUNS = 100;
    double runtime = 0.0;
    for (int run = 0; run < MAX_RUNS; ++run) {
    unsigned int elements;
    FILE * file;
    long int fstart, fend;
    unsigned int i;
    cl_event kernel_exec_event;

    std::streampos filesize;
    FILE *f_out = stdout;

    uint64_t* param1;            // train_images
    uint8_t*  param2;            // knn_mat
    int size1 = 10 * 1800;       // 10 * 1800    --  train images ??
    int size2 = 10 * 3;          // 10 * 3       --  knn mat

    uint64_t test_image;
    test_image = 0x3041060800;
    //test_image.bit_fill_hex("0x3041060800");

    printf("\n===== Host-CPU preparing data ======\n\n");
    
    if((param1 = (uint64_t*)acl_aligned_malloc(size1*sizeof(param1))) == NULL) {
        perror("Failed malloc of param0 vector");
    }
    populateData(size1, param1);
    
    if((param2 = (uint8_t*)acl_aligned_malloc(size2*sizeof(param2))) == NULL) {
        perror("Failed malloc of param1 vector");
    }

    printf("Allocated memory for host-side\n");
    printf("\n===== Host-CPU setting up the OpenCL platform and device ======\n\n");

    // Use this to check the output of each API call
    cl_int status;

    //----------------------------------------------
    // Discover and initialize the platforms
    //----------------------------------------------
    cl_uint numPlatforms = 0;
    cl_platform_id* platforms = NULL;

    // Use clGetPlatformIDs() to retrieve the
    // number of platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    fprintf(stdout,"Number of platforms = %d\n", numPlatforms);

    // Allocate enough space for each platform
    platforms = (cl_platform_id*) acl_aligned_malloc (numPlatforms * sizeof(cl_platform_id));
    printf("Allocated space for Platform\n");

    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
    printf("Filled in platforms\n");    

    //----------------------------------------------
    // Discover and initialize the devices 
    //----------------------------------------------
    cl_uint numDevices = 0;

    // Device info
    char buffer[4096];
    unsigned int buf_uint;
    int device_found = 0;
    const cl_uint maxDevices = 4;
    cl_device_id devices[maxDevices];
    DPRINTF("Initializing IDs\n");
    for (int i=0; i<numPlatforms; i++) {
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, maxDevices, devices, &numDevices); 
        CHECK(status);

        if(status == CL_SUCCESS){
          clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 4096, buffer, NULL);
#if defined(ALTERA_CL)
            if(strstr(buffer, "Altera") != NULL){
                device_found = 1;
            }
            DPRINTF("%s\n", buffer);
#elif defined(NVIDIA_CL)
            if(strstr(buffer, "NVIDIA") != NULL){
                device_found = 1;
            }
#else
            if(strstr(buffer, "Intel") != NULL){
                device_found = 1;
            }
#endif

            DPRINTF("Platform found : %s\n", buffer);
            device_found = 1;
        }
    }


    if(!device_found) {
        printf("failed to find a OpenCL device\n");
        exit(-1);
    }

    for (i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i],
                CL_DEVICE_NAME,
                4096,
                buffer,
                NULL);
        fprintf(f_out, "\nDevice Name: %s\n", buffer);

        clGetDeviceInfo(devices[i],
                CL_DEVICE_VENDOR,
                4096,
                buffer,
                NULL);
        fprintf(f_out, "Device Vendor: %s\n", buffer);

        clGetDeviceInfo(devices[i],
                CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(buf_uint),
                &buf_uint,
                NULL);
        fprintf(f_out, "Device Computing Units: %u\n", buf_uint);

        clGetDeviceInfo(devices[i],
                CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(unsigned long),
                &buffer,
                NULL);
        fprintf(f_out, "Global Memory Size: %i\n", *((unsigned long*)buffer));

        clGetDeviceInfo(devices[i],
                CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                sizeof(unsigned long),
                &buffer,
                NULL);
        fprintf(f_out, "Global Memory Allocation Size: %i\n\n", *((unsigned long*)buffer));
    }

    //----------------------------------------------
    // Create a context 
    //----------------------------------------------

    printf("\n===== Host-CPU setting up the OpenCL command queues ======\n\n");

    cl_context context = NULL;

    // Create a context using clCreateContext() and
    // associate it with the device

    context = clCreateContext(
            NULL,
            1,
            devices,
            NULL,
            NULL,
            &status); CHECK(status);

    //----------------------------------------------
    // Create command queues
    //---------------------------------------------

    cl_command_queue cmdQueue[2]; // extra queue for reading buffer C

    const char *kernel_names[] = { "default_function" };

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute on
    fprintf(stdout,"cmdQueue i = %d, kernel name = %s\n", 0, kernel_names[0]);
    cmdQueue[0] = clCreateCommandQueue(
            context,
            devices[0],
            CL_QUEUE_PROFILING_ENABLE,
            &status); CHECK(status);

    fprintf(stdout,"cmdQueue i = %d, a queue for reading the C buffer\n", 1);
    cmdQueue[1] = clCreateCommandQueue(
            context,
            devices[0],
            CL_QUEUE_PROFILING_ENABLE,
            &status); CHECK(status);

    //----------------------------------------------
    // Create device buffers
    //----------------------------------------------
    
    /* Arguments for the kmeans kernel*/
    cl_mem train_images, knn_mat; 

    printf("\n===== Host-CPU transferring matrices A,B to the FPGA device global memory (DDR4) via PCIe ======\n\n");

    train_images = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE, 
            size1 * sizeof(uint64_t), 
            NULL, 
            &status);
    CHECK(status);

    knn_mat = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE, 
            size2 * sizeof(uint8_t), 
            NULL, 
            &status);
    CHECK(status);


    //----------------------------------------------
    // Write host data to device buffers
    //----------------------------------------------

    // blocking writes
    status = clEnqueueWriteBuffer(
        cmdQueue[0],
        train_images,
        CL_TRUE,
        0,
        size1*sizeof(uint64_t),
        param1,
        0,
        NULL,
        NULL); CHECK(status);

    status = clEnqueueWriteBuffer(
        cmdQueue[0],
        knn_mat,
        CL_TRUE,
        0,
        size2*sizeof(uint8_t),
        param2,
        0,
        NULL,
        NULL); CHECK(status);


    //----------------------------------------------
    // Create the program from binaries
    //----------------------------------------------
    printf("\n===== Host-CPU setting up OpenCL program and kernels ======\n\n");

    cl_program program;

    size_t binary_length;
    const unsigned char *binary;

    printf("\nAOCX file: %s\n\n", AOCX_FILE);
    // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
    FILE *fp = fopen(AOCX_FILE, "rb");

    if (fp == NULL) {
        printf("Failed to open the AOCX file (fopen).\n");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    binary_length = ftell(fp);
    binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
    assert(binary && "Malloc failed");
    rewind(fp);

    if (fread((void*)binary, binary_length, 1, fp) == 0) {
        printf("Failed to read from the AOCX file (fread).\n");
        return -1;
    }
    fclose(fp);

    // Create a program using clCreateProgramWithBinary()
    program = clCreateProgramWithBinary(
            context,
            1,
            devices,
            &binary_length,
            (const unsigned char **)&binary,
            &status,
            NULL); CHECK(status);


    //----------------------------------------------
    // Create the kernel
    //----------------------------------------------

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(status != CL_SUCCESS) {
        char log[128*1024] = {0};
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 128*1024, log, NULL);
        printf("%s\n", log);
        CHECK(status);
    }

    printf("Creating kernel \n");
    cl_kernel kernel = clCreateKernel(program, kernel_names[0], &status); 
    CHECK(status);

    status = clSetKernelArg(
            kernel,
            0,
            sizeof(uint64_t),
            (void*)&test_image); CHECK(status);

    status = clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            (void*)&train_images); CHECK(status);

    status = clSetKernelArg(
            kernel,
            2,
            sizeof(cl_mem),
            (void*)&knn_mat); CHECK(status);

    //----------------------------------------------
    // Configure the work-item structure (using only tasks atm)
    //----------------------------------------------

    // Define the number of threads that will be created 
    // as well as the number of work groups 
    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    //----------------------------------------------
    // Enqueue the kernel for execution
    //----------------------------------------------


    // all kernels are always tasks
    globalWorkSize[0] = 1;
    localWorkSize[0]  = 1;

    printf("\n===== Host-CPU enqeuing the OpenCL kernels to the FPGA device ======\n\n");
    // Alternatively, can use clEnqueueTaskKernel
    printf("clEnqueueNDRangeKernel!\n");

    auto t1 = Clock::now();

    status = clEnqueueNDRangeKernel(
            cmdQueue[0],
            kernel,
            1,
            NULL,
            globalWorkSize,
            localWorkSize,
            0,
            NULL,
            &kernel_exec_event
            );
    CHECK(status);
    printf(" *** FPGA execution started!\n");

    status = clFlush(cmdQueue[0]); CHECK(status);
    status = clFinish(cmdQueue[0]); CHECK(status);

    auto t2 = Clock::now();

    cout << "Kernel execution time from C++ timer: "
         << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() * 1.0e-6
         << " ms" << std::endl;

    printf(" *** FPGA execution finished!\n");
    double k_start_time;	
    double k_end_time;
    double k_exec_time;

    k_exec_time = compute_kernel_execution_time(kernel_exec_event, k_start_time, k_end_time);     
    printf("\n\n");

    printf("\n===== Host-CPU transferring result matrix C from the FPGA device global memory (DDR4) via PCIe ======\n\n");

    // Read the results back from the device, blocking read
    clEnqueueReadBuffer(
        //cmdQueue[KID_DRAIN_MAT_C],
        cmdQueue[1], // using a special queue for reading buffer C
        knn_mat,
        CL_TRUE,
        0,
        size1*sizeof(uint8_t),
        param2,
        0,
        NULL,
        NULL); CHECK(status);

    status = clFinish(cmdQueue[1]); CHECK(status);

    // Check the result 
    //bool matched = true;
    /*
    for (int k = 0; k < size; k++) {
        if (input[k] != times * (1+2+3)) {
            std::cout << "Mismatched... " << input[k];
            matched = false;
        }
    }
    */

    //if (matched) std::cout << "Result matched...\n";
    //else std::cout << "Mismatch...\n";

    printf("\n===== Reporting measured throughput ======\n\n");
    double k_earliest_start_time = k_start_time;
    double k_latest_end_time     = k_end_time;	
    double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;

    printf("\n");
    printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);     
    printf("  Drainer kernels end time\t\t= %.5f s\n", k_latest_end_time);     
    printf("  FPGA MatMult exec time\t\t= %.5f s\n", k_overall_exec_time);     
    runtime = runtime + k_overall_exec_time;

    // multiplied by 1.0e-9 to get G-FLOPs
    //printf("\n");

    //double num_operations = (double)2.0 * 512 * 512 * 512 * 2;

    //printf("  # operations = %.0f\n", num_operations );     
    //printf("  Throughput: %.5f GFLOPS\n", (double)1.0e-9 * num_operations / k_overall_exec_time);     
    //----------------------------------------------
    // Release the OpenCL resources
    //----------------------------------------------

    // Free resources
    /*
    clReleaseKernel(kernel);
    clReleaseCommandQueue(cmdQueue[0]);
    clReleaseCommandQueue(cmdQueue[1]);
    clReleaseEvent(kernel_exec_event);

    acl_aligned_free(input);
    clReleaseMemObject(d_input);

    clReleaseProgram(program);
    clReleaseContext(context);

    acl_aligned_free(platforms);
    acl_aligned_free(devices);
    */
    }
    printf("Avg runtime for %d runs: %.5f s\n\n", MAX_RUNS, 1.0 * runtime / MAX_RUNS);
}


