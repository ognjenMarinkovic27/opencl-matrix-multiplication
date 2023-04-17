#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)
#define LOCAL_WORK_SIZE 32

void printDeviceData(cl_device_id device_id) {
    char deviceName[64];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);

    cl_uint numberOfComputeUnits;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfComputeUnits), &numberOfComputeUnits, NULL);

    size_t maxWorkgroupSize;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkgroupSize), &maxWorkgroupSize, NULL);

    cl_ulong globalMemorySize;
    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxWorkgroupSize), &globalMemorySize, NULL);

    cl_ulong localMemorySize;
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxWorkgroupSize), &localMemorySize, NULL);

    printf(
        "%s\n"
        "Number of compute units: %d\n"
        "Max Workgroup Size: %lu\n"
        "Global Memory Size: %lu MB\n"
        "Local Memory Size: %lu B\n",
        deviceName, numberOfComputeUnits, maxWorkgroupSize, globalMemorySize / 1048576, localMemorySize);
}

void generateInputMatrices(int m, int n, int k, float* matrixA, float* matrixB) {

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++)
            matrixA[i*k + j] = (float)(rand() % 101 - 50) / 10;
    }

    for(int i = 0; i < k; i++) {
        for(int j = 0; j < n; j++)
            matrixB[i*n + j] = (float)(rand() % 101 - 50) / 10;
    }
}

void multiplyMatricesReference(int m, int n, int k, float* A, float* B, float* C) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {

            float result = 0.0f;
            for(int t = 0; t < k; t++) {
                result += A[i*k + t] * B[n*t + j];
            }

            C[i * n + j] = result;
        }
    }
}

void checkWithReferenceMatrix(int m, int n, int k, float* matrixA, float* matrixB, float* matrixC) {
    float* matrixCReference = (float*)malloc(m*n*sizeof(float));
        multiplyMatricesReference(m, n, k, matrixA, matrixB, matrixCReference);

       printf("Comparing matrices...\n");

        double sumOfDifferences = 0;
        double maxDifference = 0;

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                double dif = fabs(matrixC[i*n + j] - matrixCReference[i*n +j]);
                sumOfDifferences += dif;
                maxDifference = fmax(maxDifference, dif);
            }
        }

       printf("Average difference between OpenCL and Reference result: %e\nMaximum difference between OpenCL and Reference result: %e\n", sumOfDifferences / (n*m), maxDifference);

        free(matrixCReference);
}

cl_int performIteration(
    int iteration,
    cl_context context, 
    cl_command_queue commandQueue, 
    cl_program program,
    int m, int n, int k,
    cl_mem matrixAInputBuffer,
    cl_mem matrixBInputBuffer, 
    cl_mem matrixCOutputBuffer, 
    float* matrixA, float* matrixB, float* matrixC, 
    cl_ulong* times) {

    cl_int errorCode;

    /* Moving data to buffers on the device */
    errorCode = clEnqueueWriteBuffer(commandQueue, matrixAInputBuffer, CL_TRUE, 0, m*k*sizeof(float), (void*)matrixA, 0, NULL, NULL);
    errorCode = clEnqueueWriteBuffer(commandQueue, matrixBInputBuffer, CL_TRUE, 0, k*n*sizeof(float), (void*)matrixB, 0, NULL, NULL);

    /* Creating Kernel and Supplying Arguments */
    cl_kernel kernel = clCreateKernel(program, "multiply_matrices", &errorCode);
    errorCode = clSetKernelArg(kernel, 0, sizeof(int), (void*)&m);
    errorCode = clSetKernelArg(kernel, 1, sizeof(int), (void*)&n);
    errorCode = clSetKernelArg(kernel, 2, sizeof(int), (void*)&k);
    errorCode = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&matrixAInputBuffer);
    errorCode = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&matrixBInputBuffer);
    errorCode = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&matrixCOutputBuffer);

    /* Defining Problem Size */
    const int localSize = LOCAL_WORK_SIZE;
    const size_t local[2] = { localSize, localSize };
    const size_t global[2] = { m, n };

    /* Executing Kernel */
    cl_event kernelExecutionEvent;
    errorCode = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global, local, 0, NULL, &kernelExecutionEvent);
    
    /* Waiting for Kernel to Finish */
    clFinish(commandQueue);

    /* Calculating time spent executing */
    cl_ulong startTime, endTime;
    errorCode = clGetEventProfilingInfo(kernelExecutionEvent, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL);
    errorCode = clGetEventProfilingInfo(kernelExecutionEvent, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, NULL);

    printf("Iteration %d: Start Time: %lu. End Time: %lu. Execution time: %lu ns\n", iteration, startTime, endTime, endTime - startTime);
    times[iteration] = endTime - startTime;

    /* Reading Output From Device */
    errorCode = clEnqueueReadBuffer(commandQueue, matrixCOutputBuffer, CL_TRUE, 0, m*n*sizeof(float), (void*)matrixC, 0, NULL, NULL);

    return errorCode;
}

int main (int argc, char* argv[]) {
    
    srand(time(NULL));

    int m, n, k, numberOfRuns;

    /* Reading command line arguments */
    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &n);
    sscanf(argv[3], "%d", &k);
    sscanf(argv[4], "%d", &numberOfRuns);

    /* Getting Platform ID */
    cl_platform_id platform;
    cl_int errorCode = clGetPlatformIDs(1, &platform, NULL);

    /* Getting Default Device ID */
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    errorCode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    printDeviceData(device_id);

    /* Context properties */
    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, platform, 0 };
    
    /* Creating context and command queue */
    cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, &errorCode);
    cl_command_queue commandQueue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &errorCode);

    /* Creating Input Buffers */
    cl_mem matrixAInputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, m*k*sizeof(float), NULL, &errorCode);
    cl_mem matrixBInputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, k*n*sizeof(float), NULL, &errorCode);

    /* Reading Kernel From Source File */
    FILE* kernelFile;
    char* sourceString;
    size_t sourceSize;
    kernelFile = fopen("multiply_matrices.cl", "r");
    if (!kernelFile) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    sourceString = (char*)malloc(MAX_SOURCE_SIZE);
    sourceSize = fread(sourceString, 1, MAX_SOURCE_SIZE, kernelFile);
    fclose(kernelFile);

    /* Creating and Building Program */
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceString, (const size_t*)&sourceSize, &errorCode);
    errorCode = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Allocating space for the Host Matrices */
    float* matrixA = (float*)malloc(m*k*sizeof(float));
    float* matrixB = (float*)malloc(k*n*sizeof(float));
    float* matrixC = (float*)malloc(m*n*sizeof(float));

    generateInputMatrices(m, n, k, matrixA, matrixB);

    /* An array to store the times taken for each run */
    cl_ulong* times = (cl_ulong*)malloc(sizeof(cl_ulong) * numberOfRuns);

    /* Creating output buffer */
    cl_mem matrixCOutputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m*n*sizeof(float), NULL, &errorCode);

    for(int i = 0; i < numberOfRuns; i++) {
        errorCode = performIteration(
            i, context, commandQueue, program, 
            m, n, k, 
            matrixAInputBuffer, matrixBInputBuffer, matrixCOutputBuffer,
            matrixA, matrixB, matrixC, 
            times);
    }

    //checkWithReferenceMatrix(m, n, k, matrixA, matrixB, matrixC);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    /* Calculating Average Time and Standard Deviation */
    cl_ulong timeSum = 0;
    for(int i = 0; i < numberOfRuns; i++) { 
        timeSum += times[i];
    }
    double averageTime = timeSum/numberOfRuns;

    double squareDifferenceSum = 0;
    for(int i = 0; i < numberOfRuns; i++) { 
        squareDifferenceSum += (averageTime - times[i])*(averageTime - times[i]);
    }

    double standardDeviation = sqrt(squareDifferenceSum / (numberOfRuns - 1));

    free(times);

    printf("Average Kernel Execution Time: %f ns\nStandard Deviation: %f ns\n", averageTime, standardDeviation);
}