////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <external/EasyBMP/EasyBMP.h>
#include <external/EasyBMP/EasyBMP.cpp>
#include <external/oclm/ocl_programs/kernelReader.hpp>

#define MEM_SIZE (512)
#define MAX_SOURCE_SIZE (0x100000)

int main()
{
    {
        float raws[MEM_SIZE*MEM_SIZE];

        cl_platform_id platform_id = NULL;
        cl_device_id device_id = NULL;
        cl_context context = NULL;
        cl_command_queue command_queue = NULL;

        cl_mem rawMemobj = NULL;

        cl_program program = NULL;
        cl_kernel kernel = NULL;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret;

        FILE *fp;
        const char fileName[] = "fractalLimit.cl";
        size_t source_size;
        char *source_str = new char[MAX_SOURCE_SIZE];
        cl_int i;
        if (!readKernel(fileName, source_size, source_str, MAX_SOURCE_SIZE))
        {
            fprintf(stderr, "Failed to load kernel.\n");
            system("Pause");
            exit(1);
        }

        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

        context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

        rawMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(raws), NULL, &ret);

        program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        kernel = clCreateKernel(program, "set", &ret);


        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&rawMemobj);

	    size_t global_work_size[3] = {MEM_SIZE, MEM_SIZE, 1};

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL);

	    if (ret)
	    {
		    std::cout << ret << std::endl;
	    }

        ret = clEnqueueReadBuffer(command_queue, rawMemobj, CL_TRUE, 0,  sizeof(raws), raws, 0, NULL, NULL);

	
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(rawMemobj);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
	
	    if (ret)
	    {
		    std::cout << ret << std::endl;
	    }

        free(source_str);
	    int size = MEM_SIZE;
	    {
            BMP SetImage;
	        SetImage.SetBitDepth(24);
	        SetImage.SetSize(size*2,size);
	        RGBApixel pix;
	        for (int i = 0; i < size; i++)
            {
		        for (int j = 0; j < size; j++)
		        {
                    for (int k = 0; k < 2; k++)
                    {
			            pix.Blue = (int)raws[i * size + j];
			            pix.Red = (int)raws[i * size + j];
			            pix.Green = (int)raws[i * size + j];
                        SetImage.SetPixel(i * 2 + k,j,pix);
                    }
		        }
            }
	        SetImage.WriteToFile("out.bmp");
	    }
    }
    system("Pause");
    return 0;
}