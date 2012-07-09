#include "OGL.hpp"
int main(int argc, char* argv[]) {

	{
        cl_platform_id platform_id = NULL;
        cl_device_id device_id = NULL;
        cl_context context = NULL;
        cl_command_queue command_queue = NULL;
        cl_mem blueMemobj = NULL;
        cl_mem redMemobj = NULL;
        cl_mem greenMemobj = NULL;
        cl_program program = NULL;
        cl_kernel kernel = NULL;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret;
    
        ebmpBYTE blue[MEM_SIZE * MEM_SIZE];
        //ebmpBYTE red[MEM_SIZE * MEM_SIZE];
        //ebmpBYTE green[MEM_SIZE * MEM_SIZE];

        FILE *fp;
        const char fileName[] = "./kernel.cl";
        size_t source_size;
        char *source_str;
        cl_int i;

        fp = fopen(fileName, "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        source_str = (char *)malloc(MAX_SOURCE_SIZE);
        source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp );
        fclose( fp );
	

        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

        context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

        blueMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(blue), NULL, &ret);
        //redMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(red), NULL, &ret);
        //greenMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(red), NULL, &ret);

        //ret = clEnqueueWriteBuffer(command_queue, blueMemobj, CL_TRUE, 0, sizeof(blue), blue, 0, NULL, NULL);

        program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        kernel = clCreateKernel(program, "set", &ret);

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&blueMemobj);
        //ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&redMemobj);
        //ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&greenMemobj);

	    size_t global_work_size[3] = {MEM_SIZE, MEM_SIZE, 1};

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL);

	    if (ret)
	    {
		    std::cout << ret << std::endl;
	    }

        ret = clEnqueueReadBuffer(command_queue, blueMemobj, CL_TRUE, 0, sizeof(blue), blue, 0, NULL, NULL);
        //ret = clEnqueueReadBuffer(command_queue, redMemobj, CL_TRUE, 0, sizeof(red), red, 0, NULL, NULL);
        //ret = clEnqueueReadBuffer(command_queue, greenMemobj, CL_TRUE, 0, sizeof(green), green, 0, NULL, NULL);

	
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(blueMemobj);
        //ret = clReleaseMemObject(redMemobj);
        //ret = clReleaseMemObject(greenMemobj);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
	
	    if (ret)
	    {
		    std::cout << ret << std::endl;
	    }

        free(source_str);
	    {
	    SetImage.SetBitDepth(24);
	    int size = MEM_SIZE;
	    SetImage.SetSize(size,size);
	    RGBApixel pix;
	    for (int i = 0; i < size; i++)
		    for (int j = 0; j < size; j++)
		    {
			    pix.Blue = blue[i * size + j];
			    pix.Red = blue[i * size + j];
			    pix.Green = blue[i * size + j];
			    SetImage.SetPixel(i,j,pix);
		    }
	    SetImage.WriteToFile("out.bmp");
	    }
    }
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA|GLUT_ALPHA|GLUT_DOUBLE|GLUT_DEPTH);
  glutInitWindowSize(screen_width, screen_height);
  glutCreateWindow("Fractal Visualizer");

  GLenum glew_status = glewInit();
  if (glew_status != GLEW_OK) {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
    return 1;
  }

  if (!GLEW_VERSION_2_0) {
    fprintf(stderr, "Error: your graphic card does not support OpenGL 2.0\n");
    return 1;
  }

  if (init_resources()) {
    glutDisplayFunc(onDisplay);
    glutReshapeFunc(onReshape);
    glutIdleFunc(onIdle);
	glutKeyboardFunc(processNormalKeys);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glutMainLoop();
  }

  free_resources();
  return 0;
}
