//Done by: Aaron Lim
//Studdent ID: 5985171

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "common.h"
#include "bmpfuncs.h"

void kernelFunction(cl::Kernel, cl::Image2D, cl::Image2D, int &, int, int, cl::CommandQueue, unsigned char*, cl::Program, bool, cl::ImageFormat, cl::Context);

#define NUM_INTS 4096
#define NUM_ITEMS 512
#define NUM_ITERATIONS 1000

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	unsigned char* inputImage;
	unsigned char* outputImage;
	int imgWidth, imgHeight, imageSize;

	//declare for imageformat and image2Ds
	cl::ImageFormat imgFormat;
	cl::Image2D inputImgBuffer, outputImgBuffer;
	
	//for later use
	int no = 0;

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		//choose image
		int c;
		while (true) {
			std::cout << "Choose 1 image: " << std::endl;
			std::cout << "1. bunnyimage1" << std::endl;
			std::cout << "2. bunnyimage2" << std::endl;
			std::cout << "Choice: ";
			std::cin >> c;
			if (c == 1 || c == 2)
				break;
			else
				std::cout << "Please input 1,2" << std::endl;
		}
		std::cin.clear();
		std::cin.ignore(1000, '\n');

		// create a context from device
		context = cl::Context(device);

		// build the program
		if(!build_program(&program, &context, "part1_kernel.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create command queue
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
		
		// read input image
		if(c == 1)
			inputImage = read_BMP_RGB_to_RGBA("bunnycity1.bmp", &imgWidth, &imgHeight);
		else
			inputImage = read_BMP_RGB_to_RGBA("bunnycity2.bmp", &imgWidth, &imgHeight);
		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		outputImage = new unsigned char[imageSize];

		// image format
		imgFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		// create image objects
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		//prompt user to enter 1,2,3
		while (true) {
			std::cout << "1a) Blur lvl (1: 3x3, 2: 5x5, 3: 7x7): ";
			std::cin >> no;
			if (no > 0 && no < 4) {
				break;
			}
			else {
				std::cout << "Please input 1,2,3" << std::endl;
			}
		}
		std::cin.clear();
		std::cin.ignore(1000,'\n');
		no--;
		bool ok = true;

		//calling the kernel function
		kernelFunction(kernel, inputImgBuffer, outputImgBuffer, no, imgWidth, imgHeight, queue, outputImage, program, ok, imgFormat, context);
		
		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		std::cout << "Done." << std::endl;

		//prompting user for 1,2,3
		while (true) {
			std::cout << "1b) Blur lvl (1: 1x3, 2: 1x5, 3: 1x7): ";
			std::cin >> no;
			if (no > 0 && no < 4) {
				break;
			}
			else {
				std::cout << "Please input 1,2,3" << std::endl;
			}
		}
		no--;
		std::cin.clear();
		std::cin.ignore(1000, '\n');
		ok = false;

		//calling the kernel function
		kernelFunction(kernel, inputImgBuffer, outputImgBuffer, no, imgWidth, imgHeight, queue, outputImage, program, ok, imgFormat, context);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		std::cout << "Done." << std::endl;

		// declare events
		cl::Event profileEvent;
		cl_ulong timeStart, timeEnd, timeTotal = 0;

		//declare offset and globalsize
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		std::cout << "1c: " << std::endl;
		for (int j = 0; j < 3; j++) {
			timeTotal = 0;
			for (int i = 0; i < NUM_ITERATIONS; i++)
			{
				// create a kernel
				kernel = cl::Kernel(program, "simple_conv");

				// set kernel arguments
				kernel.setArg(0, inputImgBuffer);
				kernel.setArg(1, outputImgBuffer);
				kernel.setArg(2, j);

				//enqueue for profiling
				queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
				queue.finish();			

				//getting the time start and end
				timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
				timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

				//getting total time taken
				timeTotal += timeEnd - timeStart;
			}
			//getting the average
			timeTotal /= NUM_ITERATIONS;
			if (j == 0) {
				std::cout << "3x3 (Naive approach): " << timeTotal << std::endl;
			}
			else if (j == 1) {
				std::cout << "5x5 (Naive approach): " << timeTotal << std::endl;
			}
			else if (j == 2) {
				std::cout << "7x7 (Naive approach): " << timeTotal << std::endl;
			}
			timeTotal = 0;
			
			ok = false;
			for (int i = 0; i < NUM_ITERATIONS; i++)
			{
				// create a kernel
				kernel = cl::Kernel(program, "simple_row");

				// set kernel arguments
				kernel.setArg(0, inputImgBuffer);
				kernel.setArg(1, outputImgBuffer);
				kernel.setArg(2, j);

				queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
				queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

				cl::Image2D columnBuffer;

				inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
				columnBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

				kernel = cl::Kernel(program, "simple_column");
				kernel.setArg(0, inputImgBuffer);
				kernel.setArg(1, columnBuffer);
				kernel.setArg(2, j);

				queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
				queue.finish();

				timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
				timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

				timeTotal += timeEnd - timeStart;
			}
			timeTotal /= NUM_ITERATIONS;
			if (j == 0) {
				std::cout << "1x3: " << timeTotal << std::endl;
			}
			else if (j == 1) {
				std::cout << "1x5: " << timeTotal << std::endl;
			}
			else if (j == 2) {
				std::cout << "1x7: " << timeTotal << std::endl;
			}
		}
		// deallocate memory
		free(inputImage);
		free(outputImage);
	}
	// catch any OpenCL function errors
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

#ifdef _WIN32
	// wait for a keypress on Windows OS before exiting
	std::cout << "\npress a key to quit...";
	std::cin.ignore();
#endif

	return 0;
}

void kernelFunction(cl::Kernel kernel, cl::Image2D inputImgBuffer, cl::Image2D outputImgBuffer, 
	int &no, int imgWidth, int imgHeight, cl::CommandQueue queue, unsigned char* outputImage, cl::Program program, bool ok,
	cl::ImageFormat imgFormat, cl::Context context) {
	
	// enqueue kernel
	cl::NDRange offset(0, 0);
	cl::NDRange globalSize(imgWidth, imgHeight);

	// enqueue command to read image from device to host memory
	cl::size_t<3> origin, region;
	origin[0] = origin[1] = origin[2] = 0;
	region[0] = imgWidth;
	region[1] = imgHeight;
	region[2] = 1;
	
	if (ok == true) {
		// create a kernel
		kernel = cl::Kernel(program, "simple_conv");

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, no);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
	}

	else if (ok == false) {
		// create a kernel
		kernel = cl::Kernel(program, "simple_row");

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, no);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
		
		cl::Image2D columnBuffer;
	
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		columnBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		kernel = cl::Kernel(program, "simple_column");
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, columnBuffer);
		kernel.setArg(2, no);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		queue.enqueueReadImage(columnBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
	}



	// output results to image file
	if (ok) {
		if (no == 0)
			write_BMP_RGBA_to_RGB("output3x3.bmp", outputImage, imgWidth, imgHeight);
		else if (no == 1)
			write_BMP_RGBA_to_RGB("output5x5.bmp", outputImage, imgWidth, imgHeight);
		else if (no == 2)
			write_BMP_RGBA_to_RGB("output7x7.bmp", outputImage, imgWidth, imgHeight);
	}
	else{
		if (no == 0)
			write_BMP_RGBA_to_RGB("output1x3.bmp", outputImage, imgWidth, imgHeight);
		else if (no == 1)
			write_BMP_RGBA_to_RGB("output1x5.bmp", outputImage, imgWidth, imgHeight);
		else if (no == 2)
			write_BMP_RGBA_to_RGB("output1x7.bmp", outputImage, imgWidth, imgHeight);
	}
}