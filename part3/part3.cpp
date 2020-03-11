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

float avgLumi(std::vector<cl_float> x, int pix) {
	float res = 0.0;
	for (int i = 0; i < pix; i++) {
		res += x[i];
	}

	//divide by the pixel
	res = res / pix;
	return res;
}

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
	cl::Image2D inputImgBuffer, inputImgBufferO, outputImgBuffer, imgBlackenBuffer;
	cl_int no;
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
		if(!build_program(&program, &context, "part3_kernel.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create command queue
		queue = cl::CommandQueue(context, device);
		
		// read input image
		if (c == 1) {
			inputImage = read_BMP_RGB_to_RGBA("bunnycity1.bmp", &imgWidth, &imgHeight);
			no = 1;
		}
		else {
			inputImage = read_BMP_RGB_to_RGBA("bunnycity2.bmp", &imgWidth, &imgHeight);
			no = 2;
		}
		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		int wnh = imgWidth * imgHeight;
		outputImage = new unsigned char[imageSize];

		// image format
		imgFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		//****************************************start for 1b****************************************
		// set offset and globalsize
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;
		
		//declare lumi
		float lumi = 0.0;
		while (true) {
			std::cout << "Please enter the average lumi you want (default: -1): ";
			std::cin >> lumi;
			if (lumi == -1) {
				break;
			}
			if (lumi <= 255 && lumi >= 0) {
				lumi /= 255;
				break;
			}
			else {
				std::cout << "Please enter a value between 0 ~ 255 (-1 to use default)" << std::endl;
			}
		}
		std::cin.clear();
		std::cin.ignore(1000, '\n');
		
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		
		//set kernel and arg
		kernel = cl::Kernel(program, "blacken");
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, no);
		kernel.setArg(3, lumi);
		
		//enqueue kernel
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		//read for image
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		write_BMP_RGBA_to_RGB("1bBlur.bmp", outputImage, imgWidth, imgHeight);
		std::cout << "1b.bmp generated" << std::endl;
		std::cout << "=====================" << std::endl;
		//ROW AND COLUMN
		//****************************************START OF 1C****************************************
		//****************************************HORI****************************************
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
		std::cin.clear();
		std::cin.ignore(1000, '\n');
		
		// create a kernel
		kernel = cl::Kernel(program, "simple_horiz");

		//reset input/output imgbuffer
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, no);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		write_BMP_RGBA_to_RGB("1cHori.bmp", outputImage, imgWidth, imgHeight);
		
		std::cout << "1cHori.bmp generated" << std::endl;

		//****************************************VERT****************************************
		//reset input/output buffer
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		kernel = cl::Kernel(program, "simple_vert");
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, no);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		write_BMP_RGBA_to_RGB("1cVert.bmp", outputImage, imgWidth, imgHeight);

		std::cout << "1cVert.bmp generated" << std::endl;

		//****************************************start for bloom****************************************
		//xor inputimage and 1cVert to get output image
		//reset input/output buffer
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		inputImgBufferO = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		kernel = cl::Kernel(program, "bloom");
		kernel.setArg(0, inputImgBufferO);
		kernel.setArg(1, inputImgBuffer);
		kernel.setArg(2, outputImgBuffer);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		write_BMP_RGBA_to_RGB("bloom.bmp", outputImage, imgWidth, imgHeight);

		std::cout << "bloom.bmp generated" << std::endl;

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