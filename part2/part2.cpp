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
	int no = 0;
	for (int i = 0; i < pix; i++) {
		res += x[i];

	}
	//multiply by 255 RGB
	res *= 255;

	//divide by the pixel
	res = res / pix;
	return res;
}

int main(void) 
{
	cl::Platform platform;					// device's platform
	cl::Device device;						// device used
	cl::Context context;					// context for the device
	cl::Program program;					// OpenCL program object
	cl::Kernel kernel, kernelReduction;		// a single kernel object
	cl::CommandQueue queue;					// commandqueue for a context and device

	// declare data and memory objects
	unsigned char* inputImage;
	unsigned char* outputImage;
	int imgWidth, imgHeight, imageSize;

	//declare imageformat, image2D and buffers
	cl::ImageFormat imgFormat;
	cl::Image2D inputImgBuffer, outputImgBuffer;
	cl::Buffer outputBufferx, outputBufferc;
	
	// declare data and memory objects
	std::vector<cl_float> scalarSum, vectorSum, vectorSumc;
	cl::Buffer dataBuffer, scalarBuffer, vectorBuffer, vectorBufferc;
	cl::LocalSpaceArg localSpace;				// to create local space for the kernel
	cl::Event profileEvent;						// for profiling
	cl_ulong timeStart, timeEnd, timeTotal;
	cl_int numOfGroups;							// number of work-groups
	cl_float sum, correctSum;					// results
	size_t workgroupSize;						// work group size
	size_t kernelWorkgroupSize;                // allowed work group size for the kernel
	cl_ulong localMemorySize;					// device's local memory size

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
		if(!build_program(&program, &context, "part2_kernel.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "bnw");

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
		int rs = imgHeight * imgWidth;

		//declare vector
		std::vector<cl_float> outputx(rs);
		std::vector<cl_float> outputc(rs);

		// create image objects
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		outputBufferx = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * rs);
		outputBufferc = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * rs);

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, outputBufferx);
		kernel.setArg(3, outputBufferc);

		// set offset and globalsize
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		//enqueue kernel
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		
		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		//read the output for vector
		queue.enqueueReadBuffer(outputBufferx, CL_TRUE, 0, sizeof(cl_float) * rs, &outputx[0]);
		queue.enqueueReadBuffer(outputBufferc, CL_TRUE, 0, sizeof(cl_float) * rs, &outputc[0]);

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		//read for image
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		write_BMP_RGBA_to_RGB("output.bmp", outputImage, imgWidth, imgHeight);

		std::cout << "2a Done, output.bmp generated." << std::endl;
		
		//task 2b
		float r = avgLumi(outputx, rs);
		std::cout << "2b) Average lumi for main(bnw): " << r << std::endl;
		r = avgLumi(outputc, rs);
		std::cout << "2b) Average lumi for main(color): " << r << std::endl;
		std::vector<cl_float> data(imageSize);

		// create a kernel
		kernel = cl::Kernel(program, "reduction_vector");

		// get device information
		workgroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		localMemorySize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
		kernelWorkgroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

		// display the information
		std::cout << "Max workgroup size: " << workgroupSize << std::endl;
		std::cout << "Local memory size: " << localMemorySize << std::endl;
		std::cout << "Kernel workgroup size: " << kernelWorkgroupSize << std::endl;

		// if kernel only allows one work-item per work-group, abort
		if (kernelWorkgroupSize == 1)
			quit_program("Abort: Cannot run reduction kernel, because kernel workgroup size is 1.");

		// if allowed kernel work group size smaller than device's max workgroup size
		if (workgroupSize > kernelWorkgroupSize)
			workgroupSize = kernelWorkgroupSize;

		// ensure sufficient local memory is available
		while (localMemorySize < sizeof(float) * workgroupSize * 4)
		{
			workgroupSize /= 4;
		}

		// compute number of groups and resize vectors
		numOfGroups = rs / workgroupSize;
		vectorSum.resize(numOfGroups / 4);

		for (int i = 0; i < numOfGroups / 4; i++)
		{
			vectorSum[i] = 0.0f;
		}

		//**********************************************REDUCTION FOR BNW**********************************************
		
		// create buffers
		dataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * rs, &outputx[0]);
		vectorBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * numOfGroups / 4, &vectorSum[0]);

		cl::NDRange localSize(workgroupSize);
		kernel.setArg(0, dataBuffer);
		localSpace = cl::Local(sizeof(float) * workgroupSize * 4);
		kernel.setArg(1, localSpace);
		kernel.setArg(2, vectorBuffer);

		globalSize = imageSize / 4;

		// enqueue kernel for execution
		queue.enqueueNDRangeKernel(kernel, 0, globalSize, localSize, NULL, &profileEvent);
		
		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(vectorBuffer, CL_TRUE, 0, sizeof(cl_float) * numOfGroups / 4, &vectorSum[0]);

		sum = 0.0f;
		for (int i = 0; i < numOfGroups / 4; i++)
		{
			sum += vectorSum[i];
		}
		sum = (sum / rs) * 255;
		std::cout << "2c) Average lumi for reduction(bnw): " << sum << std::endl;

		//**********************************************DO REDUCTION FOR COLOR**********************************************
		dataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * rs, &outputc[0]);

		kernel.setArg(0, dataBuffer);
		kernel.setArg(1, localSpace);
		kernel.setArg(2, vectorBuffer);

		// enqueue kernel for execution
		queue.enqueueNDRangeKernel(kernel, 0, globalSize, localSize, NULL, &profileEvent);
		
		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(vectorBuffer, CL_TRUE, 0, sizeof(cl_float) * numOfGroups / 4, &vectorSum[0]);
		
		sum = 0.0f;
		for (int i = 0; i < numOfGroups / 4; i++)
		{
			sum += vectorSum[i];
		}
		sum = (sum / rs) * 255;
		std::cout << "2c) Average lumi for reduction(color): " << sum << std::endl;

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