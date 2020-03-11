__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void bnw(read_only image2d_t src_image,
					write_only image2d_t dst_image,
					__global float *returnValue,
					__global float *clr) {

	/* Get work-item’s row and column position */
	int column = get_global_id(0); 
	int row = get_global_id(1);

	/* Accumulated pixel value */
	float4 sum = (float4)(0.0);

	/* Filter's current index */
	int filter_index =  0;

	float4 pixel;
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
   
	pixel = read_imagef(src_image, sampler, coord);
   
	/* Acculumate weighted sum */ 		
	pixel.x *= 0.299;
	pixel.y *= 0.587;
	pixel.z *= 0.114;

	sum += pixel.x + pixel.y + pixel.z;

	pixel.xyz = sum.xyz;
	
	/* Write new pixel value to output */
	// coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);

	int width = get_image_width(src_image);
	int index  = width * coord.y + coord.x;
	returnValue[index] = (pixel.x + pixel.y + pixel.z) / 3;

	pixel = read_imagef(src_image, sampler, coord);
	clr[index] = (pixel.x + pixel.y + pixel.z) / 3;
}

__kernel void reduction_vector(__global float4* data,
      __local float4* partial_sums, __global float* output) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      output[get_group_id(0)] = dot(partial_sums[0], (float4)(1.0f));
   }
}