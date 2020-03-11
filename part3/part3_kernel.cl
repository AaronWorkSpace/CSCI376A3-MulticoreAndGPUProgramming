__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

// 1x3 / 3x1
__constant float BlurringFilter3x1[3] = {0.27901, 0.44198, 0.27901};

// 1x5 / 5x1
__constant float BlurringFilter5x1[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};

// 1x7 / 7x1
__constant float BlurringFilter7x1[7] = {0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598};

//set constant avg
//ASK LECTURER IF USE COLORED OR GRAYSCALE AVG
__constant float bnw1 = 101.449 / 255;
__constant float bnw2 = 116.563 / 255;

//part 3c (hori)
__kernel void simple_horiz(read_only image2d_t src_image,
					write_only image2d_t dst_image,
					int no) {
	
	/* Get work-item’s row and column position */
	int column = get_global_id(0); 
	int row = get_global_id(1);

	/* Accumulated pixel value */
	float4 sum = (float4)(0.0);

	/* Filter's current index */
	int filter_index =  0;

	int2 coord;
	float4 pixel;

	coord.y = row;

	if(no == 1){
		for(int i = -1 ; i <= 1; i++){
			coord.x = column + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter3x1[filter_index++];
		}
	}
	else if(no == 2){
		for(int i = -1 ; i < 4; i++){
			coord.x = column + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter5x1[filter_index++];
		}
	}
	else if(no == 3){
		for(int i = -1 ; i < 6; i++){
			coord.x = column + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter7x1[filter_index++];
		}
	}
	/* Write new pixel value to output */
	coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);

	/* Write new pixel value to output */
	coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);
}

//part c (vert)
__kernel void simple_vert(read_only image2d_t src_image,
					write_only image2d_t dst_image,
					int no) {
	/* Get work-item’s row and column position */
	int column = get_global_id(0); 
	int row = get_global_id(1);

	/* Accumulated pixel value */
	float4 sum = (float4)(0.0);

	/* Filter's current index */
	int filter_index =  0;

	int2 coord;
	float4 pixel;
	coord.x = column;

	if(no == 1){
		for(int i = -1 ; i <= 1; i++){
			coord.y = row + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter3x1[filter_index++];
		}
	}
	else if(no == 2){
		for(int i = -1 ; i < 4; i++){
			coord.y = row + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter5x1[filter_index++];
		}
	}
	else if(no == 3){
		for(int i = -1 ; i < 6; i++){
			coord.y = row + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter7x1[filter_index++];
		}
	}

	/* Write new pixel value to output */
	coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);
}

//part 3b
__kernel void blacken(read_only image2d_t src_image,
					write_only image2d_t dst_image,
					int image, float l) {

	/* Get work-item’s row and column position */
	int column = get_global_id(0); 
	int row = get_global_id(1);

	/* Accumulated pixel value */
	float4 sum = (float4)(0.0);
	float4 sum2 = (float4)(0.0);

	/* Filter's current index */
	int filter_index =  0;

	float4 pixel;
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
   
	pixel = read_imagef(src_image, sampler, coord);

	if(image == 0){
		if(l == -1){
			if((pixel.x + pixel.y + pixel.z) / 3 < bnw1){
				pixel.xyz = 0;
			}
		}
		else{
			if((pixel.x + pixel.y + pixel.z) / 3 < l){
				pixel.xyz = 0;
			}
		}
	}
	else{
		if(l == -1){
			if((pixel.x + pixel.y + pixel.z) / 3 < bnw2){
				pixel.xyz = 0;
			}
		}
		else{
			if((pixel.x + pixel.y + pixel.z) / 3 < l){
				pixel.xyz = 0;
			}
		}
	}
	/* Write new pixel value to output */
	write_imagef(dst_image, coord, pixel);
}

__kernel void bloom(read_only image2d_t src_imageO,
					read_only image2d_t src_imageV,
					write_only image2d_t dst_image){
	/* Get work-item’s row and column position */
	int column = get_global_id(0); 
	int row = get_global_id(1);

	/* Accumulated pixel value */
	float4 sum = (float4)(0.0);

	/* Filter's current index */
	int filter_index =  0;

	float4 pixel;
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
   
   	pixel = read_imagef(src_imageO, sampler, coord);
	sum.x = pixel.x;
	sum.y = pixel.y;
	sum.z = pixel.z;

	pixel = read_imagef(src_imageV, sampler, coord);
	
	pixel.x += sum.x;
	pixel.y += sum.y;
	pixel.z += sum.z;

	if((pixel.x + pixel.y + pixel.z) / 3 > 1.0){
		pixel.xyz = 1;
	}

	/* Write new pixel value to output */
	write_imagef(dst_image, coord, pixel);
}