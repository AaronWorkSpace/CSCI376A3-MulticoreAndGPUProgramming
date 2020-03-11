__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

// 3x3 Blurring filter
__constant float BlurringFilter9[9] = {
										0.077847, 0.123317, 0.077847,
										0.123317, 0.195346, 0.123317,
										0.077847, 0.123317, 0.077847};

// 5x5 Blurring filter
__constant float BlurringFilter25[25] = {
										0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
										0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
										0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
										0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
										0.003765, 0.015019, 0.023792, 0.015019, 0.003765};

// 7x7 Blurring filter
__constant float BlurringFilter49[49] = {
										0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036,
										0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
										0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
										0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291,
										0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
										0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
										0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036};

// 1x3 / 3x1
__constant float BlurringFilter3x1[3] = {0.27901, 0.44198, 0.27901};

// 1x5 / 5x1
__constant float BlurringFilter5x1[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};

// 1x7 / 7x1
__constant float BlurringFilter7x1[7] = {0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598};

//part 1a
__kernel void simple_conv(read_only image2d_t src_image,
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
	
	if(no == 0){
		/* Iterate over the rows */
		for(int i = -1; i < 2; i++) {
			coord.y =  row + i;

			/* Iterate over the columns */
			for(int j = -1; j < 2; j++) {
				coord.x = column + j;

				/* Read value pixel from the image */ 		
				pixel = read_imagef(src_image, sampler, coord);
				/* Acculumate weighted sum */ 		
				sum.xyz += pixel.xyz * BlurringFilter9[filter_index++];
			}
		}
	}
	else if(no == 1){
		/* Iterate over the rows */
		for(int i = -1; i < 4; i++) {
			coord.y =  row + i;

			/* Iterate over the columns */
			for(int j = -1; j < 4; j++) {
				coord.x = column + j;

				/* Read value pixel from the image */ 		
				pixel = read_imagef(src_image, sampler, coord);
				/* Acculumate weighted sum */ 		
				sum.xyz += pixel.xyz * BlurringFilter25[filter_index++];
			}
		}
	}

	else if(no == 2){
		/* Iterate over the rows */
		for(int i = -1; i < 6; i++) {
			coord.y =  row + i;

			/* Iterate over the columns */
			for(int j = -1; j < 6; j++) {
				coord.x = column + j;

				/* Read value pixel from the image */ 		
				pixel = read_imagef(src_image, sampler, coord);
				/* Acculumate weighted sum */ 		
				sum.xyz += pixel.xyz * BlurringFilter49[filter_index++];
			}
		}
	}
	/* Write new pixel value to output */
	coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);
}

//part 1b
__kernel void simple_row(read_only image2d_t src_image,
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

	if(no == 0){
		for(int i = -1 ; i <= 1; i++){
			coord.x = column + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter3x1[filter_index++];
		}
	}
	else if(no == 1){
		for(int i = -1 ; i < 4; i++){
			coord.x = column + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter5x1[filter_index++];
		}
	}

	else if(no == 2){
		for(int i = -1 ; i < 6; i++){
			coord.x = column + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter7x1[filter_index++];
		}
	}
	/* Write new pixel value to output */
	coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);
}

__kernel void simple_column(read_only image2d_t src_image,
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
	
	if(no == 0){
		for(int i = -1 ; i <= 1; i++){
			coord.y = row + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter3x1[filter_index++];
		}
	}
	else if(no == 1){
		for(int i = -1 ; i < 4; i++){
			coord.y = row + i;
			pixel = read_imagef(src_image, sampler, coord);
			sum.xyz += pixel.xyz * BlurringFilter5x1[filter_index++];
		}
	}

	else if(no == 2){
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