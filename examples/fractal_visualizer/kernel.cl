typedef unsigned char  ebmpBYTE;

__kernel void set(__global ebmpBYTE* Blue)//, __global ebmpBYTE* Red)//, __global ebmpBYTE* Red, __global ebmpBYTE* Green)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int size = get_global_size(0);
	int iteration = 0;
	int max_iteration = 255;
    float limit = 2;
			float x = 0, y = 0;
			float x0 = ((float) gidx) * 3.5 / size - 2.5;
			float y0 = ((float) gidy) * 2 / size - 1;
			  
            while ( x*x + y*y < limit*limit  &&  iteration < max_iteration )
			  {
				float xtemp = x*x - y*y + x0;
				y = 2*x*y + y0;
					
				x = xtemp;

				iteration++;
			  }
            /*(
            if (iteration >= max_iteration)
            {
                x = 0; y = 0;
                iteration = 0;
                limit = 1.75;
            while ( x*x + y*y < limit*limit  &&  iteration < max_iteration )
			  {
				float xtemp = x*x - y*y + x0;
				y = 2*x*y + y0;
					
				x = xtemp;

				iteration++;
			  }
            }*/
			Blue[gidx*size + gidy] = (iteration*255)/max_iteration;
			//Red[gidx*size + gidy] = (iteration*255)/max_iteration;
			//Green[gidx*size + gidy] = (iteration*255)/max_iteration;
}