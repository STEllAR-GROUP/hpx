__kernel void set(__global float* raws)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int size = get_global_size(0);
	int iteration = 0;
	int max_iteration = 10000;
    float limit = 0;
    int count = 0;
    int max_count = 10000;
    float inc = 0.00025;
			float x0 = ((float) gidx) * 3.5 / size - 2.5;
			float y0 = ((float) gidy) * 2 / size - 1;
			  while (count < max_count)
              {
			    float x = 0, y = 0;
                limit += inc;
                while ( x*x + y*y < limit*limit &&  iteration < max_iteration )
			      {
				    float xtemp = x*x - y*y + x0;
				    y = 2*x*y + y0;
					
				    x = xtemp;

				    iteration++;
			      }
                count+= iteration;
                iteration = 0;
              }
			raws[gidx*size + gidy] = limit*90;//(float)((int)limit)%255;//iteration/8;
			//Red[gidx*size + gidy] = (float)(((int)limit)/255)%255;//255-((limit-inc) * 255/max_limit);
			//Green[gidx*size + gidy] = (iteration*255)/max_iteration;
}