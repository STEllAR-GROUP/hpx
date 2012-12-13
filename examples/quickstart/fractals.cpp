////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Andrew Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <iostream>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include <external/EasyBMP/EasyBMP.h>
#include <external/EasyBMP/EasyBMP.cpp>

const int sizeY = 256;
const int sizeX = sizeY;

//[fib_action
// forward declaration of the Fibonacci function
int fractals(float x0, float y0, int max_iteration);

// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(fractals, fractals_action);


///////////////////////////////////////////////////////////////////////////////
//[fib_func
int fractals(float x0, float y0, int max_iteration)
{
    
            float x = 0, y = 0;
            int iteration = 0;
              while ( x*x + y*y < 2*2  &&  iteration < max_iteration )
              {
                float xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                    
                x = xtemp;

                iteration++;
              }
    return iteration;   // wait for the Futures to return their values
}
//]
//]


///////////////////////////////////////////////////////////////////////////////
//[fib_hpx_main
int hpx_main()
{
    BMP SetImage;
    SetImage.SetBitDepth(24);
    SetImage.SetSize(sizeX * 2,sizeY);
    hpx::util::high_resolution_timer t;
    {
        int max_iteration = 255;
        using hpx::lcos::future;
        using hpx::async;
        using namespace std;
        using hpx::wait_all;
    vector<fractals_action> fAct;//[sizeX * sizeY];
    vector<future<int>> iteration;
    iteration.reserve(sizeX*sizeY);
    std::cout << "Initial setup completed in " << t.elapsed() << "s. Initializing and running futures...\n";
    t.restart();
    for (int i = 0; i < sizeX; i++)
        for (int j = 0; j < sizeY; j++)
        {
            hpx::naming::id_type const locality_id = hpx::find_here();
            float x0 = (float)i * 3.5 / (float)sizeX - 2.5;
            float y0 = (float)j * 2.0 / (float)sizeY - 1.0;
            //int it = iteration.get();
            fractals_action temp;
            iteration.push_back(async(temp, locality_id, x0, y0, max_iteration));
            fAct.push_back(temp);
        }
    wait_all(iteration);
    //int total = 0;
    std::cout << sizeX*sizeY << " calculations run in " << t.elapsed() << "s. Transferring from futures to general memory...\n";
    t.restart();
    for (int i = 0; i < sizeX; i++)
        for (int j = 0; j < sizeY; j++)
        {
            int it = iteration[i*sizeX + j].get();
            //total += it;
            for (int k = 0; k < 2; k++)
            {
            RGBApixel pix;
            pix.Blue = (it*255)/max_iteration;
            pix.Red = (it*255)/max_iteration;
            pix.Green = (it*255)/max_iteration;
            SetImage.SetPixel(i * 2 + k,j,pix);
            }
        }
    }
    std::cout << "Transfer process completed in " << t.elapsed() << "s. Writing to hard disk...\n";
    t.restart();
    SetImage.WriteToFile("out.bmp");
    
    std::cout << "Fractal image written to file \"out.bmp\" from memory in " << t.elapsed() << "s.\nInitializing shutdown process.\n";

    return hpx::finalize(); // Handles HPX shutdown
}
//]

int main()
{
    return hpx::init();
}
