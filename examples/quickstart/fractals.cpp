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
#include <hpx/include/iostreams.hpp>

#include <vector>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include "EasyBMP/EasyBMP.h"
#include "EasyBMP/EasyBMP.cpp"

int const sizeY = 256;
int const sizeX = sizeY;

int fractals(float x0, float y0, int max_iteration);

// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(fractals, fractals_action);

///////////////////////////////////////////////////////////////////////////////
int fractals(float x0, float y0, int max_iteration)
{
    float x = 0, y = 0;
    int iteration = 0;
    while (x*x + y*y < 2*2 && iteration < max_iteration)
    {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;

        ++iteration;
    }
    return iteration;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    BMP SetImage;
    SetImage.SetBitDepth(24);
    SetImage.SetSize(sizeX * 2,sizeY);
    hpx::util::high_resolution_timer t;

    {
        using namespace std;

        using hpx::future;
        using hpx::async;
        using hpx::wait_all;

        int const max_iteration = 255;

        vector<future<int> > iteration;
        iteration.reserve(sizeX*sizeY);

        hpx::cout << "Initial setup completed in "
                  << t.elapsed()
                  << "s. Initializing and running futures...\n";
        t.restart();

        hpx::id_type const here = hpx::find_here();
        fractals_action fractal_pixel;

        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                float x0 = (float)i * 3.5f / (float)sizeX - 2.5f;
                float y0 = (float)j * 2.0f / (float)sizeY - 1.0f;

                iteration.push_back(async(fractal_pixel, here, x0, y0, max_iteration));
            }
        }
        wait_all(iteration);

        hpx::cout << sizeX*sizeY << " calculations run in "
                  << t.elapsed()
                  << "s. Transferring from futures to general memory...\n";
        t.restart();

        for (int i = 0; i < sizeX; ++i)
        {
            for (int j = 0; j < sizeY; ++j)
            {
                int it = iteration[i*sizeX + j].get();
                for (int k = 0; k < 2; ++k)
                {
                    int p = (it * 255) / max_iteration;
                    SetImage.SetPixel(i * 2 + k, j, RGBApixel(p, p, p));
                }
            }
        }
    }

    hpx::cout << "Transfer process completed in "
              << t.elapsed() << "s. Writing to hard disk...\n";
    t.restart();

    SetImage.WriteToFile("out.bmp");

    hpx::cout << "Fractal image written to file \"out.bmp\" from memory in "
              << t.elapsed() << "s.\nInitializing shutdown process.\n";

    return hpx::finalize(); // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}

