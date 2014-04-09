//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <examples/mini_ghost/write_grid.hpp>

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

namespace mini_ghost {
    template <typename Real>
    void write_grid(grid<Real> const & g, std::string filename, std::size_t z)
    {
        filename += "_";
        filename += boost::lexical_cast<std::string>(z);
        filename += ".ppm";

        std::ofstream outfile(filename.c_str());
        if(!outfile) std::cerr << "Cannot open file ...\n";
        outfile << "P6 " << g.nx_ << " " << g.ny_ << " 255\n";
        for(std::size_t y = 0; y < g.ny_; ++y)
        {
            for(std::size_t x = 0; x < g.nx_; ++x)
            {
                double tmp = g(x, y, z);
                int r = 0;
                int g = 0;
                int b = 0;

                if(tmp >= 0)
                {
                    if(tmp < 0.25)
                    {
                        g = 254 * static_cast<int>(tmp / 0.25);
                        b = 254;
                    }
                    else if(tmp < 0.5)
                    {
                        g = 254;
                        b = 254 - 254 * static_cast<int>((tmp - 0.25) / 0.25);
                    }
                    else if(tmp < 0.75)
                    {
                        r = 254 * static_cast<int>((tmp - 0.5) / 0.25);
                        g = 254;
                    }
                    else if(tmp < 1.0)
                    {
                        r = 254;
                        g = 254 - 254 * static_cast<int>((tmp - 0.75) / 0.25);
                    }
                    else
                    {
                        r = 254;
                        g = 254;
                        b = 254;
                    }
                }

                outfile << (char)r << (char)g << (char)b;
            }
        }
        outfile.flush();
        outfile.close();
    }

    template void write_grid(grid<float> const &, std::string, std::size_t);
    template void write_grid(grid<double> const &, std::string, std::size_t);
}
