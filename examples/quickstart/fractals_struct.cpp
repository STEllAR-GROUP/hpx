////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Max Moorkamp
//  Copyright (c) 2013 Bryce Adelstein-Lelbach
//  Copyright (c) 2012 Andrew Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/serialization.hpp>

#include <cstddef>
#include <memory>
#include <vector>

const std::size_t sizeY = 256;
const std::size_t sizeX = sizeY;

class FracInfo
{
  public:
    float x0;
    float y0;
    std::size_t max_iteration;

  private:
    friend class hpx::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & x0;
        ar & y0;
        ar & max_iteration;
    }
};

std::size_t fractals(std::shared_ptr<FracInfo> Info);

HPX_PLAIN_ACTION(fractals, fractals_action);

std::size_t fractals(std::shared_ptr<FracInfo> Info)
{
    float x = 0, y = 0;
    std::size_t iteration = 0;

    while (x * x + y * y < 2 * 2 && iteration < Info->max_iteration)
    {
        float xtemp = x * x - y * y + Info->x0;
        y = 2 * x * y + Info->y0;

        x = xtemp;

        iteration++;
    }

    return iteration;
}

int hpx_main()
{
    {
        int max_iteration = 255;

        std::vector<hpx::future<std::size_t> > iteration;
        iteration.reserve(sizeX * sizeY);

        fractals_action act;

        for (std::size_t i = 0; i < sizeX; i++)
            for (std::size_t j = 0; j < sizeY; j++)
            {
                hpx::id_type const here = hpx::find_here();

                float x0 = (float) i * 3.5f / (float) sizeX - 2.5f;
                float y0 = (float) j * 2.0f / (float) sizeY - 1.0f;

                std::shared_ptr<FracInfo> Info(new FracInfo);
                Info->x0 = x0;
                Info->y0 = y0;
                Info->max_iteration = max_iteration;
                iteration.push_back(hpx::async(act, here, Info));
            }

        hpx::wait_all(iteration);
    }

    return hpx::finalize();
}

int main()
{
    return hpx::init();
}

