//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MANDELBROT_JANUARY_23_2009_1020AM)
#define HPX_MANDELBROT_JANUARY_23_2009_1020AM

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>

namespace mandelbrot
{
    ///////////////////////////////////////////////////////////////////////////
    struct data
    {
        data()
          : x_(0), y_(0), sizex_(0), sizey_(0), iterations_(0),
            minx_(0), maxx_(1.0), miny_(0), maxy_(1.0), debug_(false)
        {}

        data(int x, int y, int sizex, int sizey, int iterations,
                double minx = 0.0, double maxx = 1.0, 
                double miny = 0.0, double maxy = 1.0)
          : x_(x), y_(y), sizex_(sizex), sizey_(sizey), iterations_(iterations),
            minx_(minx), maxx_(maxx), miny_(miny), maxy_(maxy), debug_(false)
        {}

        boost::uint32_t x_;
        boost::uint32_t y_;
        boost::uint32_t sizex_;
        boost::uint32_t sizey_;
        boost::uint32_t iterations_;

        double minx_;
        double maxx_;
        double miny_;
        double maxy_;

        bool debug_;

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & x_ & y_ & sizex_ & sizey_ & iterations_; 
            ar & minx_ & maxx_ & miny_ & maxy_; 
            ar & debug_;
        }
    };

    struct result
    {
        result()
          : x_(0), y_(0), iterations_(0)
        {}

        result(int x, int y, int iterations)
          : x_(x), y_(y), iterations_(iterations)
        {}

        boost::uint32_t x_;
        boost::uint32_t y_;
        boost::uint32_t iterations_;

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & x_ & y_ & iterations_; 
        }
    };

}

///////////////////////////////////////////////////////////////////////////////
HPX_COMPONENT_EXPORT 
    mandelbrot::result mandelbrot_func(mandelbrot::data const& data);

typedef hpx::actions::plain_result_action1<
    mandelbrot::result, mandelbrot::data const&, mandelbrot_func
> mandelbrot_action;

#endif
