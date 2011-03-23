//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_NBODY_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_NBODY_STENCIL_DATA_NOV_10_2008_0719PM

#include <boost/serialization/serialization.hpp>
#include <vector>

#include <hpx/c/types.h>
#include <hpx/lcos/mutex.hpp>

#include "../had_config.hpp"

struct nodedata
{
    had_double_type phi[2][num_eqns];

private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & phi;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct stencil_data 
{
    stencil_data() 
      : row(0), column(0) 
    {}
    ~stencil_data() {}

    stencil_data(stencil_data const& rhs)
      : node_type(rhs.node_type), x(rhs.x), y(rhs.y), z(rhs.z),
        ax(rhs.ax),ay(rhs.ay),az(rhs.az),
        vx(rhs.vx),vy(rhs.vy),vz(rhs.vz),
        row(rhs.row),column(rhs.column)
    {
        // intentionally do not copy mutex, new copy will have it's own mutex
    }

    stencil_data& operator=(stencil_data const& rhs)
    {
        if (this != &rhs) {
            node_type = rhs.node_type;
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
            ax = rhs.ax;
            ay = rhs.ay;
            az = rhs.az;
            vx = rhs.vx;
            vy = rhs.vy;
            vz = rhs.vz;
            row = rhs.row;
            column = rhs.column;
            // intentionally do not copy mutex, new copy will have it's own mutex
        }
        return *this;
    }

    hpx::lcos::mutex mtx_;    // lock for this data block

    std::vector<int> node_type;
    std::vector<had_double_type> x,y,z;
    std::vector<had_double_type> ax,ay,az;
    std::vector<had_double_type> vx,vy,vz;
    std::size_t row;
    std::size_t column;
    
//     int num_particles;
//     std::vector<int> node_type;
//     std::vector<had_double_type> x,y,z;
//     std::vector<had_double_type> ax,ay,az;
//     std::vector<had_double_type> vx,vy,vz;
//        std::size_t row;
//        std::size_t column;

private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & node_type;
        ar & x & y & z & ax & ay & az;
        ar & vx & vy & vz & row & column;
    }
};

typedef struct stencil_data stencil_data;

#endif
