//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM

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
      : max_index_(0), index_(0), timestep_(0), cycle_(0), granularity(0),
        level_(0), gw_iter_(0),g_startx_(0),g_endx_(0),g_dx_(0),ghostwidth_(0)
    {}
    ~stencil_data() {}

    stencil_data(stencil_data const& rhs)
      : max_index_(rhs.max_index_), index_(rhs.index_), 
        timestep_(rhs.timestep_), cycle_(rhs.cycle_), 
        granularity(rhs.granularity), level_(rhs.level_), 
        value_(rhs.value_), x_(rhs.x_),
        gw_iter_(rhs.gw_iter_),
        g_startx_(rhs.g_startx_),g_endx_(rhs.g_endx_),g_dx_(rhs.g_dx_),ghostwidth_(rhs.ghostwidth_)
    {
        // intentionally do not copy mutex, new copy will have it's own mutex
    }

    stencil_data& operator=(stencil_data const& rhs)
    {
        if (this != &rhs) {
            max_index_ = rhs.max_index_;
            index_ = rhs.index_;
            timestep_ = rhs.timestep_;
            cycle_ = rhs.cycle_; 
            granularity = rhs.granularity;
            level_ = rhs.level_;
            value_ = rhs.value_;
            x_ = rhs.x_; 
            gw_iter_= rhs.gw_iter_; 
            g_startx_= rhs.g_startx_; 
            g_endx_= rhs.g_endx_; 
            g_dx_= rhs.g_dx_; 
            ghostwidth_= rhs.ghostwidth_; 
            // intentionally do not copy mutex, new copy will have it's own mutex
        }
        return *this;
    }

    hpx::lcos::mutex mtx_;    // lock for this data block

    size_t max_index_;   // overall number of data points
    size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
    had_double_type timestep_;    // current time step
    size_t cycle_; // counts the number of subcycles
    size_t granularity;
    size_t level_;    // refinement level
    std::vector< nodedata > value_;            // current value
    std::vector< had_double_type > x_;      // x coordinate value
    size_t gw_iter_;      // subcycle indicator
    had_double_type g_startx_;
    had_double_type g_endx_;
    had_double_type g_dx_;
    size_t ghostwidth_;

private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & max_index_ & index_ & timestep_ & cycle_ & granularity & level_ & value_;
        ar & x_ & gw_iter_ & g_startx_ & g_endx_ & g_dx_ & ghostwidth_;
    }
};

typedef struct stencil_data stencil_data;

#endif
