//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM

#if defined(__cplusplus)
#include <boost/serialization/serialization.hpp>
#endif

#include <hpx/c/types.h>
#include "../had_config.hpp"

struct nodedata
{
  had_double_type phi[2][num_eqns];
 
#if defined(__cplusplus)
private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & phi;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////
struct stencil_data 
{
    ~stencil_data() {}

    size_t max_index_;   // overall number of data points
    size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
    had_double_type timestep_;    // current time step
    size_t cycle_; // counts the number of subcycles
    size_t granularity;
    size_t level_;    // refinement level
    std::vector< nodedata > value_;            // current value
    std::vector< had_double_type > x_;      // x coordinate value
    size_t iter_;      // rk subcycle indicator
    gid overwrite_; // gid of overwrite stencil point
    gid right_;     // gid of right stencil point
    gid left_;      // gid of left stencil point
    bool overwrite_alloc_;
    bool right_alloc_;
    bool left_alloc_;
    bool refine_;     // whether to refine

#if defined(__cplusplus)
private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & max_index_ & index_ & timestep_ & cycle_ & level_ & value_;
        ar & x_ & iter_ & overwrite_ & right_ & left_;
        ar & overwrite_alloc_ & right_alloc_ & left_alloc_ & refine_; 
    }
#endif
};

typedef struct stencil_data stencil_data;

#endif
