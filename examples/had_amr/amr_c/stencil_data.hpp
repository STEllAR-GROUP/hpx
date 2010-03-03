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
  had_double_type phi0;
  had_double_type phi1;
 
#if defined(__cplusplus)
private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & phi0 & phi1;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////
struct stencil_data 
{
    ~stencil_data() {}

    size_t max_index_;   // overall number of data points
    size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
    double timestep_;    // current time step
    int cycle_; // counts the number of subcycles
    size_t level_;    // refinement level
    nodedata value_;            // current value
    double x_;      // x coordinate value
    int iter_;      // rk subcycle indicator
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
