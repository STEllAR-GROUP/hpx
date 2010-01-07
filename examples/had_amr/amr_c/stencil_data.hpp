//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM

#if defined(__cplusplus)
#include <boost/serialization/serialization.hpp>
#endif

#include <hpx/c/types.h>

///////////////////////////////////////////////////////////////////////////////
struct stencil_data
{
    size_t max_index_;   // overall number of data points
    size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
    double timestep_;    // current time step
    size_t level_;    // refinement level
    double value_;            // current value
    double x_;      // x coordinate value
    bool refine_;     // whether to refine
    gid overwrite_; // gid of overwrite stencil point
    gid right_;     // gid of right stencil point
    gid left_;      // gid of left stencil point
    gid reference_;  // coarser gid reference
    size_t right_alloc_;
    size_t left_alloc_;
    size_t overwrite_alloc_;
    size_t reference_alloc_;

#if defined(__cplusplus)
private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & max_index_ & index_ & timestep_ & value_; 
    }
#endif
};

typedef struct stencil_data stencil_data;

#endif
