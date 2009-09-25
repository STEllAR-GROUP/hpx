//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM

#if defined(__cplusplus)
#include <boost/serialization/serialization.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
struct stencil_data
{
    size_t max_index_;   // overall number of data points
    size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
    size_t timestep_;    // current time step
    size_t level_;    // refinement level
    double value_;            // current value

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
