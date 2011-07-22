//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2011 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <vector>

#include <hpx/lcos/mutex.hpp>
#include <boost/serialization/valarray.hpp>

#include <examples/dataflow/parameter.hpp>
#include <examples/dataflow/array1d.hpp>

namespace hpx { namespace components { namespace amr 
{

///////////////////////////////////////////////////////////////////////////////
struct stencil_data 
{
    stencil_data() 
      : max_index_(0), index_(0), timestep_(0)
    {}
    ~stencil_data() {}

    stencil_data(stencil_data const& rhs)
      : max_index_(rhs.max_index_), index_(rhs.index_), 
        timestep_(rhs.timestep_),
        value_(rhs.value_)
    {
        // intentionally do not copy mutex, new copy will have it's own mutex
    }

    stencil_data& operator=(stencil_data const& rhs)
    {
        if (this != &rhs) {
            max_index_ = rhs.max_index_;
            index_ = rhs.index_;
            timestep_ = rhs.timestep_;
            value_ = rhs.value_;
            // intentionally do not copy mutex, new copy will have it's own mutex
        }
        return *this;
    }

    hpx::lcos::mutex mtx_;    // lock for this data block

    std::size_t max_index_;   // overall number of data points
    std::size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
    double_type timestep_;    // current time step
    array1d<double> value_;    // current value

private:
    // serialization support
    friend class boost::serialization::access;

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar & max_index_ & index_ & timestep_;
        value_.do_save(ar,0,2);
    } 

    template<class Archive>
    void load(Archive & ar, const unsigned int version) 
    {
        ar & max_index_ & index_ & timestep_;
        value_.do_load(ar);
    } 
};

}}}

#endif

