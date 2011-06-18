//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PARAMETER_OCT_19_2009_0834AM)
#define HPX_COMPONENTS_PARAMETER_OCT_19_2009_0834AM

#include <vector>

#include <boost/config.hpp>

#if !defined(HPX_SMP_AMR3D_NUM_EQUATIONS)
    #define HPX_SMP_AMR3D_NUM_EQUATIONS 5
#endif

#if !defined(HPX_SMP_AMR3D_MAX_LEVELS)
    #define HPX_SMP_AMR3D_MAX_LEVELS 20
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{

typedef double double_type;

namespace detail {

struct HPX_COMPONENT_EXPORT parameter
{
    double_type lambda;
    std::size_t allowedl;
    std::size_t num_rows;
    std::size_t loglevel;
    double_type output;
    std::size_t output_stdout;
    std::size_t nt0;
    std::size_t nx[HPX_SMP_AMR3D_MAX_LEVELS];
    double refine_level[HPX_SMP_AMR3D_MAX_LEVELS];
    double_type minx0;
    double_type maxx0;
    double_type dx0;
    double_type dt0;
    double_type ethreshold;
    double_type R0;
    double_type delta;
    double_type amp;
    double_type amp_dot;
    double_type eps;
    std::size_t output_level;
    std::size_t granularity;
    std::size_t time_granularity;
    std::size_t gw;
    std::vector<std::size_t> rowsize,level_row;
    std::vector<std::size_t> level_begin, level_end;
    std::vector<double_type> min;
    std::vector<double_type> max;
};

} // detail

struct HPX_COMPONENT_EXPORT parameter 
{
    typedef detail::parameter value_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef value_type* pointer;
    typedef value_type const* const_pointer;

    boost::shared_ptr<value_type> p;

    parameter() : p(new value_type) {}

    pointer operator->()
    { return p.get(); }
    
    const_pointer operator->() const
    { return p.get(); }
    
    reference operator*()
    { return *p; }
    
    const_reference operator*() const
    { return *p; }
};

///////////////////////////////////////////////////////////////////////////////
}}}

#include <examples/smp_amr3d/serialize_parameter.hpp>

#endif 

