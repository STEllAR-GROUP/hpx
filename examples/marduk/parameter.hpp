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
    std::size_t output_level;
    std::size_t nt0;
    std::size_t nx0;
    std::size_t ny0;
    std::size_t nz0;
    std::size_t shadow;
    double refine_level[HPX_SMP_AMR3D_MAX_LEVELS];
    double_type minx0;
    double_type maxx0;
    double_type miny0;
    double_type maxy0;
    double_type minz0;
    double_type maxz0;
    double_type h;
    double_type ethreshold;
    double_type minefficiency;
    std::size_t num_px_threads;
    std::size_t refine_every;
    std::size_t ghostwidth;
    std::size_t bound_width;
    std::size_t clusterstyle;
    std::size_t mindim;
    std::size_t refine_factor;
    
    std::vector<int> gr_sibling;
    std::vector<double_type> gr_t;
    std::vector<double_type> gr_minx;
    std::vector<double_type> gr_miny;
    std::vector<double_type> gr_minz;
    std::vector<double_type> gr_maxx;
    std::vector<double_type> gr_maxy;
    std::vector<double_type> gr_maxz;
    std::vector<std::size_t> gr_nx;
    std::vector<std::size_t> gr_ny;
    std::vector<std::size_t> gr_nz;
    std::vector<std::size_t> gr_proc;
    std::vector<double_type> gr_h;
    std::vector<int> gr_alive;
    std::vector<std::size_t> levelp;
    std::vector<std::size_t> item2gi;

    std::vector<std::size_t> rowsize,level_row;
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

#include <examples/marduk/serialize_parameter.hpp>

#endif 

