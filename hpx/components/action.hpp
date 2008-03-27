//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_ACTION_MAR_26_2008_1054AM)
#define HPX_COMPONENTS_ACTION_MAR_26_2008_1054AM

#include <cstdlib>
#include <stdexcept>

#include <boost/shared_ptr.hpp>

#include <hpx/components/component_type.hpp>
#include <hpx/exception.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    struct action_base
    {
        virtual std::size_t get_action_code() const = 0;
        virtual components::component_type get_component_type() const = 0;
    };

    typedef boost::shared_ptr<action_base> action_type;

///////////////////////////////////////////////////////////////////////////////
}}

#endif

