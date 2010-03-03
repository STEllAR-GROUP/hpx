//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PROPS_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_PROPS_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/props.hpp"

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a props class is the client side representation of a
    /// specific \a server#props component
    class props
      : public client_base<props, stubs::props>
    {
        typedef client_base<props, stubs::props> base_type;

    public:
        props() {}

        /// Create a client side representation for the existing
        /// \a server#props instance with the given global id \a gid.
        props(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the props
        int init(int order)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init(gid_, order);
        }

        int color(int d)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::color(gid_, d);
        }

        double incr(double d)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::incr(gid_, d);
        }
    };
    
}}

#endif
