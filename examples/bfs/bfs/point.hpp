//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_POINT)
#define HPX_COMPONENTS_CLIENT_POINT

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"


namespace hpx { namespace geometry
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#point class is the client side representation of all
    /// \a server#point components
    class point
        : public components::client_base<point, stubs::point>
    {
        typedef components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component)
        point()
        {}

        /// Create client side representation from a newly create component
        /// instance.
        point(naming::id_type where, double x, double y)
          : base_type(base_type::create_sync(where))    // create component
        {
            //init(x, y);   // initialize coordinates
        }

        /// Create a client side representation for the existing
        /// \a server#point instance with the given global id \a gid.
        point(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#point instance with the given \a gid
        lcos::promise<void> init_async(std::size_t objectid)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid);
        }

        void init(std::size_t objectid)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid);
        }

        //lcos::promise< std::vector<std::size_t> > traverse_async(std::size_t level, std::size_t parent)
        //{
        //    BOOST_ASSERT(gid_);
        //    return this->base_type::traverse_async(gid_,level,parent);
       // }

        //std::vector<std::size_t> traverse(std::size_t level,std::size_t parent)
        //{
        //    BOOST_ASSERT(gid_);
        //    this->base_type::traverse(gid_,level,parent);
        //}
    };
}}

#endif
