//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_STUB_BASE_CLIENT_OCT_31_2008_0441PM)
#define HPX_COMPONENTS_STUBS_STUB_BASE_CLIENT_OCT_31_2008_0441PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ServerComponent>
    class stub_base 
    {
    public:
        stub_base(applier::applier& appl)
          : appl_(appl)
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously create a new instance of a distributing_factory
        static lcos::future_value<naming::id_type>
        create_async(applier::applier& appl, naming::id_type const& gid, 
            std::size_t count = 1)
        {
            return stubs::runtime_support::create_component_async(
                appl, gid, get_component_type<ServerComponent>(), count);
        }

        lcos::future_value<naming::id_type>
        create_async(naming::id_type const& gids, std::size_t count = 1)
        {
            return create_async(appl_, gid, count);
        }

        /// Create a new instance of an simple_accumulator
        static naming::id_type 
        create(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, std::size_t count = 1)
        {
            return stubs::runtime_support::create_component(self, appl, 
                gid, get_component_type<ServerComponent>(), count);
        }

        naming::id_type 
        create(threads::thread_self& self, naming::id_type const& gid,
            std::size_t count = 1)
        {
            return create(self, appl_, gid, count);
        }

        /// Delete an existing component
        static void
        free(applier::applier& appl, naming::id_type const& gid)
        {
            stubs::runtime_support::free_component(appl, 
                get_component_type<ServerComponent>(), gid);
        }

        void free(naming::id_type const& gid)
        {
            free(appl_, gid);
        }

    protected:
        applier::applier& appl_;
    };

}}}

#endif

