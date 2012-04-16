//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_STUBS_REMOTE_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_STUBS_REMOTE_OBJECT_HPP

#include <boost/type_traits/is_void.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/remote_object/server/remote_object.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a stubs#remote_object class is the client side
    // representation of a \a server#remote_object component
    struct remote_object : stub_base<server::remote_object>
    {
    public:
        template <typename F>
        static lcos::future<typename F::result_type>
        apply_async(naming::id_type const & target_id, F f)
        {
            typedef typename
                server::remote_object_apply_action1<
                    F
                >
                action_type;
            using namespace boost::archive::detail::extra_detail;
            init_guid<action_type>::g.initialize();
            return hpx::async<action_type>(target_id, boost::move(f));
        }

        template <typename F>
        static typename F::result_type
        apply(naming::id_type const & target_id, F f)
        {
            return apply_async(target_id, boost::move(f)).get();
        }

        template <typename F>
        static lcos::future<void>
        set_dtor_async(naming::id_type const & target_id, F const & f)
        {
            typedef server::remote_object::set_dtor_action action_type;
            return hpx::async<action_type>(target_id, f);
        }

        template <typename F>
        static void
        set_dtor(naming::id_type const & target_id, F const & f)
        {
            set_dtor_async(target_id, f).get();
        }
    };
}}}
#endif
