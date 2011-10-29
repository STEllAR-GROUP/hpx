//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_STUBS_REMOTE_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_STUBS_REMOTE_OBJECT_HPP

#include <boost/type_traits/is_void.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/remote_object/server/remote_object.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a stubs#remote_object class is the client side
    // representation of a \a server#remote_object component
    struct remote_object : stub_base<server::remote_object>
    {
    private:
        template <typename F>
        static lcos::promise<typename F::result_type>
        apply_async_invoke(
            naming::id_type const & target_id
          , F const & f
          , boost::mpl::false_
        )
        {
            typedef typename F::result_type result_type;
            typedef typename
                server::remote_object_apply_action<
                    result_type
                >
                action_type;
            using namespace boost::archive::detail::extra_detail;
            init_guid<action_type>::g.initialize();
            return lcos::eager_future<action_type>(target_id, f, 0);
        }

        template <typename F>
        static lcos::promise<void>
        apply_async_invoke(
            naming::id_type const & target_id
          , F const & f
          , boost::mpl::true_
        )
        {
            typedef typename
                server::remote_object_apply_action<void>
                action_type;
            return lcos::eager_future<action_type>(target_id, f, 0);
        }

    public:
        template <typename F>
        static lcos::promise<typename F::result_type>
        apply_async(naming::id_type const & target_id, F const & f)
        {
            return
                apply_async_invoke(
                    target_id
                  , f
                  , typename boost::is_void<typename F::result_type>::type()
                );
        }

        template <typename F>
        static typename F::result_type
        apply(naming::id_type const & target_id, F const & f)
        {
            return apply_async(target_id, f).get();
        }

        template <typename F>
        static lcos::promise<void>
        set_dtor_async(naming::id_type const & target_id, F const & f)
        {
            typedef server::remote_object::set_dtor_action action_type;
            return lcos::eager_future<action_type>(target_id, f, 0);
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
