//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_NEW_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_NEW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/components/remote_object/stubs/remote_object.hpp>
#include <hpx/util/static.hpp>

namespace hpx { namespace components
{
    namespace remote_object
    {
        naming::id_type
        new_impl(naming::id_type const & target_id, util::function<void(void**)> f);

        typedef
            actions::plain_result_action2<
                naming::id_type
              , naming::id_type const &
              , util::function<void(void**)>
              , &new_impl
            >
            new_impl_action;

        template <typename T, typename A0 = void, typename Enable = void>
        struct ctor_fun;

        template <typename T>
        struct ctor_fun<T>
        {
            typedef void result_type;

            void operator()(void ** p) const
            {
                T * t = new T();
                *p = t;
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {}
        };

        template <typename T, typename A0>
        struct ctor_fun<T, A0>
        {
            typedef void result_type;

            ctor_fun() {}
            ctor_fun(A0 const & a0) : a0(a0) {}

            void operator()(void ** p) const
            {
                T * t = new T(a0);
                *p = t;
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & a0;
            }

            A0 a0;
        };

        template <typename T, typename F>
        struct invoke_apply_fun
        {
            invoke_apply_fun() {}
            invoke_apply_fun(F f) : f(f) {}

            typedef void result_type;

            void operator()(void ** p) const
            {
                f(*reinterpret_cast<T *>(*p));
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & f;
            }

            F f;
        };
    }

    template <typename T>
    struct object
    {
        object() {}
        explicit object(naming::id_type const & gid) : gid_(gid) {}
        naming::id_type gid_;

        template <typename F>
        lcos::promise<void>
        operator<=(F f)
        {
            return stubs::remote_object::apply_async(gid_, remote_object::invoke_apply_fun<T, F>(f));
        }
    };

    template <typename T>
    lcos::promise<object<T> >
    new_(naming::id_type const & target_id)
    {
        util::function<void(void**)> f = remote_object::ctor_fun<T>();
        return lcos::eager_future<remote_object::new_impl_action, object<T> >(target_id, target_id, f);
    }

    template <typename T, typename A0>
    lcos::promise<object<T> >
    new_(naming::id_type const & target_id, A0 const & a0)
    {
        util::function<void(void**)> f = remote_object::ctor_fun<T, A0>(a0);
        return lcos::eager_future<remote_object::new_impl_action, object<T> >(target_id, target_id, f);
    }
}}

namespace hpx { namespace traits
{
    template <typename T>
    struct promise_remote_result<hpx::components::object<T> >
    {
        typedef ::hpx::naming::id_type type;
    };
}}

namespace hpx { namespace components
{
    template <typename T>
    struct component_type_database<
        hpx::lcos::base_lco_with_value<hpx::components::object<T>, hpx::naming::id_type>
    >
    {
        static component_type get()
        {
            return
                component_type_database<
                    lcos::base_lco_with_value<
                        hpx::naming::id_type
                      , hpx::naming::id_type
                    >
                >::get();
        }
        static void set(component_type t)
        {
            component_type_database<
                lcos::base_lco_with_value<
                    hpx::naming::id_type
                  , hpx::naming::id_type
                >
            >::set(t);
        }
    };
}}

#endif
