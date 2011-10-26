//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_OBJECT_HPP

#include <hpx/components/remote_object/stubs/remote_object.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/naming/address.hpp>

namespace hpx { namespace components
{
    namespace remote_object
    {
        // helper functor, casts void pointer to the right object type and
        // invokes the passed functor
        template <typename T, typename F>
        struct invoke_apply_fun
        {
            invoke_apply_fun() {}
            invoke_apply_fun(F f) : f(f) {}

            typedef typename boost::result_of<F(T &)>::type result_type;

            result_type operator()(void ** p) const
            {
                return f(*reinterpret_cast<T *>(*p));
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
        lcos::promise<
            typename boost::result_of<F(T &)>::type
        >
        operator<=(F f) const
        {
            return
                stubs::remote_object::apply_async(
                    gid_
                  , remote_object::invoke_apply_fun<T, F>(f)
                );
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & gid_;
        }
    };
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
        hpx::lcos::base_lco_with_value<
            hpx::components::object<T>
          , hpx::naming::id_type
        >
    >
    {
        static component_type HPX_ALWAYS_EXPORT
        get()
        {
            return hpx::components::component_base_lco_with_value;
        }
        static void HPX_ALWAYS_EXPORT
        set(component_type)
        {
            BOOST_ASSERT(false);
        }
    };
}}

#define HPX_REMOTE_OBJECT_REGISTER_RETURN_TYPE(TYPE)                            \
    HPX_REGISTER_ACTION_EX(                                                     \
    hpx::components::server::remote_object::apply_action<TYPE>::type,           \
    BOOST_PP_CAT(remote_object_apply_action_, TYPE));                           \
/**/

#endif
