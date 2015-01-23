//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef HPX_COMPONENTS_REMOTE_OBJECT_NEW_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_NEW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/components/remote_object/stubs/remote_object.hpp>
#include <hpx/components/remote_object/object.hpp>
#include <hpx/components/remote_object/new_impl.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/functional/new.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace components
{
    namespace remote_object
    {
        template <typename T, typename ...Ts>
        struct ctor_fun
        {
            typedef void result_type;

            // default constructor is needed for serialization
            ctor_fun() {}

            template <typename ...Args>
            ctor_fun(util::tuple<Args...>&& args)
                : args_(std::move(args))
            {}

            ctor_fun(ctor_fun const& other)
              : args_(other.args_)
            {}

            ctor_fun(ctor_fun&& other)
              : args_(std::move(other.args_))
            {}

            void operator()(void ** p) const
            {
                *p = util::invoke_fused(
                    util::functional::new_<T>(), std::move(args_));
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & args_;
            }

        private:
            util::tuple<typename util::decay<Ts>::type...> args_;
        };

        template <typename T>
        struct dtor_fun
        {
            typedef void result_type;

            void operator()(void ** o) const
            {
                delete (*reinterpret_cast<T **>(o));
            }

            template <typename Archive>
            void serialize(Archive &, unsigned)
            {}
        };
    }

    // asynchronously creates an instance of type T on locality target_id
    template <typename T, typename ...Ts>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, Ts&&... vs)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;

        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<T, Ts...>(
                util::forward_as_tuple(std::forward<Ts>(vs)...))
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
}}

#endif
