//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_SERVER_REMOTE_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_SERVER_REMOTE_OBJECT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/function.hpp>

namespace hpx { namespace components { namespace server
{
    // component to hold a pointer to an object, ability to apply arbitrary
    // functions objects on that pointer.
    class HPX_COMPONENT_EXPORT remote_object
        : public simple_component_base<remote_object>
    {
        public:
            remote_object()
                : object(0)
            {}
            ~remote_object()
            {
                BOOST_ASSERT(dtor);
                dtor(&object);
            }

            enum actions
            {
                remote_object_apply    = 1
              , remote_object_set_dtor = 2
            };

            template <typename R>
            R apply(hpx::util::function<R(void**)> ctor, std::size_t count);

            template <typename R>
            struct apply_action
            {
                typedef
                    hpx::actions::result_action2<
                        remote_object
                      , R
                      , remote_object_apply
                      , hpx::util::function<R(void**)>
                      , std::size_t
                      , &remote_object::apply<R>
                    >
                    type;
            };

            void set_dtor(hpx::util::function<void(void**)> d, std::size_t count);

            typedef
                hpx::actions::action2<
                    remote_object
                  , remote_object_set_dtor
                  , hpx::util::function<void(void**)>
                  , std::size_t
                  , &remote_object::set_dtor
                >
                set_dtor_action;
        private:
            void *object;
            hpx::util::function<void(void**)> dtor;
    };

    template <>
    struct remote_object::apply_action<void>
    {
        typedef
            hpx::actions::action2<
                remote_object
              , remote_object_apply
              , hpx::util::function<void(void**)>
              , std::size_t
              , &remote_object::apply<void>
            >
            type;
    };

    template <typename R>
    R remote_object::apply(hpx::util::function<R(void**)> f, std::size_t)
    {
        return f(&object);
    }

}}}

namespace boost { namespace serialization {
    template<typename R>
    struct guid_defined<
        hpx::actions::result_action2<
            hpx::components::server::remote_object
          , R
          , hpx::components::server::remote_object::remote_object_apply
          , hpx::util::function<R(void**)>
          , std::size_t
          , &hpx::components::server::remote_object::apply<R>
        >
    > : boost::mpl::true_ {};

    namespace ext {
        template <typename R>
        struct guid_impl<
            hpx::actions::result_action2<
                hpx::components::server::remote_object
              , R
              , hpx::components::server::remote_object::remote_object_apply
              , hpx::util::function<R(void**)>
              , std::size_t
              , &hpx::components::server::remote_object::apply<R>
            >
        >
        {
            static inline const char * call()
            {
                return
                    hpx::util::detail::type_hash<
                        hpx::actions::result_action2<
                            hpx::components::server::remote_object
                          , R
                          , hpx::components::server::remote_object::actions::remote_object_apply
                          , hpx::util::function<R(void**)>
                          , std::size_t
                          , &hpx::components::server::remote_object::apply<R>
                        >
                    >();
            }
        };
    }
}}
namespace boost { namespace archive { namespace detail { namespace extra_detail {
    template <typename R>
    struct init_guid<
        hpx::actions::result_action2<
            hpx::components::server::remote_object
          , R
          , hpx::components::server::remote_object::remote_object_apply
          , hpx::util::function<R(void**)>
          , std::size_t
          , &hpx::components::server::remote_object::apply<R>
        >
    >
    {
        static
            hpx::util::detail::guid_initializer_helper<
                hpx::actions::result_action2<
                    hpx::components::server::remote_object
                  , R
                  , hpx::components::server::remote_object::remote_object_apply
                  , hpx::util::function<R(void**)>
                  , std::size_t
                  , &hpx::components::server::remote_object::apply<R>
                >
            > const &
            g;
    };

    template <typename R>
    hpx::util::detail::guid_initializer_helper<
        hpx::actions::result_action2<
            hpx::components::server::remote_object
          , R
          , hpx::components::server::remote_object::remote_object_apply
          , hpx::util::function<R(void**)>
          , std::size_t
          , &hpx::components::server::remote_object::apply<R>
        >
    > const &
    init_guid<
        hpx::actions::result_action2<
            hpx::components::server::remote_object
          , R
          , hpx::components::server::remote_object::remote_object_apply
          , hpx::util::function<R(void**)>
          , std::size_t
          , &hpx::components::server::remote_object::apply<R>
        >
    >::g = ::boost::serialization::singleton<
        hpx::util::detail::guid_initializer_helper<
            hpx::actions::result_action2<
                hpx::components::server::remote_object
              , R
              , hpx::components::server::remote_object::remote_object_apply
              , hpx::util::function<R(void**)>
              , std::size_t
              , &hpx::components::server::remote_object::apply<R>
            >
        >
    >::get_mutable_instance().export_guid();
}}}}

namespace hpx { namespace actions { namespace detail { namespace ext {
    template <typename R>
    struct get_action_name_impl<
        hpx::actions::result_action2<
            hpx::components::server::remote_object
          , R
          , hpx::components::server::remote_object::remote_object_apply
          , hpx::util::function<R(void**)>
          , std::size_t
          , &hpx::components::server::remote_object::apply<R>
        >
    >
    {
        static HPX_ALWAYS_EXPORT const char * call()
        {
            return
                hpx::util::detail::type_hash<
                    hpx::actions::result_action2<
                        hpx::components::server::remote_object
                      , R
                      , hpx::components::server::remote_object::remote_object_apply
                      , hpx::util::function<R(void**)>
                      , std::size_t
                      , &hpx::components::server::remote_object::apply<R>
                    >
                >();
        }
    };
}}}}
#endif
