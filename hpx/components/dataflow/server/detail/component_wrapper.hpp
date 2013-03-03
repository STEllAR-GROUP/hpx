//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_COMPONENT_WRAPPER_ARG
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_COMPONENT_WRAPPER_ARG

#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    struct component_wrapper_base
    {
        virtual ~component_wrapper_base() {}
        virtual base_lco &operator*() = 0;
        virtual base_lco const &operator*() const = 0;
        virtual base_lco * operator->() = 0;
        virtual base_lco const * operator->() const = 0;
        virtual void finalize() = 0;
    };

    template <typename T>
    struct component_wrapper
        : component_wrapper_base
    {
        typedef components::managed_component<T> component_type;

        component_type * component_ptr;

        component_wrapper()
        {
            T * t = new T;
            component_ptr = new component_type(t);
        }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        component_wrapper(HPX_ENUM_FWD_ARGS(N, A, a))                           \
        {                                                                       \
            T * t = new T(HPX_ENUM_FORWARD_ARGS(N, A, a));                      \
            component_ptr = new component_type(t);                              \
        }                                                                       \
    /**/
        BOOST_PP_REPEAT_FROM_TO(1, 10, HPX_LCOS_DATAFLOW_M0, _)
#undef HPX_LCOS_DATAFLOW_M0

        void finalize()
        {
            component_ptr->finalize();
        }

        ~component_wrapper()
        {
            delete component_ptr;
        }

        T &operator*()
        {
            return *component_ptr->get_checked();
        }

        T const&operator*() const
        {
            return *component_ptr->get_checked();
        }

        T *operator->()
        {
            return component_ptr->get_checked();
        }

        T const *operator->() const
        {
            return component_ptr->get_checked();
        }
    };
}}}}

#endif
