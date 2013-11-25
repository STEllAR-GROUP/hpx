//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_SERVER_REMOTE_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_SERVER_REMOTE_OBJECT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/void_cast.hpp>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable:4251 4275)
#endif

namespace hpx { namespace components { namespace server
{
    // component to hold a pointer to an object, ability to apply arbitrary
    // functions objects on that pointer.
    class HPX_COMPONENT_EXPORT remote_object
        : public managed_component_base<remote_object>
    {
    public:
        remote_object()
            : object(0)
        {}
        ~remote_object()
        {
            HPX_ASSERT(dtor);
            dtor(&object);
        }

        template <typename F>
        typename F::result_type apply1(F const & f);
        template <typename F, typename A>
        typename F::result_type apply2(F const & f, A const & a);

        void set_dtor(hpx::util::function<void(void**)> const & dtor);

        HPX_DEFINE_COMPONENT_ACTION(remote_object, set_dtor);

    private:
        void *object;
        hpx::util::function<void(void**)> dtor;
    };

    template <typename F>
    typename F::result_type remote_object::apply1(F const & f)
    {
        return f(&object);
    }

    template <typename F, typename A>
    typename F::result_type remote_object::apply2(F const & f, A const & a)
    {
        return f(&object, a);
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_GCC_VERSION) && (HPX_GCC_VERSION <= 40400)
    template <typename F>
    struct remote_object_apply_action1
      : hpx::actions::result_action1<
            remote_object
          , typename F::result_type
          , F const &
          , &remote_object::apply1<F>
          , remote_object_apply_action1<F>
        >
    {};

    template <typename F, typename A>
    struct remote_object_apply_action2
      : hpx::actions::result_action2<
            remote_object
          , typename F::result_type
          , F const &
          , A const &
          , &remote_object::apply2<F, A>
          , remote_object_apply_action2<F, A>
        >
    {};
#else
    template <typename F>
    struct remote_object_apply_action1
      : hpx::actions::result_action1<
            typename F::result_type (remote_object::*)(F const &)
          , &remote_object::apply1<F>
          , remote_object_apply_action1<F>
        >
    {};

    template <typename F, typename A>
    struct remote_object_apply_action2
      : hpx::actions::result_action2<
            typename F::result_type (remote_object::*)(F const &, A const &)
          , &remote_object::apply2<F, A>
          , remote_object_apply_action2<F, A>
        >
    {};
#endif
}}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

HPX_SERIALIZATION_REGISTER_TEMPLATE_ACTION(
    (template <typename F>)
  , (hpx::components::server::remote_object_apply_action1<F>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE_ACTION(
    (template <typename F, typename A>)
  , (hpx::components::server::remote_object_apply_action2<F, A>)
)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::remote_object::set_dtor_action
  , remote_object_set_dtor_action
)

HPX_UTIL_REGISTER_FUNCTION_DECLARATION(void(void**), hpx::util::function<void(void**)>, hpx_remote_object_dtor_functor)

#endif
