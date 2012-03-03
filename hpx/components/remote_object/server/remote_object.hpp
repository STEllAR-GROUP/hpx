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
        R apply(hpx::util::function<R(void**)> const & f, std::size_t count);

        void set_dtor(hpx::util::function<void(void**)> const & dtor, std::size_t count);

        typedef
            hpx::actions::action2<
                remote_object
              , remote_object_set_dtor
              , hpx::util::function<void(void**)> const &
              , std::size_t
              , &remote_object::set_dtor
            >
            set_dtor_action;
    private:
        void *object;
        hpx::util::function<void(void**)> dtor;
    };

    template <typename R>
    R remote_object::apply(hpx::util::function<R(void**)> const & f, std::size_t)
    {
        return f(&object);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct remote_object_apply_action
      : hpx::actions::result_action2<
            remote_object
          , R
          , remote_object::remote_object_apply
          , hpx::util::function<R(void**)> const &
          , std::size_t
          , &remote_object::apply<R>
          , hpx::threads::thread_priority_default
          , remote_object_apply_action<R>
        >
    {
    private:
        typedef hpx::actions::result_action2<
                remote_object
              , R
              , remote_object::remote_object_apply
              , hpx::util::function<R(void**)> const &
              , std::size_t
              , &remote_object::apply<R>
              , hpx::threads::thread_priority_default
              , remote_object_apply_action<R>
            >
            base_type;

    public:
        remote_object_apply_action() {}

        // construct an action from its arguments
        remote_object_apply_action(hpx::util::function<R(void**)> const& f,
                std::size_t size)
          : base_type(f, size)
        {}

        remote_object_apply_action(threads::thread_priority p,
                hpx::util::function<R(void**)> const & f, std::size_t size)
          : base_type(p, f, size)
        {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<remote_object_apply_action, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    template <>
    struct remote_object_apply_action<void>
      : hpx::actions::action2<
            remote_object
          , remote_object::remote_object_apply
          , hpx::util::function<void(void**)> const &
          , std::size_t
          , &remote_object::apply<void>
          , hpx::threads::thread_priority_default
          , remote_object_apply_action<void>
        >
    {
    private:
        typedef hpx::actions::action2<
                remote_object
              , remote_object::remote_object_apply
              , hpx::util::function<void(void**)> const &
              , std::size_t
              , &remote_object::apply<void>
              , hpx::threads::thread_priority_default
              , remote_object_apply_action<void>
            >
            base_type;

    public:
        remote_object_apply_action() {}

        // construct an action from its arguments
        remote_object_apply_action(hpx::util::function<void(void**)> const & f,
                std::size_t size)
          : base_type(f, size)
        {}

        remote_object_apply_action(threads::thread_priority p,
                hpx::util::function<void(void**)> const & f, std::size_t size)
          : base_type(p, f, size)
        {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<remote_object_apply_action, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
}}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename R>)
  , (hpx::components::server::remote_object_apply_action<R>)
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::remote_object_apply_action<void>
  , remote_object_apply_action_void
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::remote_object::set_dtor_action
  , remote_object_set_dtor_action
)

#endif
