//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_impl.hpp>
#include <hpx/components/dataflow/server/detail/component_wrapper.hpp>

namespace hpx { namespace lcos { namespace server
{
    /// The dataflow server side representation
    struct HPX_COMPONENT_EXPORT dataflow
        : base_lco
        , components::managed_component_base<
            dataflow
          , hpx::components::detail::this_type
          , hpx::traits::construct_with_back_ptr
        >
    {
        typedef
            components::managed_component_base<
                dataflow
              , hpx::components::detail::this_type
              , hpx::traits::construct_with_back_ptr
            >
            base_type;
        typedef hpx::components::managed_component<dataflow> component_type;

        // disambiguate base classes
        typedef base_lco base_type_holder;
        using base_type::finalize;
        typedef base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<dataflow>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<dataflow>(type);
        }

        void finalize()
        {
        }

        ~dataflow()
        {
            BOOST_ASSERT(component_ptr);
            component_ptr->finalize();
            LLCO_(info)
                << "~server::dataflow::dataflow()";
            BOOST_ASSERT(component_ptr);
            delete component_ptr;

            lcos::local::spinlock::scoped_lock l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.destructed_;
        }

        /// init initializes the dataflow, it creates a dataflow_impl object
        /// that holds old type information and does the remaining processing
        /// of managing the dataflow.
        /// init is a variadic function. The first template parameter denotes
        /// the Action that needs to get spawned once all arguments are
        /// computed
        template <typename Action>
        void init(naming::id_type const & target)
        {
            typedef detail::dataflow_impl<Action> wrapped_type;
            typedef
                detail::component_wrapper<wrapped_type>
                component_type;

            LLCO_(info)
                << "server::dataflow::init() " << get_gid();

            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init();

            lcos::local::spinlock::scoped_lock l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }

        dataflow()
        {
            BOOST_ASSERT(false);
        }

        dataflow(component_type * back_ptr)
            : base_type(back_ptr)
        {
            BOOST_ASSERT(false);
        }

        template <typename Action>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            /*
            applier::register_thread(
                HPX_STD_BIND(&dataflow::init<typename Action::type>
                  , this
                  , target
                )
              , "hpx::lcos::server::dataflow::init<>"
            );
            */
            init<typename Action::type>(target);
        }

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/components/dataflow/server/preprocessed/dataflow.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/dataflow_" HPX_LIMIT_STR ".hpp")
#endif

        // Vertical preprocessor repetition to define the remaining
        // init functions and actions
#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_ACTION_ARGUMENT_LIMIT                                           \
          , <hpx/components/dataflow/server/dataflow.hpp>                       \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

        /// the connect function is used to connect the current dataflow
        /// to the specified target lco
        void connect(naming::id_type const & target)
        {
            LLCO_(info) <<
                "hpx::lcos::server::dataflow::connect(" << target << ") {" << get_gid() << "}"
                ;
            {
                lcos::local::spinlock::scoped_lock l(mtx);

                // wait until component_ptr is initialized.
                if(component_ptr == 0)
                {
                    targets.push_back(target);
                    return;
                }
            }
            (*component_ptr)->connect_nonvirt(target);
        }

        void set_event() {}

    private:
        detail::component_wrapper_base * component_ptr;
        lcos::local::spinlock mtx;
        std::vector<naming::id_type> targets;
    };
}}}

#endif

#else // BOOST_PP_IS_ITERATING
#define N BOOST_PP_ITERATION()
        // TODO: get rid of the call to impl_ptr->init

#define M0(Z, N, D) BOOST_FWD_REF(BOOST_PP_CAT(A, N)) BOOST_PP_CAT(a, N)
#define M1(Z, N, D) boost::forward<BOOST_PP_CAT(A, N)>(BOOST_PP_CAT(a, N))
#define M2(Z, N, D)                                                             \
    typename boost::remove_const<                                               \
            typename hpx::util::detail::remove_reference<                       \
                BOOST_PP_CAT(A, N)                                              \
            >::type                                                             \
        >::type                                                                 \
/**/

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>
        void init(
            naming::id_type const & target
          , BOOST_PP_ENUM(N, M0, _) // A0 && a0, A1 && a1, ..., AN && aN
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , BOOST_PP_ENUM(N, M2, _)
                >
                wrapped_type;

            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(BOOST_PP_ENUM(N, M1, _));

            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_PP_ENUM(N, M0, _) // A0 && a0, A1 && a1, ..., AN && aN
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            /*
            applier::register_thread(
                HPX_STD_BIND(&dataflow::init<
                        typename Action::type, BOOST_PP_ENUM_PARAMS(N, A)
                    >
                  , this
                  , target
                  , BOOST_PP_ENUM(N, A)
                )
              , "hpx::lcos::server::dataflow::init<>"
            );
            */
            init<typename Action::type>(target, BOOST_PP_ENUM(N, M1, _));
        }

#undef M2
#undef M1
#undef M0
#undef N
#endif
