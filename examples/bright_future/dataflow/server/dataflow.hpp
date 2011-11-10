//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_SERVER_DATAFLOW_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_SERVER_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <examples/bright_future/dataflow/server/detail/dataflow_impl.hpp>
#include <examples/bright_future/dataflow/server/detail/component_wrapper.hpp>

namespace hpx { namespace lcos { namespace server
{
    /// The dataflow server side representation
    struct dataflow
        : components::managed_component_base<dataflow>
    {
        dataflow()
            : component_ptr(0)
            , connect_thread_id(0)
        {}

        ~dataflow()
        {
            LLCO_(info)
                << "~server::dataflow::dataflow()";
            delete component_ptr;
            if(connect_thread_id != 0)
            {
                threads::set_thread_state(connect_thread_id, threads::pending
                  , threads::wait_abort);
            }
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

            component_type * w = new component_type(target);
            (*w)->init();

            typename hpx::util::spinlock::scoped_lock l(mtx);
            component_ptr = w;

            // waking up a possible sleeping thread that was trying to connect
            if(connect_thread_id != 0)
            {
                threads::set_thread_state(connect_thread_id, threads::pending);
                connect_thread_id = 0;
            }
        }

            // Vertical preprocessor repetition to define the remaining
            // init functions and actions
            // TODO: get rid of the call to impl_ptr->init
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                         \
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>       \
        void init(                                                            \
            naming::id_type const & target                                    \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                      \
        )                                                                     \
        {                                                                     \
            typedef                                                           \
                detail::dataflow_impl<                                        \
                    Action                                                    \
                  , BOOST_PP_ENUM_PARAMS(N, A)                                \
                >                                                             \
                wrapped_type;                                                 \
                                                                              \
            typedef                                                           \
                detail::component_wrapper<                                    \
                    wrapped_type                                              \
                >                                                             \
                component_type;                                               \
            component_type * w = new component_type(target);                  \
            (*w)->init(BOOST_PP_ENUM_PARAMS(N, a));                           \
                                                                              \
            typename hpx::util::spinlock::scoped_lock l(mtx);                 \
            component_ptr = w;                                                \
                                                                              \
            if(connect_thread_id != 0)                                        \
            {                                                                 \
                threads::set_thread_state(connect_thread_id, threads::pending); \
                connect_thread_id = 0;                                        \
            }                                                                 \
        }                                                                     \
    /**/

        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
          , HPX_LCOS_DATAFLOW_M0
          , _
        )

#undef HPX_LCOS_DATAFLOW_M0

        /// the connect function is used to connect the current dataflow
        /// to the specified target lco
        void connect(naming::id_type const & target)
        {
            {
                hpx::util::spinlock::scoped_lock l(mtx);

                // wait until component_ptr is initialized.
                if(component_ptr == 0)
                {
                    LLCO_(info)
                        << "server::dataflow::connect() executed before server::dataflow::init finished.";
                    threads::thread_self *self = threads::get_self_ptr_checked();
                    connect_thread_id = self->get_thread_id();
                    threads::thread_id_type id = connect_thread_id;
                    threads::set_thread_lco_description(connect_thread_id, "server::dataflow::connect");

                    threads::thread_state_ex_enum statex = threads::wait_unknown;
                    {
                        util::unlock_the_lock<hpx::util::spinlock::scoped_lock> ul(l);
                        statex = self->yield(threads::suspended);
                    }

                    threads::set_thread_lco_description(id);
                    connect_thread_id = 0;
                    if(statex == threads::wait_abort)
                    {
                        hpx::util::osstream strm;
                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROW_EXCEPTION(yield_aborted,
                            "dataflow::connect()",
                            hpx::util::osstream_get_string(strm));

                        return;
                    }
                }
                BOOST_ASSERT(component_ptr);
            }

            (*component_ptr)->connect_nonvirt(target);
        }

        typedef
            ::hpx::actions::action1<
                dataflow
              , 0
              , naming::id_type const &
              , &dataflow::connect
            >
            connect_action;

    private:
        detail::component_wrapper_base * component_ptr;
        hpx::util::spinlock mtx;
        threads::thread_id_type connect_thread_id;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// init_action is the action that can be used to call the variadic
    /// function from a client
    template <
        typename Action
      , BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
            HPX_ACTION_ARGUMENT_LIMIT
          , typename A
          , void
        )
      , typename Enable = void
    >
    struct init_action;

    template <typename Action>
    struct init_action<Action>
      : hpx::actions::action1<
            dataflow
          , 0
          , naming::id_type const &
          , &dataflow::init<Action>
          , threads::thread_priority_default
          , init_action<Action>
        >
    {
    private:
        typedef hpx::actions::action1<
            dataflow
          , 0
          , naming::id_type const &
          , &dataflow::init<Action>
          , threads::thread_priority_default
          , init_action<Action>
        > base_type;

    public:
        init_action() {}

        // construct an action from its arguments
        init_action(naming::id_type const & target)
          : base_type(target)
        {}

        init_action(threads::thread_priority p, naming::id_type const & target)
          : base_type(p, target)
        {}

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<init_action, base_type>();
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

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Action>)
  , (hpx::lcos::server::init_action<Action>)
)

// bring in the remaining specializations for init_action
#include <examples/bright_future/dataflow/server/dataflow_impl.hpp>

#endif
