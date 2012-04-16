//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PACKAGED_TASK_JUN_27_2008_0420PM)
#define HPX_LCOS_PACKAGED_TASK_JUN_27_2008_0420PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/full_empty_memory.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    /// A packaged_task can be used by a single \a thread to invoke a
    /// (remote) action and wait for the result. The result is expected to be
    /// sent back to the packaged_task using the LCO's set_event action
    ///
    /// A packaged_task is one of the simplest synchronization primitives
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result.
    ///
    /// \tparam Action   The template parameter \a Action defines the action
    ///                  to be executed by this packaged_task instance. The
    ///                  arguments \a arg0,... \a argN are used as parameters
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this
    ///                  packaged_task is expected to return from its associated
    ///                  future \a packaged_task#get_future.
    /// \tparam DirectExecute The template parameter \a DirectExecute is an
    ///                  optimization aid allowing to execute the action
    ///                  directly if the target is local (without spawning a
    ///                  new thread for this). This template does not have to be
    ///                  supplied explicitly as it is derived from the template
    ///                  parameter \a Action.
    ///
    /// \note            The action executed using the packaged_task as a
    ///                  continuation must return a value of a type convertible
    ///                  to the type as specified by the template parameter
    ///                  \a Result.
    template <typename Action, typename Result, typename DirectExecute>
    class packaged_task;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_task<Action, Result, boost::mpl::false_>
      : public promise<Result,
            typename hpx::actions::extract_action<Action>::result_type>
    {
    private:
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef promise<Result, typename action_type::result_type> base_type;
        typedef typename base_type::completed_callback_type
            completed_callback_type;

        struct profiler_tag {};

    public:
        /// Construct a (non-functional) instance of an \a packaged_task. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        packaged_task()
          : apply_logger_("packaged_task")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ") args(0)";
        }

        explicit packaged_task(completed_callback_type const& data_sink)
          : base_type(data_sink),
            apply_logger_("packaged_task")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ") args(0)";
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
            hpx::apply_c<action_type>(this->get_gid(), gid);
        }

        void apply_p(naming::id_type const& gid, threads::thread_priority priority)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
            hpx::apply_c_p<action_type>(this->get_gid(), gid, priority);
        }

        /// Construct a new \a packaged_task instance. The \a thread
        /// supplied to the function \a packaged_task#get will be
        /// notified as soon as the result of the operation associated with
        /// this packaged_task instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               packaged_task instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        packaged_task(naming::id_type const& gid)
          : apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(0)";
            apply(gid);
        }

        packaged_task(naming::id_type const& gid,
                completed_callback_type const& data_sink)
          : base_type(data_sink),
            apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(0)";
            apply(gid);
        }

        packaged_task(naming::id_type const& gid,
                threads::thread_priority priority)
          : apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(0)";
            apply_p(gid, priority);
        }

        packaged_task(naming::id_type const& gid,
                threads::thread_priority priority,
                completed_callback_type const& data_sink)
          : base_type(data_sink),
            apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(0)";
            apply_p(gid, priority);
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
            hpx::apply_c<action_type>(this->get_gid(), gid,
                boost::forward<Arg0>(arg0));
        }

        template <typename Arg0>
        void apply_p(naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
            hpx::apply_c_p<action_type>(
                this->get_gid(), gid, priority, boost::forward<Arg0>(arg0));
        }

        /// Construct a new \a packaged_task instance. The \a thread
        /// supplied to the function \a packaged_task#get will be
        /// notified as soon as the result of the operation associated with
        /// this packaged_task instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               packaged_task instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        packaged_task(naming::id_type const& gid,
                BOOST_FWD_REF(Arg0) arg0)
          : apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(1)";
            apply(gid, boost::forward<Arg0>(arg0));
        }

        template <typename Arg0>
        packaged_task(naming::id_type const& gid,
                completed_callback_type const& data_sink,
                BOOST_FWD_REF(Arg0) arg0)
          : base_type(data_sink),
            apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(1)";
            apply(gid, boost::forward<Arg0>(arg0));
        }

        template <typename Arg0>
        packaged_task(naming::id_type const& gid,
                threads::thread_priority priority,
                BOOST_FWD_REF(Arg0) arg0)
          : apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(1)";
            apply_p(gid, priority, boost::forward<Arg0>(arg0));
        }

        template <typename Arg0>
        packaged_task(naming::id_type const& gid,
                threads::thread_priority priority,
                completed_callback_type const& data_sink,
                BOOST_FWD_REF(Arg0) arg0)
          : base_type(data_sink),
            apply_logger_("packaged_task::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(1)";
            apply_p(gid, priority, boost::forward<Arg0>(arg0));
        }

        // pull in remaining constructors
        #include <hpx/lcos/packaged_task_constructors.hpp>

        util::block_profiler<profiler_tag> apply_logger_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_task<Action, Result, boost::mpl::true_>
      : public promise<Result,
          typename hpx::actions::extract_action<Action>::result_type>
    {
    private:
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef promise<Result, typename action_type::result_type> base_type;
        typedef typename base_type::completed_callback_type
            completed_callback_type;

        struct profiler_tag {};

    public:
        /// Construct a (non-functional) instance of an \a packaged_task. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        packaged_task(completed_callback_type const& data_sink)
          : base_type(data_sink),
            apply_logger_("packaged_task_direct::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ") args(0)";
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        template <typename IdType>
        void apply(IdType const& gid)
        {
            BOOST_STATIC_ASSERT((boost::is_same<IdType, naming::id_type>::value));

            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_,
                    components::get_component_type<
                        typename action_type::component_type>()));
                (*this->impl_)->set_data(
                    action_type::execute_function(addr.address_));
            }
            else {
                // remote execution
                hpx::applier::detail::apply_c<action_type>(addr,
                    this->get_gid(), gid);
            }
        }

        /// Construct a new \a packaged_task instance. The \a thread
        /// supplied to the function \a packaged_task#get will be
        /// notified as soon as the result of the operation associated with
        /// this packaged_task instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               packaged_task instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        packaged_task(naming::id_type const& gid)
          : apply_logger_("packaged_task_direct::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(0)";
            apply(gid);
        }
        packaged_task(naming::id_type const& gid,
                completed_callback_type const& data_sink)
          : base_type(data_sink),
            apply_logger_("packaged_task_direct::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(0)";
            apply(gid);
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_,
                    components::get_component_type<
                        typename action_type::component_type>()));
                (*this->impl_)->set_data(
                    action_type::execute_function(addr.address_,
                        boost::forward<Arg0>(arg0)));
            }
            else {
                // remote execution
                hpx::applier::detail::apply_c<action_type>(addr,
                    this->get_gid(), gid, boost::forward<Arg0>(arg0));
            }
        }

        /// Construct a new \a packaged_task instance. The \a thread
        /// supplied to the function \a packaged_task#get will be
        /// notified as soon as the result of the operation associated with
        /// this packaged_task instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               packaged_task instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        packaged_task(naming::id_type const& gid,
                BOOST_FWD_REF(Arg0) arg0)
          : apply_logger_("packaged_task_direct::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(1)";
            apply(gid, boost::forward<Arg0>(arg0));
        }
        template <typename Arg0>
        packaged_task(naming::id_type const& gid,
                completed_callback_type const& data_sink,
                BOOST_FWD_REF(Arg0) arg0)
          : base_type(data_sink),
            apply_logger_("packaged_task_direct::apply")
        {
            LLCO_(info) << "packaged_task::packaged_task("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(1)";
            apply(gid, boost::forward<Arg0>(arg0));
        }

        // pull in remaining constructors
        #include <hpx/lcos/packaged_task_constructors_direct.hpp>

        util::block_profiler<profiler_tag> apply_logger_;
    };
}}

#endif
