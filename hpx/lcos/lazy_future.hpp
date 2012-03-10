//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LAZY_FUTURE_JUN_27_2008_0420PM)
#define HPX_LCOS_LAZY_FUTURE_JUN_27_2008_0420PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/util/block_profiler.hpp>

#include <boost/variant.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    /// A lazy_future can be used by a single \a thread to invoke a
    /// (remote) action and wait for the result. The result is expected to be
    /// sent back to the lazy_future using the LCO's set_event action
    ///
    /// A lazy_future is one of the simplest synchronization primitives
    /// provided by HPX. It allows to synchronize on a lazily evaluated remote
    /// operation returning a result of the type \a Result.
    ///
    /// A lazy_future is similar to an \a eager_future, except that the action
    /// is invoked only if the value is requested.
    ///
    /// \tparam Action   The template parameter \a Action defines the action
    ///                  to be executed by this lazy_future instance. The
    ///                  arguments \a arg0,... \a argN are used as parameters
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this
    ///                  lazy_future is expected to return from
    ///                  \a lazy_future#get.
    /// \tparam DirectExecute The template parameter \a DirectExecute is an
    ///                  optimization aid allowing to execute the action
    ///                  directly if the target is local (without spawning a
    ///                  new thread for this). This template does not have to be
    ///                  supplied explicitly as it is derived from the template
    ///                  parameter \a Action.
    ///
    /// \note            The action executed using the lazy_future as a
    ///                  continuation must return a value of a type convertible
    ///                  to the type as specified by the template parameter
    ///                  \a Result.
    template <typename Action, typename Result, typename DirectExecute>
    class lazy_future;

    ///////////////////////////////////////////////////////////////////////////
    struct lazy_future_tag {};

    template <typename Action, typename Result>
    class lazy_future<Action, Result, boost::mpl::false_>
      : public promise<Result, typename Action::result_type>
    {
    private:
        typedef promise<Result, typename Action::result_type> base_type;

    public:
        /// Construct a (non-functional) instance of a \a lazy_future. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        lazy_future()
          : apply_logger_("lazy_future::apply"), closure_(0)
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ") args(0)";
        }

        /// Get the result of the requested action. This call invokes the
        /// action and yields control if the result is not ready. As soon as
        /// the result has been returned and the waiting thread has been
        /// re-scheduled by the thread manager the function \a lazy_future#get
        /// will return.
        Result get(error_code& ec = throws) const
        {
            if (!closure_)
            {
                HPX_THROWS_IF(ec, uninitialized_value,
                    "lazy_future::closure_", "closure not properly initialized");
                return Result();
            }

            closure_();

            // wait for the result (yield control)
            return (*this->impl_)->get_data(ec);
        }

        void invalidate(boost::exception_ptr const& e)
        {
            (*this->impl_)->set_error(e); // set the received error
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            util::block_profiler_wrapper<lazy_future_tag> bp(apply_logger_);
            hpx::applier::apply_c<Action>(this->get_gid(), gid);
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The lazy_future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        static void invoke(
            hpx::lcos::lazy_future<Action, Result, boost::mpl::false_> *th,
            naming::id_type const& gid)
        {
            // FIXME: Simultaneous calls to invokeN() methods may result in
            // multiple calls to apply(). This is a benign race condition,
            // as the underlying FEB prevents the action from being called
            // more than once; but it would be more efficient to reduce the
            // number of calls to apply().
            if (!((*th->impl_)->is_ready()))
                th->apply(gid);
        }

        // suppress warning about using this in constructor base initializer list
        lazy_future* this_() { return this; }

    public:
        /// Construct a new \a lazy_future instance. The \a thread
        /// supplied to the function \a lazy_future#get will be
        /// notified as soon as the result of the operation associated with
        /// this lazy_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_result action. Any error has to be
        ///               reported using a \a base_lco::set_error action. The
        ///               target for either of these actions has to be this
        ///               lazy_future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        lazy_future(naming::gid_type const& gid)
          : apply_logger_("lazy_future::apply"),
            closure_(boost::bind(&lazy_future::invoke, this_(),
                     naming::id_type(gid, naming::id_type::unmanaged)))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(0)";
        }

        lazy_future(naming::id_type const& gid)
          : apply_logger_("lazy_future::apply"),
            closure_(boost::bind(&lazy_future::invoke, this_(), gid))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(0)";
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, Arg0 const& arg0)
        {
            util::block_profiler_wrapper<lazy_future_tag> bp(apply_logger_);
            hpx::applier::apply_c<Action>(this->get_gid(), gid, arg0);
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The lazy_future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void invoke1(naming::id_type const& gid, Arg0 const& arg0)
        {
            if (!((*this->impl_)->is_ready()))
                this->apply(gid, arg0);
        }

    public:
        /// Construct a new \a lazy_future instance. The \a thread
        /// supplied to the function \a lazy_future#get will be
        /// notified as soon as the result of the operation associated with
        /// this lazy_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_result action. Any error has to be
        ///               reported using a \a base_lco::set_error action. The
        ///               target for either of these actions has to be this
        ///               lazy_future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        lazy_future(naming::gid_type const& gid, Arg0 const& arg0)
          : apply_logger_("lazy_future::apply"),
            closure_(boost::bind(&lazy_future::template invoke1<Arg0>, this_(),
                naming::id_type(gid, naming::id_type::unmanaged), arg0))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(1)";
        }

        template <typename Arg0>
        lazy_future(naming::id_type const& gid, Arg0 const& arg0)
          : apply_logger_("lazy_future::apply"),
            closure_(boost::bind(&lazy_future::template invoke1<Arg0>, this_(), gid, arg0))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(1)";
        }

        // pull in remaining constructors
        #include <hpx/lcos/lazy_future_constructors.hpp>

        util::block_profiler<lazy_future_tag> apply_logger_;
        HPX_STD_FUNCTION<void()> closure_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct lazy_future_direct_tag {};

    template <typename Action, typename Result>
    class lazy_future<Action, Result, boost::mpl::true_>
      : public promise<Result, typename Action::result_type>
    {
    private:
        typedef promise<Result, typename Action::result_type> base_type;

    public:
        /// Construct a (non-functional) instance of an \a lazy_future. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        lazy_future()
          : apply_logger_("lazy_future_direct::apply")
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ") args(0)";
        }

        /// Get the result of the requested action. This call invokes the
        /// action and yields control if the result is not ready. As soon as
        /// the result has been returned and the waiting thread has been
        /// re-scheduled by the thread manager the function \a lazy_future#get
        /// will return.
        Result get(error_code& ec = throws) const
        {
            if (!closure_)
            {
                HPX_THROWS_IF(ec, uninitialized_value,
                    "lazy_future::closure_", "closure not properly initialized");
                return Result();
            }

            closure_();

            // wait for the result (yield control)
            return (*this->impl_)->get_data(ec);
        }

        void invalidate(boost::exception_ptr const& e)
        {
            (*this->impl_)->set_error(e); // set the received error
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            util::block_profiler_wrapper<lazy_future_direct_tag> bp(apply_logger_);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_,
                    components::get_component_type<typename Action::component_type>()));
                (*this->impl_)->set_data(
                    Action::execute_function(addr.address_));
            }
            else {
                // remote execution
                hpx::applier::apply_c<Action>(addr, this->get_gid(), gid);
            }
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The lazy_future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        static void invoke(hpx::lcos::lazy_future<Action,Result,boost::mpl::true_> *th,
                           naming::id_type const& gid)
        {
            if (!((*th->impl_)->is_ready()))
              th->apply(gid);
        }

    public:
        /// Construct a new \a lazy_future instance. The \a thread
        /// supplied to the function \a lazy_future#get will be
        /// notified as soon as the result of the operation associated with
        /// this lazy_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_result action. Any error has to be
        ///               reported using a \a base_lco::set_error action. The
        ///               target for either of these actions has to be this
        ///               lazy_future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        lazy_future(naming::gid_type const& gid)
          : apply_logger_("lazy_future_direct::apply"),
            closure_(boost::bind(&lazy_future::invoke,
                naming::id_type(gid, naming::id_type::unmanaged)))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(0)";
        }

        lazy_future(naming::id_type const& gid)
          : apply_logger_("lazy_future_direct::apply"),
            closure_(boost::bind(&lazy_future::invoke, this, gid))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(0)";
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, Arg0 const& arg0)
        {
            util::block_profiler_wrapper<lazy_future_direct_tag> bp(apply_logger_);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_,
                    components::get_component_type<typename Action::component_type>()));
                (*this->impl_)->set_data(
                    Action::execute_function(addr.address_, arg0));
            }
            else {
                // remote execution
                hpx::applier::apply_c<Action>(addr, this->get_gid(), gid, arg0);
            }
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The lazy_future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        static void invoke1(hpx::lcos::lazy_future<Action,Result,boost::mpl::true_> *th,
                            naming::id_type const& gid, Arg0 const& arg0)
        {
            if (!((*th->impl_)->is_ready()))
                th->apply(gid, arg0);
        }

    public:
        /// Construct a new \a lazy_future instance. The \a thread
        /// supplied to the function \a lazy_future#get will be
        /// notified as soon as the result of the operation associated with
        /// this lazy_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_result action. Any error has to be
        ///               reported using a \a base_lco::set_error action. The
        ///               target for either of these actions has to be this
        ///               lazy_future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        lazy_future(naming::gid_type const& gid, Arg0 const& arg0)
          : apply_logger_("lazy_future_direct::apply"),
            closure_(boost::bind(&lazy_future::template invoke1<Arg0>, this,
                naming::id_type(gid, naming::id_type::unmanaged), arg0))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(1)";
        }

        template <typename Arg0>
        lazy_future(naming::id_type const& gid, Arg0 const& arg0)
          : apply_logger_("lazy_future_direct::apply"),
            closure_(boost::bind(&lazy_future::template invoke1<Arg0>, this, gid, arg0))
        {
            LLCO_(info) << "lazy_future::lazy_future("
                        << hpx::actions::detail::get_action_name<Action>()
                        << ", "
                        << gid
                        << ") args(1)";
        }

        // pull in remaining constructors
        #include <hpx/lcos/lazy_future_constructors_direct.hpp>

        util::block_profiler<lazy_future_direct_tag> apply_logger_;
        HPX_STD_FUNCTION<void()> closure_;
    };
}}

#endif
