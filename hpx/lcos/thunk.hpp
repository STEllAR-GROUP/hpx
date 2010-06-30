//  Copyright (c) 2007-2010 Hartmut Kaiser, Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_THUNK_JUN_27_2008_0420PM)
#define HPX_LCOS_THUNK_JUN_27_2008_0420PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/util/block_profiler.hpp>

#include <boost/variant.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class thunk thunk.hpp hpx/lcos/thunk.hpp
    ///
    /// A thunk can be used by a single \a thread to invoke a 
    /// (remote) action and wait for the result. The result is expected to be 
    /// sent back to the thunk using the LCO's set_event action
    ///
    /// A thunk is one of the simplest synchronization primitives 
    /// provided by HPX. It allows to synchronize on a lazily evaluated remote
    /// operation returning a result of the type \a Result. 
    ///
    /// A thunk is similar to an \a eager_future, except that the action
    /// is invoked only if the value is requested.
    ///
    /// \tparam Action   The template parameter \a Action defines the action 
    ///                  to be executed by this thunk instance. The 
    ///                  arguments \a arg0,... \a argN are used as parameters 
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this 
    ///                  thunk is expected to return from 
    ///                  \a thunk#get.
    /// \tparam DirectExecute The template parameter \a DirectExecute is an
    ///                  optimization aid allowing to execute the action 
    ///                  directly if the target is local (without spawning a 
    ///                  new thread for this). This template does not have to be
    ///                  supplied explicitly as it is derived from the template 
    ///                  parameter \a Action.
    ///
    /// \note            The action executed using the thunk as a 
    ///                  continuation must return a value of a type convertible 
    ///                  to the type as specified by the template parameter 
    ///                  \a Result.
    template <typename Action, typename Result, typename DirectExecute>
    class thunk;

    ///////////////////////////////////////////////////////////////////////////
    struct thunk_tag {};

    template <typename Action, typename Result>
    class thunk<Action, Result, boost::mpl::false_> 
        : public future_value<Result, typename Action::result_type>
    {
    private:
        typedef future_value<Result, typename Action::result_type> base_type;

    public:
        /// Construct a (non-functional) instance of a \a thunk. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        thunk()
          : apply_logger_("thunk::apply"), closure_(0)
        {}

        /// Get the result of the requested action. This call invokes the 
        /// action and yields control if the result is not ready. As soon as 
        /// the result has been returned and the waiting thread has been 
        /// re-scheduled by the thread manager the function \a thunk#get 
        /// will return.
        Result get() const
        {
            if (!closure_)
            {
                boost::throw_exception(
                    std::logic_error("Closure uninitialized"));
            }

            closure_();

            // wait for the result (yield control)
            return (*this->impl_)->get_data(0);
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            util::block_profiler_wrapper<thunk_tag> bp(apply_logger_);
            hpx::applier::apply_c<Action>(this->get_gid(), gid);
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The thunk.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        static void invoke(hpx::lcos::thunk<Action,Result> *th, 
                           naming::id_type const& gid)
        {
            if (!((*th->impl_)->is_data()))
              th->apply(gid);
        }

    public:
        /// Construct a new \a thunk instance. The \a thread 
        /// supplied to the function \a thunk#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this thunk instance has been returned.
        /// 
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               thunk instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        thunk(naming::gid_type const& gid)
          : apply_logger_("thunk::apply"),
            closure_(boost::bind(invoke, this, 
                     naming::id_type(gid, naming::id_type::unmanaged)))
        { }
        thunk(naming::id_type const& gid)
          : apply_logger_("thunk::apply"),
            closure_(boost::bind(invoke, this, gid))
        { }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the 
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, Arg0 const arg0)
        {
            util::block_profiler_wrapper<thunk_tag> bp(apply_logger_);
            hpx::applier::apply_c<Action>(this->get_gid(), gid, arg0);
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The thunk.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void invoke1(naming::id_type const& gid, Arg0 const arg0)
        {
            if (!((*this->impl_)->is_data()))
                this->apply(gid, arg0);
        }

    public:
        /// Construct a new \a thunk instance. The \a thread 
        /// supplied to the function \a thunk#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this thunk instance has been returned.
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
        ///               thunk instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        thunk(naming::gid_type const& gid, Arg0 const& arg0)
          : apply_logger_("thunk::apply"),
            closure_(boost::bind(invoke1, this, 
                naming::id_type(gid, naming::id_type::unmanaged), arg0))
        { }
        template <typename Arg0>
        thunk(naming::id_type const& gid, Arg0 const& arg0)
          : apply_logger_("thunk::apply"),
            closure_(boost::bind(invoke1, this, gid, arg0))
        { }

        // pull in remaining constructors
        #include <hpx/lcos/thunk_constructors.hpp>

        util::block_profiler<thunk_tag> apply_logger_;
        boost::function<void()> closure_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct thunk_direct_tag {};

    template <typename Action, typename Result>
    class thunk<Action, Result, boost::mpl::true_> 
        : public future_value<Result, typename Action::result_type>
    {
    private:
        typedef future_value<Result, typename Action::result_type> base_type;

    public:
        /// Construct a (non-functional) instance of an \a thunk. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        thunk()
          : apply_logger_("thunk_direct::apply")
        {}

        /// Get the result of the requested action. This call invokes the 
        /// action and yields control if the result is not ready. As soon as 
        /// the result has been returned and the waiting thread has been 
        /// re-scheduled by the thread manager the function \a thunk#get 
        /// will return.
        Result get() const
        {
            if (!closure_)
            {
                boost::throw_exception(
                    std::logic_error("Closure uninitialized"));
            }

            closure_();

            // wait for the result (yield control)
            return (*this->impl_)->get_data(0);
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            util::block_profiler_wrapper<thunk_direct_tag> bp(apply_logger_);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (hpx::applier::get_applier().address_is_local(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                    components::get_component_type<typename Action::component_type>()));
                (*this->impl_)->set_data(0, 
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
        /// \param th     [in] The thunk.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        static void invoke(hpx::lcos::thunk<Action,Result> *th, 
                           naming::id_type const& gid)
        {
            if (!((*th->impl_)->is_data()))
              th->apply(gid);
        }

    public:
        /// Construct a new \a thunk instance. The \a thread 
        /// supplied to the function \a thunk#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this thunk instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// 
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               thunk instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        thunk(naming::gid_type const& gid)
          : apply_logger_("thunk_direct::apply"),
            closure_(boost::bind(invoke, 
                naming::id_type(gid, naming::id_type::unmanaged)))
        { }
        thunk(naming::id_type const& gid)
          : apply_logger_("thunk_direct::apply"),
            closure_(boost::bind(invoke, this, gid))
        { }

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
            util::block_profiler_wrapper<thunk_direct_tag> bp(apply_logger_);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (hpx::applier::get_applier().address_is_local(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                    components::get_component_type<typename Action::component_type>()));
                (*this->impl_)->set_data(
                    0, Action::execute_function(addr.address_, arg0));
            }
            else {
                // remote execution
                hpx::applier::apply_c<Action>(addr, this->get_gid(), gid, arg0);
            }
        }

    private:
        /// Invoke the action if the data is not ready.
        ///
        /// \param th     [in] The thunk.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the
        ///               apply operation for the embedded action.
        template <typename Arg0>
        static void invoke1(hpx::lcos::thunk<Action,Result> *th, 
                            naming::id_type const& gid, Arg0 const arg0)
        {
            if (!((*th->impl_)->is_data()))
                th->apply(gid, arg0);
        }

    public:
        /// Construct a new \a thunk instance. The \a thread 
        /// supplied to the function \a thunk#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this thunk instance has been returned.
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
        ///               thunk instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        thunk(naming::gid_type const& gid, Arg0 const& arg0)
          : apply_logger_("thunk_direct::apply"),
            closure_(boost::bind(invoke1, this, 
                naming::id_type(gid, naming::id_type::unmanaged), arg0))
        { }
        template <typename Arg0>
        thunk(naming::id_type const& gid, Arg0 const& arg0)
          : apply_logger_("thunk_direct::apply"),
            closure_(boost::bind(invoke1, this, gid, arg0))
        { }

        // pull in remaining constructors
        #include <hpx/lcos/thunk_constructors_direct.hpp>

        util::block_profiler<thunk_direct_tag> apply_logger_;
        boost::function<void()> closure_;
    };

}}

#endif
