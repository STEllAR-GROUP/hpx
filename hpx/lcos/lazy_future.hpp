//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LAZY_FUTURE_JUN_27_2008_0446PM)
#define HPX_LCOS_LAZY_FUTURE_JUN_27_2008_0446PM

#include <boost/intrusive_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/full_empty_memory.hpp>
#include <hpx/components/action.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/server/wrapper.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/simple_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class lazy_future lazy_future.hpp hpx/lcos/lazy_future.hpp
    ///
    /// A lazy_future can be used by a single \a px_thread to invoke a 
    /// (remote) action and wait for the result. The result is expected to be 
    /// sent back to the lazy_future using the LCO's set_event action
    ///
    /// A lazy_future is one of the simplest synchronization primitives 
    /// provided by HPX. It allows to synchronize on a lazily evaluated remote
    /// operation returning a result of the type \a Result. 
    ///
    /// \tparam Action   The template parameter \a Action defines the action 
    ///                  to be executed by this lazy_future instance. The 
    ///                  arguments \a arg0,... \a argN are used as parameters 
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this 
    ///                  lazy_future is expected to return from 
    ///                  \a lazy_future#get_result.
    ///
    /// \note            The action executed using the lazy_future as a 
    ///                  continuation must return a value of a type convertible 
    ///                  to the type as specified by the template parameter 
    ///                  \a Result.
    template <typename Action, typename Result>
    class lazy_future : public simple_future<Result>
    {
    private:
        typedef simple_future<Result> base_type;

    public:
        /// Construct a new \a lazy_future instance. The \a px_thread 
        /// supplied to the function \a lazy_future#get_result will be 
        /// notified as soon as the result of the operation associated with 
        /// this lazy_future instance has been returned.
        /// 
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               lazy_future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        lazy_future(applier::applier& appl, naming::id_type const& gid)
        {}

        /// Get the result of the requested action. This call blocks (yields 
        /// control) if the result is not ready. As soon as the result has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a lazy_future#get_result will return.
        ///
        /// \param self   [in] The \a px_thread which will be unconditionally
        ///               while waiting for the result. 
        /// \param appl   [in] The \a applier instance to be used to execute 
        ///               the embedded action.
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Result get_result(threadmanager::px_thread_self& self,
            applier::applier& appl, naming::id_type const& gid) const
        {
            // initialize the operation
            appl.apply_c<Action>(this->get_gid(appl), gid);

            // wait for the result (yield control)
            return (*this->impl_)->get_result(self);
        }

        /// Get the result of the requested action. This call blocks (yields 
        /// control) if the result is not ready. As soon as the result has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a lazy_future#get_result will return.
        ///
        /// \param self   [in] The \a px_thread which will be unconditionally
        ///               while waiting for the result. 
        /// \param appl   [in] The \a applier instance to be used to execute 
        ///               the embedded action.
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the 
        ///               apply operation for the embedded action.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        template <typename Arg0>
        Result get_result(threadmanager::px_thread_self& self,
            applier::applier& appl, naming::id_type const& gid,
            Arg0 const& arg0) const
        {
            // initialize the operation
            appl.apply_c<Action>(this->get_gid(appl), gid, arg0);

            // wait for the result (yield control)
            return (*this->impl_)->get_result(self);
        }

        // pull in remaining get_result's
        #include <hpx/lcos/lazy_future_get_results.hpp>
    };

}}

#endif
