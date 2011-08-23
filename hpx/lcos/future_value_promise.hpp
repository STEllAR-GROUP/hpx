//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_VALUE_PROMISE_FEB_02_2009_0658)
#define HPX_LCOS_FUTURE_VALUE_JUN_12_2008_0654PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/future.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/if.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail 
{
    /// A future_value can be used by a single thread to invoke a (remote) 
    /// action and wait for the result. 
    template <typename Result>
    class future_value : public lcos::base_lco_with_value<Result>
    {
    private:
        typedef Result result_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_future };

        future_value()
          : promise_(), data_(promise_.get_future())
        {
        }

        /// Reset the future_value to allow to restart an asynchronous 
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            promise_ = promise<result_type>();
            data_ = promise_.get_future();
        }

        /// Get the result of the requested action. This call blocks (yields 
        /// control) if the result is not ready. As soon as the result has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a lazy_future#get will return.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Result get_data() 
        {
            return data_.get();
        };

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_result (result_type const& result)
        {
            // set the received result
            promise_.set_value(result);
        }

        // trigger the future with the given error condition
        void set_error (hpx::error code, std::string const& msg)
        {
            // store the error code
            try {
                HPX_RETHROW_EXCEPTION(code, "future_value::set_error", msg);
            }
            catch (...) {
                promise_.set_exception(boost::current_exception());
            }
        }

    private:
        util::promise<result_type> promise_;
        util::unique_future<result_type> data_;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class future_value future_value.hpp hpx/lcos/future_value.hpp
    ///
    /// A future_value can be used by a single \a thread to invoke a 
    /// (remote) action and wait for the result. The result is expected to be 
    /// sent back to the future_value using the LCO's set_event action
    ///
    /// A future_value is one of the simplest synchronization primitives 
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result. The \a future_value
    /// allows to synchronize exactly one \a thread (the one passed during 
    /// construction time).
    ///
    /// \code
    ///     // Create the future_value (the expected result is a id_type)
    ///     lcos::future_value<naming::id_type> f;
    ///
    ///     // initiate the action supplying the future_value as a 
    ///     // continuation
    ///     applier_.appy<some_action>(new continuation(f.get_gid()), ...);
    ///
    ///     // Wait for the result to be returned, yielding control 
    ///     // in the meantime.
    ///     naming::id_type result = f.get(thread_self);
    ///     // ...
    /// \endcode
    ///
    /// \tparam Result   The template parameter \a Result defines the type this 
    ///                  future_value is expected to return from 
    ///                  \a future_value#get.
    ///
    /// \note            The action executed using the future_value as a 
    ///                  continuation must return a value of a type convertible 
    ///                  to the type as specified by the template parameter 
    ///                  \a Result
    template <typename Result>
    class future_value 
    {
    protected:
        typedef typename boost::mpl::if_<
                boost::is_same<Result, void>,
                detail::future_value<Result>,
                detail::future_value<util::unused_type>
            >::type wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

        /// Construct a new \a future instance. The supplied 
        /// \a thread will be notified as soon as the result of the 
        /// operation associated with this future instance has been 
        /// returned.
        /// 
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        future_value()
          : impl_(new wrapping_type(new wrapped_type()))
        {}

        /// \brief Return the global id of this \a future instance
        naming::gid_type get_gid() const
        {
            return impl_->get_gid();
        }

        /// Reset the future_value to allow to restart an asynchronous 
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            (*impl_)->reset();
        }

    public:
        typedef Result result_type;

        ~future_value()
        {}

        /// Get the result of the requested action. This call blocks (yields 
        /// control) if the result is not ready. As soon as the result has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a eager_future#get will return.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Result get() const
        {
            return (*impl_)->get_data();
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };

}}

#endif
