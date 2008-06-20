//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SIMPLE_FUTURE_JUN_12_2008_0654PM)
#define HPX_LCOS_SIMPLE_FUTURE_JUN_12_2008_0654PM

#include <boost/intrusive_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant.hpp>
#include <boost/detail/atomic_count.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/full_empty_memory.hpp>
#include <hpx/components/action.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/server/wrapper.hpp>

namespace hpx { namespace lcos { namespace detail 
{
    // A simple_future can be used by a single thread to invoke a (remote) 
    // action and wait for the result. 
    template <typename Result>
    class simple_future : public base_lco_with_value<Result>
    {
    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_simple_future};

        simple_future(threadmanager::px_thread_self& self)
          : target_thread_(self), use_count_(0)
        {
        }

        Result get_result() 
        {
            // yields control if needed
            data_type d;
            data_.read(target_thread_, d);

            // the thread has been re-activated by one of the actions 
            // supported by this simple_future (see \a simple_future::set_event
            // and simple_future::set_error).
            if (1 == d.which())
            {
                // an error has been reported in the meantime, throw 
                error_type e = boost::get<error_type>(d);
                boost::throw_exception(
                    boost::system::system_error(e.first, e.second));
            }

            // no error has been reported, return the result
            return boost::get<result_type>(d);
        };

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        // trigger the future, set the result
        threadmanager::thread_state 
        set_result (threadmanager::px_thread_self& self, 
            applier::applier& appl, Result const& result)
        {
            // set the received result, reset error status
            data_.set(self, result);

            // re-activate the target thread if previously yielded
            appl.get_thread_manager().set_state(
                target_thread_.get_thread_id(), threadmanager::pending);

            // this thread has nothing more to do
            return threadmanager::terminated;
        }

        // trigger the future with the given error condition
        threadmanager::thread_state 
        set_error (threadmanager::px_thread_self& self, applier::applier& appl,
            hpx::error code, std::string msg)
        {
            // store the error code
            data_.set(self, error_type(make_error_code(code), msg));

            // re-activate the target thread
            appl.get_thread_manager().set_state(
                target_thread_.get_thread_id(), threadmanager::pending);

            // this thread has nothing more to do
            return threadmanager::terminated;
        }

    private:
        typedef Result result_type;
        typedef std::pair<boost::system::error_code, std::string> error_type;
        typedef boost::variant<result_type, error_type> data_type;

        threadmanager::px_thread_self& target_thread_;
        lcos::full_empty<data_type> data_;
        std::string error_msg_;

    public:
        boost::detail::atomic_count use_count_;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the simple_future actions
HPX_SERIALIZE_ACTION(hpx::lcos::detail::base_lco_with_value<hpx::naming::id_type>::set_result_action);
HPX_SERIALIZE_ACTION(hpx::lcos::detail::base_lco_with_value<hpx::naming::id_type>::set_error_action);
HPX_SERIALIZE_ACTION(hpx::lcos::detail::base_lco_with_value<double>::set_result_action);
HPX_SERIALIZE_ACTION(hpx::lcos::detail::base_lco_with_value<double>::set_error_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// A simple_future can be used by a single \a px_thread to invoke a 
    /// (remote) action and wait for the result. The result is expected to be 
    /// sent back to the simple_future using the LCO's set_event action
    ///
    /// A simple_future is one of the simplest synchronization primitives 
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result. The \a simple_future
    /// allows to synchronize exactly one \a px_thread (the one passed during 
    /// construction time).
    ///
    /// \code
    ///     // Create the simple_future supplying the thread to re-activate
    ///     lcos::simple_future<naming::id_type> f(thread_self);
    ///
    ///     // initiate the action supplying the simple_future as a 
    ///     // continuation
    ///     applier_.appy<some_action>(new continuation(f.get_gid()), ...);
    ///
    ///     // Wait for the result to be returned, yielding control 
    ///     // in the meantime.
    ///     naming::id_type result = f.get_result();
    ///     // ...
    /// \endcode
    template <typename Result>
    class simple_future 
    {
    private:
        typedef detail::simple_future<Result> wrapped_type;
        typedef components::wrapper<simple_future, wrapped_type> wrapping_type;

    public:
        /// Construct a new \a simple_future instance. The supplied 
        /// \a px_thread will be notified as soon as the result of the 
        /// operation associated with this simple_future instance has been 
        /// returned.
        /// 
        /// \param self   [in] The \a px_thread which will be notified as soon 
        ///               as the result has been returned. This thread will be
        ///               notified (re-activated) as soon as this simple_future
        ///               instance receives the result.
        ///
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_event action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               simple_future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        simple_future(threadmanager::px_thread_self& self)
          : impl_(new wrapping_type(new wrapped_type(self)))
        {}

        /// Get the result of the requested action. This call blocks (yields 
        /// control) if the result is not ready. As soon as the result has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the \a get_result() will return.
        ///
        /// \param self   [in] The \a px_thread which will be unconditionally
        ///               suspended while waiting for the result. This needs to 
        ///               be the same thread as has been used to 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Result get_result() const
        {
            return (*impl_)->get_result();
        }

        /// \brief Return the global id of this \a simple_future instance
        naming::id_type get_gid(applier::applier& appl) const
        {
            return impl_->get_gid(appl);
        }

    private:
        boost::intrusive_ptr<wrapping_type> impl_;
    };

}}

#endif
