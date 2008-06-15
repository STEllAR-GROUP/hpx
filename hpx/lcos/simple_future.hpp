//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SIMPLE_FUTURE_JUN_12_2008_0654PM)
#define HPX_LCOS_SIMPLE_FUTURE_JUN_12_2008_0654PM

#include <boost/throw_exception.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/server/wrapper.hpp>

namespace hpx { namespace lcos { namespace detail 
{
    /// A simple_future can be used by a single thread to invoke a (remote) 
    /// action and wait for the result. The result is expected to be sent back 
    /// to the simple_future using the LCO's set_event action
    template <typename Result>
    class simple_future : public base_lco_with_value<Result>
    {
    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_simple_future};
        
        ///
        template <typename F>
        simple_future(threadmanager::px_thread_self& target, F f,
                naming::id_type const& my_gid)
          : target_thread_(target), result_(), code_(hpx::success)
        {
            // initiate the required action, provide the continuation 
            // information,function takes over ownership of allocated 
            // continuation
            f (new components::continuation(my_gid));
        }
        
        /// 
        Result get_result(threadmanager::px_thread_self& self) const
        {
            self.yield(threadmanager::suspended);
            if (code_)
            {
                boost::throw_exception(
                    boost::system::system_error(code_, error_msg_));
            }
            return result_;
        };

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// trigger the future, set the result
        threadmanager::thread_state 
        set_event_proc (threadmanager::px_thread_self&, applier::applier& appl,
            Result const& result) 
        {
            // set the received result
            result_ = result;
            
            // re-activate the target thread
            appl.get_thread_manager().set_state(
                target_thread_.get_thread_id(), threadmanager::pending);

            // this thread has nothing more to do
            return threadmanager::terminated;
        }
        
        /// trigger the future with the given error condition
        threadmanager::thread_state set_error_proc (
            threadmanager::px_thread_self&, applier::applier& appl,
            hpx::error code, std::string msg)
        {
            // store the error code
            code_ = make_error_code(code);
            error_msg_ = msg;
            
            // re-activate the target thread
            appl.get_thread_manager().set_state(
                target_thread_.get_thread_id(), threadmanager::pending);

            // this thread has nothing more to do
            return threadmanager::terminated;
        }

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef components::action2<
            simple_future, base_lco::set_error, hpx::error, std::string,
            &simple_future::set_error_proc
        > set_error_action;

        typedef components::action1<
            simple_future, base_lco::set_event, Result const&, 
            &simple_future::set_event_proc
        > set_event_action;

    private:
        threadmanager::px_thread_self& target_thread_;
        Result result_;
        boost::system::error_code code_;
        std::string error_msg_;
    };
    
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class simple_future 
    {
    private:
        typedef detail::simple_future<Result> wrapped_type;
        typedef components::wrapper<wrapped_type> wrapping_type;
        
        // avoid warning about usage of this in member initializer list
        simple_future* This() { return this; }
        
    public:
        template <typename F>
        simple_future(threadmanager::px_thread_self& target, F f)
          : impl_(new wrapping_type())
        {
            impl_->set_wrapped(new wrapped_type(target, f, get_gid()));
        }

        ///
        Result get_result(threadmanager::px_thread_self& self) const
        {
            return (*impl_)->get_result(self);
        }

        naming::id_type get_gid() const
        {
            return impl_->get_gid();
        }

    private:
        components::wrapper<detail::simple_future<Result> >* impl_;
    };

}}

#endif
