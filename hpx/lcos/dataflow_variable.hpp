//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DATAFLOW_VARIABLE_FEB_03_2009_0841AMM)
#define HPX_LCOS_DATAFLOW_VARIABLE_FEB_03_2009_0841AMM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/full_empty_memory.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/variant.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/identity.hpp>
  
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail 
{
    template <typename Value, typename RemoteValue>
    struct get_value
    {
        static Value call(RemoteValue const& rhs)
        {
            return Value(rhs);
        }
    };

    template <typename Value>
    struct get_value<Value, Value>
    {
        static Value const& call(Value const& rhs)
        {
            return rhs;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// A dataflow_variable is a declarative variable. Attempting to read
    /// an uninitialized variable will cause the thread to suspend until
    /// a value is bound to that variable.
    ///
    /// Note: we do not support partial values nor unification, so a
    /// variable should only be set once.
    template <typename Value, typename RemoteValue>
    class dataflow_variable : public lcos::base_lco_with_value<RemoteValue>
    {
    protected:
        typedef Value value_type;
        typedef boost::exception_ptr error_type;
        typedef boost::variant<value_type, error_type> data_type;

    public:
        enum { value = components::component_dataflow_variable };

        dataflow_variable() {}

        /// Reset the future_value to allow to restart an asynchronous 
        /// operation. Allows any subsequent set_data operation to succeed.
        //void reset()
        //{
        //    data_->set_empty();
        //}

        /// Return whether or not the data is available for this
        /// \a future_value.
        //bool is_data()
        //{
        //    return !(data_->is_empty());
        //}

        /// Get the value bound to the dataflow variable. The calling
        /// thread is suspended if the variable is uninitialized. When the
        /// variable becomes bound, all waiting threads will be reactivated.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               suspended while waiting for the value. 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Value read(void) 
        {
            // Suspend calling thread, if necessary
            data_type d;
            data_.read(d);

            // Continue execution
            return boost::get<value_type>(d);
        };

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        void bind(RemoteValue const& value)
        {
            data_.set(data_type(get_value<Value, RemoteValue>::call(value)));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // bind the value to the dataflow variable
        void set_result(RemoteValue const& value)
        {
            bind(value);
        }

        template <typename ManagedType>
        naming::id_type const& get_gid(ManagedType* p) const
        {
            if (!id_) {
                naming::gid_type gid = p->get_base_gid(); 
                naming::strip_credit_from_gid(gid);
                id_ = naming::id_type(gid, naming::id_type::unmanaged);
            }
            return id_;
        }

    private:
        util::full_empty<data_type> data_;
        mutable naming::id_type id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// A dataflow_variable is a declarative variable. Attempting to read
    /// an uninitialized variable will cause the thread to suspend until
    /// a value is bound to that variable.
    ///
    /// Note: we do not support partial values nor unification, so a
    /// variable should only be set once.
    template<>
    class dataflow_variable<naming::id_type, naming::gid_type>
      : public lcos::base_lco_with_value<naming::gid_type>
    {
    protected:
        typedef naming::id_type value_type;
        typedef boost::exception_ptr error_type;
        typedef boost::variant<value_type, error_type> data_type;

    public:
        dataflow_variable() {}

        /// Reset the future_value to allow to restart an asynchronous 
        /// operation. Allows any subsequent set_data operation to succeed.
        //void reset()
        //{
        //    data_->set_empty();
        //}

        /// Return whether or not the data is available for this
        /// \a future_value.
        //bool is_data()
        //{
        //    return !(data_->is_empty());
        //}

        /// Get the value bound to the dataflow variable. The calling
        /// thread is suspended if the variable is uninitialized. When the
        /// variable becomes bound, all waiting threads will be reactivated.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               suspended while waiting for the value. 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        value_type read(void) 
        {
            // Suspend calling thread, if necessary
            data_type d;
            data_.read(d);

            // Continue execution
            return boost::get<naming::id_type>(d);
        };

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        void bind(naming::gid_type const& value)
        {
            // store the value as a managed id
            data_.set(
                data_type(naming::id_type(value, naming::id_type::managed)));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the value
        void set_result(naming::gid_type const& value)
        {
            bind(value);
        }

        template <typename ManagedType>
        naming::id_type const& get_gid(ManagedType* p) const
        {
            if (!id_) {
                naming::gid_type gid = p->get_base_gid(); 
                naming::strip_credit_from_gid(gid);
                id_ = naming::id_type(gid, naming::id_type::unmanaged);
            }
            return id_;
        }

    private:
        util::full_empty<data_type> data_;
        mutable naming::id_type id_;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class future_value future_value.hpp hpx/lcos/future_value.hpp
    ///
    /// A future_value can be used by a single \a thread to invoke a 
    /// (remote) action and wait for the value. The value is expected to be 
    /// sent back to the future_value using the LCO's set_event action
    ///
    /// A future_value is one of the simplest synchronization primitives 
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a value of the type \a Value. The \a future_value
    /// allows to synchronize exactly one \a thread (the one passed during 
    /// construction time).
    ///
    /// \code
    ///     // Create the future_value (the expected value is a id_type)
    ///     lcos::future_value<naming::id_type> f;
    ///
    ///     // initiate the action supplying the future_value as a 
    ///     // continuation
    ///     applier_.appy<some_action>(new continuation(f.get_gid()), ...);
    ///
    ///     // Wait for the value to be returned, yielding control 
    ///     // in the meantime.
    ///     naming::id_type value = f.get(thread_self);
    ///     // ...
    /// \endcode
    ///
    /// \tparam Value   The template parameter \a Value defines the type this 
    ///                  future_value is expected to return from 
    ///                  \a future_value#get.
    ///
    /// \note            The action executed using the future_value as a 
    ///                  continuation must return a value of a type convertible 
    ///                  to the type as specified by the template parameter 
    ///                  \a Value
    template <typename Value>
    struct dataflow_variable_remote_value
      : boost::mpl::identity<Value>
    {};

    template <typename Value, typename RemoteValue>
    class dataflow_variable
    {
    protected:
        typedef detail::dataflow_variable<Value, RemoteValue> wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        /// Construct a new \a future instance. The supplied 
        /// \a thread will be notified as soon as the value of the 
        /// operation associated with this future instance has been 
        /// returned.
        /// 
        /// \note         The value of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_value action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        dataflow_variable()
          : impl_(new wrapping_type(new wrapped_type()))
        {
            LLCO_(info) << "dataflow_variable::dataflow_variable(" << impl_->get_gid() << ")";
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type const& get_gid() const
        {
            return (*impl_)->get_gid(impl_.get());
        }

        /// \brief Return the full address of this \a future instance
//         bool get_full_address(naming::full_address& fa) const
//         {
//             return impl_->get_full_address(fa);
//         }

        /// Reset the future_value to allow to restart an asynchronous 
        /// operation. Allows any subsequent set_data operation to succeed.
        //void reset()
        //{
        //    (*impl_)->reset();
        //}

    public:
        typedef Value value_type;

        ~dataflow_variable()
        {}

    public:
        /// Get the value of the requested action. This call blocks (yields 
        /// control) if the value is not ready. As soon as the value has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a eager_future#get will return.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               blocked (yielded) while waiting for the value. 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Value get(void) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_);
            return (*impl_)->read();
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct dataflow_variable_remote_value<void>
      : boost::mpl::identity<util::unused_type>
    {};

    template<>
    class dataflow_variable<void, util::unused_type>
    {
    protected:
        typedef detail::dataflow_variable<util::unused_type, util::unused_type> 
            wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        /// Construct a new \a future instance. The supplied 
        /// \a thread will be notified as soon as the value of the 
        /// operation associated with this future instance has been 
        /// returned.
        /// 
        /// \note         The value of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_value action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        dataflow_variable()
          : impl_(new wrapping_type(new wrapped_type()))
        {
            LLCO_(info) << "dataflow_variable<void>::dataflow_variable(" << impl_->get_gid() << ")";
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type const& get_gid() const
        {
            return (*impl_)->get_gid(impl_.get());
        }

        /// \brief Return the full address of this \a future instance
//         bool get_full_address(naming::full_address& fa) const
//         {
//             return impl_->get_full_address(fa);
//         }

        /// Reset the future_value to allow to restart an asynchronous 
        /// operation. Allows any subsequent set_data operation to succeed.
        //void reset()
        //{
        //    (*impl_)->reset();
        //}

    public:
        typedef util::unused_type value_type;

        ~dataflow_variable()
        {}

        /// Get the value of the requested action. This call blocks (yields 
        /// control) if the value is not ready. As soon as the value has been 
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a eager_future#get will return.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               blocked (yielded) while waiting for the value. 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        util::unused_type get(void) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_);
            return (*impl_)->read();
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };
}}

#endif
