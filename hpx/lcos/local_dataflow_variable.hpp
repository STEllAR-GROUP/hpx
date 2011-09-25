//  Copyright (c) 2010-2011 Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
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
    struct find_value
    {
        static Value call(RemoteValue const& rhs)
        {
            return Value(rhs);
        }
    };

    template <typename Value>
    struct find_value<Value, Value>
    {
        static Value const& call(Value const& rhs)
        {
            return rhs;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// A local_dataflow_variable is a declarative variable. Attempting to read
    /// an uninitialized variable will cause the thread to suspend until
    /// a value is bound to that variable.
    ///
    /// Note: we do not support partial values nor unification, so a
    /// variable should only be set once.
    template <typename Value, typename RemoteValue>
    class local_dataflow_variable : public lcos::base_lco_with_value<Value,RemoteValue>
    {
    protected:
        typedef Value value_type;
        typedef boost::exception_ptr error_type;
        typedef boost::variant<value_type, error_type> data_type;

    public:
        enum { value = components::component_dataflow_variable };

        local_dataflow_variable() {}

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
        Value read(error_code& ec = throws) 
        {
            // Suspend calling thread, if necessary
            data_type d;
            data_.read(d, ec);

            if (&ec != &throws && !ec)
                return Value();

            if (1 == d.which())
            {
                // an error has been reported in the meantime, throw or set
                // the error code 
                error_type e = boost::get<error_type>(d);

                if (&ec == &throws)
                {
                    // REVIEW: should HPX_RETHROW_EXCEPTION be used instead?
                    boost::rethrow_exception(e);
                    // never reached
                }

                else
                {
                    try
                    {
                        boost::rethrow_exception(e);
                    }

                    catch (hpx::exception const& he)
                    {
                        ec = he.get_error_code(hpx::rethrow);
                        return Value();
                    }
                }       
            }

            if (&ec != &throws)
                ec = make_success_code();

            // Continue execution
            return boost::get<value_type>(d);
        };

        // trigger the variable with the given error condition
        void set_error(boost::exception_ptr const& e)
        {
            // store the error code
            data_.set(data_type(e));
        }

        /// Bind the variable with a value.
        ///
        /// \param value  [in] The value to bind with the variable.
        void bind(RemoteValue const& value)
        {
            data_.set(data_type(find_value<Value, RemoteValue>::call(value)));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        void set_result(RemoteValue const& value)
        {
            bind(value);
        }

        Value get_value()
        {
            return read();
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
    /// A local_dataflow_variable is a declarative variable. Attempting to read
    /// an uninitialized variable will cause the thread to suspend until
    /// a value is bound to that variable.
    ///
    /// Note: we do not support partial values nor unification, so a
    /// variable should only be set once.
    template<>
    class local_dataflow_variable<naming::id_type, naming::gid_type>
      : public lcos::base_lco_with_value<naming::id_type, naming::gid_type>
    {
    protected:
        typedef naming::id_type value_type;
        typedef boost::exception_ptr error_type;
        typedef boost::variant<value_type, error_type> data_type;

    public:
        local_dataflow_variable() {}

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
        value_type read(error_code& ec = throws) 
        {
            // Suspend calling thread, if necessary
            data_type d;
            data_.read(d, ec);

            if (&ec != &throws && !ec)
                return value_type();

            if (1 == d.which())
            {
                // an error has been reported in the meantime, throw or set
                // the error code 
                error_type e = boost::get<error_type>(d);

                if (&ec == &throws)
                {
                    // REVIEW: should HPX_RETHROW_EXCEPTION be used instead?
                    boost::rethrow_exception(e);
                    // never reached
                }

                else
                {
                    try
                    {
                        boost::rethrow_exception(e);
                    }

                    catch (hpx::exception const& he)
                    {
                        ec = he.get_error_code(hpx::rethrow);
                        return value_type();
                    }
                }       
            }

            if (&ec != &throws)
                ec = make_success_code();

            // Continue execution
            return boost::get<naming::id_type>(d);
        };

        // trigger the variable with the given error condition
        void set_error(boost::exception_ptr const& e)
        {
            // store the error code
            data_.set(data_type(e));
        }

        /// Bind the variable with a value.
        ///
        /// \param value  [in] The value to bind with the variable.
        void bind(naming::gid_type const& value)
        {
            data_.set(
                data_type(naming::id_type(value, naming::id_type::managed)));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        void set_result(naming::gid_type const& value)
        {
            bind(value);
        }

        value_type get_value()
        {
            return read();
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
    /// \class local_dataflow_variable local_dataflow_variable.hpp 
    ///     hpx/lcos/local_dataflow_variable.hpp
    ///
    /// A local_dataflow_variable can be used to synchronize multiple threads on 
    /// the availability of a value. 
    ///
    /// \tparam Value   The template parameter \a Value defines the type this 
    ///                 local_dataflow_variable is expected to be bound with
    template <typename Value>
    struct local_dataflow_variable_remote_value
      : boost::mpl::identity<Value>
    {};

    template <typename Value, typename RemoteValue>
    class local_dataflow_variable
    {
    protected:
        typedef detail::local_dataflow_variable<Value, RemoteValue>
            wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        local_dataflow_variable()
          : impl_(new wrapping_type(new wrapped_type()))
        {}

        /// \brief Return the global id of this \a future instance
        naming::id_type const& get_gid() const
        {
            return (*impl_)->get_gid(impl_.get());
        }

    public:
        typedef Value value_type;

        ~local_dataflow_variable()
        {}

    public:
        /// Get the value of the local_dataflow_variable. This call blocks
        /// (yields control) if the value is not ready. As soon as the value has
        /// been bound and the waiting thread has been re-scheduled by the
        /// thread manager the function \a local_dataflow_variable#get will
        /// return.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               blocked (yielded) while waiting for the value. 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        Value get(error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return (*impl_)->read(ec);
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct local_dataflow_variable_remote_value<void>
      : boost::mpl::identity<util::unused_type>
    {};

    template<>
    class local_dataflow_variable<void, util::unused_type>
    {
    protected:
        typedef detail::local_dataflow_variable<
            util::unused_type, util::unused_type
        > wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        local_dataflow_variable()
          : impl_(new wrapping_type(new wrapped_type()))
        {}

        /// \brief Return the global id of this \a future instance
        naming::id_type const& get_gid() const
        {
            return (*impl_)->get_gid(impl_.get());
        }

    public:
        typedef util::unused_type value_type;

        ~local_dataflow_variable()
        {}

        /// Get the value of the local_dataflow_variable. This call blocks
        /// (yields control) if the value is not ready. As soon as the value
        /// has been bound and the waiting thread has been re-scheduled by the
        /// thread manager the function \a local_dataflow_variable#get will
        /// return.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               blocked (yielded) while waiting for the value. 
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and 
        ///               error description.
        util::unused_type get(error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return (*impl_)->read(ec);
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };
}}

#endif
