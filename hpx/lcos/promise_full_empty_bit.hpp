//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PROMISE_FULL_EMPTY_FEB_03_2009_0841AM)
#define HPX_LCOS_PROMISE_FULL_EMPTY_FEB_03_2009_0841AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/value_or_error.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/exception_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Data, int N>
    struct data_store
    {
    private:
        // make sure N is in a reasonable range
        BOOST_STATIC_ASSERT(N > 0);   // N must be greater than zero

    public:
        Data& operator[](int slot)
        {
            return data_[slot];
        }

        Data const& operator[](int slot) const
        {
            return data_[slot];
        }

        Data data_[N];
    };

    template <typename Data>
    struct data_store<Data, 1>
    {
        Data& operator[](int /*slot*/)
        {
            return data_;
        }

        Data const& operator[](int /*slot*/) const
        {
            return data_;
        }

        Data data_;
    };

    /// A promise can be used by a single thread to invoke a (remote)
    /// action and wait for the result.
    template <typename Result, typename RemoteResult, int N>
    class promise : public lcos::base_lco_with_value<Result, RemoteResult>
    {
    protected:
        typedef Result result_type;
        typedef boost::exception_ptr error_type;
        typedef util::value_or_error<result_type> data_type;

        result_type handle_error(data_type const& d, error_code &ec)
        {
            // an error has been reported in the meantime, throw or set
            // the error code
            if (&ec == &throws) {
                // REVIEW: should HPX_RETHROW_EXCEPTION be used instead?
                boost::rethrow_exception(d.get_error());
                // never reached
            }
            else {
                try {
                    boost::rethrow_exception(d.get_error());
                }
                catch (hpx::exception const& he) {
                    ec = make_error_code(he.get_error(), he.what(),
                        hpx::rethrow);
                }
            }
            return result_type();
        }

        static bool
        verify_slot(int slot, char const* function_name, error_code& ec = throws)
        {
            if (slot < 0 || slot >= N) {
                HPX_THROWS_IF(ec, bad_parameter,
                    function_name, "slot index out of range");
                return false;
            }
            return true;
        }

        template <typename T>
        void set_data_impl(int slot, BOOST_FWD_REF(T) result)
        {
            if (!verify_slot(slot, "promise<>::set_data_impl"))
                return;

            // set the received result, reset error status
            try {
                // store the value
                typedef typename boost::remove_const<
                    typename hpx::util::detail::remove_reference<T>::type
                >::type naked_type;

                data_[slot].set(
                    traits::get_remote_result<Result, naked_type>::call(
                        boost::forward<T>(result)));
            }
            catch (hpx::exception const&) {
                // store the error instead
                data_[slot].set(boost::current_exception());
            }
        }

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_future };

        promise()
          : back_ptr_(0)
        {}

        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset(int slot, error_code& ec = throws)
        {
            if (!verify_slot(slot, "promise<>::reset", ec))
                return;

            data_[slot].set_empty(ec);
        }

        /// Return whether or not the data is available for this
        /// \a promise.
        bool is_ready(int slot, error_code& ec = throws) const
        {
            if (!verify_slot(slot, "promise<>::is_ready", ec))
                return false;

            return !(data_[slot].is_empty());
        }

        /// Get the result of the requested action. This call blocks (yields
        /// control) if the result is not ready. As soon as the result has been
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function will return.
        ///
        /// \param slot   [in] The number of the slot the value has to be
        ///               returned for. This number must be positive, but
        ///               smaller than the template parameter \a N.
        /// \param ec     [in,out] this represents the error status on exit,
        ///               if this is pre-initialized to \a hpx#throws
        ///               the function will throw on error instead. If the
        ///               operation blocks and is aborted because the object
        ///               went out of scope, the code \a hpx#yield_aborted is
        ///               set or thrown.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and
        ///               error description if <code>&ec == &throws</code>.
        result_type get_data(int slot, error_code& ec = throws)
        {
            if (!verify_slot(slot, "promise<>::get_data", ec))
                return result_type();

            // yields control if needed
            data_type d;
            data_[slot].read(d, ec);      // copies the data out of the store
            if (ec) return result_type();

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_error).
            if (!d.stores_value())
                return handle_error(d, ec);

            // no error has been reported, return the result
            return boost::move(d.get_value());
        }

        result_type move_data(int slot, error_code& ec = throws)
        {
            if (!verify_slot(slot, "promise<>::move_data", ec))
                return result_type();

            // yields control if needed
            data_type d;
            data_[slot].read_and_empty(d, ec); // moves the data from the store
            if (ec) return result_type();

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_error).
            if (!d.stores_value())
                return handle_error(d, ec);

            // no error has been reported, return the result
            return boost::move(d.get_value());
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_data(int slot, BOOST_FWD_REF(T) result)
        {
            return set_data_impl(slot, boost::forward<RemoteResult>(result));
        }

        template <typename T>
        void set_local_data(int slot, BOOST_FWD_REF(T) result)
        {
            return set_data_impl(slot, boost::forward<result_type>(result));
        }

        // trigger the future with the given error condition
        void set_error(int slot, boost::exception_ptr const& e)
        {
            if (!verify_slot(slot, "promise<>::set_error"))
                return;

            // store the error code
            data_[slot].set(data_type(e));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_result (BOOST_RV_REF(RemoteResult) result)
        {
            // set the received result, reset error status
            set_data(0, boost::move(result));
        }

        void set_error (boost::exception_ptr const& e)
        {
            set_error(0, e);        // set the received error
        }

        Result get_value()
        {
            return boost::move(get_data(0));
        }

        Result move_value()
        {
            return boost::move(move_data(0));
        }

        naming::id_type get_gid() const
        {
            return naming::id_type(
                naming::strip_credit_from_cgid(get_base_gid())
              , naming::id_type::unmanaged);
        }

        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }

    private:
        template <typename, typename>
        friend class components::managed_component;

        void set_back_ptr(components::managed_component<promise>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }

        data_store<util::full_empty<data_type>, N> data_;
        components::managed_component<promise>* back_ptr_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct log_on_exit
    {
        log_on_exit(boost::shared_ptr<T> const& impl, error_code const& ec)
          : impl_(impl), ec_(ec)
        {
            LLCO_(info) << "promise::get(" << impl_->get_gid()
                        << "), throws(" << ((&ec_ == &throws) ? "true" : "false")
                        << ")";
        }
        ~log_on_exit()
        {
            LLCO_(info) << "promise::got(" << impl_->get_gid()
                        << "), throws(" << ((&ec_ == &throws) ? "true" : "false")
                        << ")";
        }
        boost::shared_ptr<T> const& impl_;
        error_code const& ec_;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    /// A promise can be used by a single \a thread to invoke a
    /// (remote) action and wait for the result. The result is expected to be
    /// sent back to the promise using the LCO's set_event action
    ///
    /// A promise is one of the simplest synchronization primitives
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result. The \a promise
    /// allows to synchronize exactly one \a thread (the one passed during
    /// construction time).
    ///
    /// \code
    ///     // Create the promise (the expected result is a id_type)
    ///     lcos::promise<naming::id_type> f;
    ///
    ///     // initiate the action supplying the promise as a
    ///     // continuation
    ///     appy<some_action>(new continuation(f.get_gid()), ...);
    ///
    ///     // Wait for the result to be returned, yielding control
    ///     // in the meantime.
    ///     naming::id_type result = f.get();
    ///     // ...
    /// \endcode
    ///
    /// \tparam Result   The template parameter \a Result defines the type this
    ///                  promise is expected to return from
    ///                  \a promise#get.
    /// \tparam RemoteResult The template parameter \a RemoteResult defines the
    ///                  type this signalling_promise is expected to receive
    ///                  from the remote action.
    ///
    /// \note            The action executed by the promise must return a value
    ///                  of a type convertible to the type as specified by the
    ///                  template parameter \a RemoteResult
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult, int N>
    class promise
    {
    public:
        typedef detail::promise<Result, RemoteResult, N> wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

        /// Construct a new \a promise instance. The supplied
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
        promise()
          : impl_(new wrapping_type(new wrapped_type()))
        {
            LLCO_(info) << "promise::promise(" << impl_->get_gid() << ")";
        }

    protected:
        template <typename Impl>
        promise(Impl* impl)
          : impl_(impl)
        {}

    public:
        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            (*impl_)->reset(0);
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type get_gid() const
        {
            return (*impl_)->get_gid();
        }

        /// \brief Return the global id of this \a future instance
        naming::gid_type get_base_gid() const
        {
            return (*impl_)->get_base_gid();
        }

        /// Return whether or not the data is available for this
        /// \a promise.
        bool is_ready() const
        {
            return (*impl_)->is_ready(0);
        }

        /// Return whether this instance has been properly initialized
        bool valid() const
        {
            return impl_;
        }

        typedef Result result_type;

        ~promise()
        {}

        /// Get the result of the requested action. This call blocks (yields
        /// control) if the result is not ready. As soon as the result has been
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a eager_future#get will return.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               blocked (yielded) while waiting for the result.
        /// \param ec     [in,out] this represents the error status on exit,
        ///               if this is pre-initialized to \a hpx#throws
        ///               the function will throw on error instead. If the
        ///               operation blocks and is aborted because the object
        ///               went out of scope, the code \a hpx#yield_aborted is
        ///               set or thrown.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and
        ///               error description if <code>&ec == &throws</code>.
        Result get(int slot, error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return boost::move((*impl_)->get_data(slot, ec));
        }
        Result get(error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return boost::move((*impl_)->get_data(0, ec));
        }

        ///
        Result move_out(int slot, error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return boost::move((*impl_)->move_data(slot, ec));
        }
        Result move_out(error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return boost::move((*impl_)->move_data(0, ec));
        }

        ///
        void set(int slot, RemoteResult const& result)
        {
            return (*impl_)->set_data(slot, result);
        }
        void set(RemoteResult const& result)
        {
            return (*impl_)->set_data(0, result);
        }

        void set(int slot, BOOST_RV_REF(RemoteResult) result)
        {
          return (*impl_)->set_data(slot, boost::move(result));
        }
        void set(BOOST_RV_REF(RemoteResult) result)
        {
            return (*impl_)->set_data(0, boost::move(result));
        }

        void invalidate(int slot, boost::exception_ptr const& e)
        {
            (*impl_)->set_error(slot, e);   // set the received error
        }

        void invalidate(boost::exception_ptr const& e)
        {
            (*impl_)->set_error(0, e);      // set the received error
        }

        template <typename T>
        void set_local_data(int slot, BOOST_FWD_REF(T) result)
        {
            (*impl_)->set_local_data(0, boost::forward<Result>(result));
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    class promise<void, util::unused_type, N>
    {
    public:
        typedef detail::promise<util::unused_type, util::unused_type, N>
            wrapped_type;
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
        promise()
          : impl_(new wrapping_type(new wrapped_type()))
        {
            LLCO_(info) << "promise<void>::promise(" << impl_->get_gid() << ")";
        }

    protected:
        template <typename Impl>
        promise(Impl* impl)
          : impl_(impl)
        {}

    public:
        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            (*impl_)->reset(0);
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type get_gid() const
        {
            return (*impl_)->get_gid();
        }

        /// \brief Return the global id of this \a future instance
        naming::gid_type get_base_gid() const
        {
            return (*impl_)->get_base_gid();
        }

        /// Return whether or not the data is available for this
        /// \a promise.
        bool is_ready() const
        {
            return (*impl_)->is_ready(0);
        }

        typedef util::unused_type result_type;

        ~promise()
        {}

        /// Get the result of the requested action. This call blocks (yields
        /// control) if the result is not ready. As soon as the result has been
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function \a eager_future#get will return.
        ///
        /// \param self   [in] The \a thread which will be unconditionally
        ///               blocked (yielded) while waiting for the result.
        /// \param ec     [in,out] this represents the error status on exit,
        ///               if this is pre-initialized to \a hpx#throws
        ///               the function will throw on error instead.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_error), this function will throw an
        ///               exception encapsulating the reported error code and
        ///               error description if <code>&ec == &throws</code>.
        util::unused_type get(int slot, error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return (*impl_)->get_data(slot, ec);
        }

        util::unused_type get(error_code& ec = throws) const
        {
            detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
            return (*impl_)->get_data(0, ec);
        }

        void set(int slot)
        {
            return (*impl_)->set_data(slot, util::unused_type());
        }

        void set()
        {
            return (*impl_)->set_data(0, util::unused_type());
        }

        void invalidate(int slot, boost::exception_ptr const& e)
        {
            (*impl_)->set_error(slot, e); // set the received error
        }

        void invalidate(boost::exception_ptr const& e)
        {
            (*impl_)->set_error(0, e); // set the received error
        }

    protected:
        boost::shared_ptr<wrapping_type> impl_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Result, typename RemoteResult, int N>
    struct component_type_database<
        lcos::detail::promise<Result, RemoteResult, N> >
    {
        static components::component_type get()
        {
            return component_type_database<
                lcos::base_lco_with_value<Result, RemoteResult>
            >::get();
        }

        static void set(components::component_type t)
        {
            component_type_database<
                lcos::base_lco_with_value<Result, RemoteResult>
            >::set(t);
        }
    };
}}

#endif
