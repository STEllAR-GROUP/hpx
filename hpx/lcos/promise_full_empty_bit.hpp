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
#include <hpx/util/future_data.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/exception_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Result, typename RemoteResult>
    class promise;
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Result, typename RemoteResult>
    struct managed_component_dtor_policy<
        lcos::detail::promise<Result, RemoteResult> >
    {
        typedef managed_object_is_lifetime_controlled type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// A promise can be used by a single thread to invoke a (remote)
    /// action and wait for the result.
    template <typename Result, typename RemoteResult>
    class promise
      : public lcos::base_lco_with_value<Result, RemoteResult>,
        public util::future_data<Result, RemoteResult>
    {
    protected:
        typedef util::future_data<Result, RemoteResult> future_data_type;
        typedef typename future_data_type::result_type result_type;
        typedef boost::exception_ptr error_type;
        typedef util::value_or_error<result_type> data_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_future };

        promise()
          : back_ptr_(0)
        {}

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~promise()
        {
            this->finalize();
            delete back_ptr_;
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_local_data(BOOST_FWD_REF(T) result)
        {
            return set_data(boost::forward<result_type>(result));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_result (BOOST_RV_REF(RemoteResult) result)
        {
            // set the received result, reset error status
            set_data(boost::move(result));
        }

        Result get_value()
        {
            return this->get_data();
        }

        Result move_value()
        {
            return this->move_data();
        }

        void set_error(boost::exception_ptr e)
        {
            return this->future_data_type::set_error(e);
        }

        // retrieve the gid of this promise
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

        util::full_empty<data_type> data_;
        components::managed_component<promise>* back_ptr_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void, util::unused_type>
      : public lcos::base_lco_with_value<void, util::unused_type>,
        public util::future_data<void, util::unused_type>
    {
    protected:
        typedef boost::exception_ptr error_type;
        typedef util::value_or_error<result_type> data_type;
        typedef util::future_data<void, util::unused_type> future_data_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_future };

        promise()
          : back_ptr_(0)
        {}

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~promise()
        {
            this->finalize();
            delete back_ptr_;
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_local_data(BOOST_FWD_REF(T) result)
        {
            return set_data(boost::forward<result_type>(result));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_result (BOOST_RV_REF(util::unused_type) result)
        {
            // set the received result, reset error status
            set_data(boost::move(result));
        }

        void get_value()
        {
            this->get_data();
        }

        void move_value()
        {
            this->move_data();
        }

        void set_error(boost::exception_ptr e)
        {
            return this->future_data_type::set_error(e);
        }

        // retrieve the gid of this promise
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

        util::full_empty<data_type> data_;
        components::managed_component<promise>* back_ptr_;
    };

//     ///////////////////////////////////////////////////////////////////////////
//     template <typename T>
//     struct log_on_exit
//     {
//         log_on_exit(boost::shared_ptr<T> const& impl, error_code const& ec)
//           : impl_(impl), ec_(ec)
//         {
//             LLCO_(info) << "promise::get(" << impl_->get_gid()
//                         << "), throws(" << ((&ec_ == &throws) ? "true" : "false")
//                         << ")";
//         }
//         ~log_on_exit()
//         {
//             LLCO_(info) << "promise::got(" << impl_->get_gid()
//                         << "), throws(" << ((&ec_ == &throws) ? "true" : "false")
//                         << ")";
//         }
//         boost::shared_ptr<T> const& impl_;
//         error_code const& ec_;
//     };
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
    ///     naming::id_type result = f.get_future().get();
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
    template <typename Result, typename RemoteResult>
    class promise
    {
    public:
        typedef detail::promise<Result, RemoteResult> wrapped_type;
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
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info) << "promise::promise(" << impl_->get_gid() << ")";
        }

    protected:
        template <typename Impl>
        promise(Impl* impl)
          : impl_(impl), future_obtained_(false)
        {}

    public:
        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            (*impl_)->reset();
            future_obtained_ = false;
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
            return (*impl_)->is_ready();
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
//         Result get(int slot, error_code& ec = throws) const
//         {
//             detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
//             return boost::move((*impl_)->get_data(slot, ec));
//         }
//         Result get(error_code& ec = throws) const
//         {
//             detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
//             return boost::move((*impl_)->get_data(0, ec));
//         }
//
//         ///
//         Result move_out(int slot, error_code& ec = throws) const
//         {
//             detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
//             return boost::move((*impl_)->move_data(slot, ec));
//         }
//         Result move_out(error_code& ec = throws) const
//         {
//             detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
//             return boost::move((*impl_)->move_data(0, ec));
//         }

        lcos::future<Result, RemoteResult> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<Result>::get_future",
                    "future already has been retrieved from this packaged_task");
                return lcos::future<Result, RemoteResult>();
            }

            future_obtained_ = true;
            return lcos::future<Result, RemoteResult>(impl_->get());
        }

        ///
        void set(RemoteResult const& result)
        {
            return (*impl_)->set_data(result);
        }

        void set(BOOST_RV_REF(RemoteResult) result)
        {
            return (*impl_)->set_data(boost::move(result));
        }

        void invalidate(boost::exception_ptr const& e)
        {
            (*impl_)->set_error(e);      // set the received error
        }

        template <typename T>
        void set_local_data(BOOST_FWD_REF(T) result)
        {
            (*impl_)->set_local_data(boost::forward<Result>(result));
        }

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void, util::unused_type>
    {
    public:
        typedef detail::promise<void, util::unused_type> wrapped_type;
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
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info) << "promise<void>::promise(" << impl_->get_gid() << ")";
        }

    protected:
        template <typename Impl>
        promise(Impl* impl)
          : impl_(impl), future_obtained_(false)
        {}

    public:
        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            (*impl_)->reset();
            future_obtained_ = false;
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
            return (*impl_)->is_ready();
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
//         util::unused_type get(int slot, error_code& ec = throws) const
//         {
//             detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
//             return (*impl_)->get_data(slot, ec);
//         }
//
//         util::unused_type get(error_code& ec = throws) const
//         {
//             detail::log_on_exit<wrapping_type> on_exit(impl_, ec);
//             return (*impl_)->get_data(0, ec);
//         }

        lcos::future<void> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<void>::get_future",
                    "future already has been retrieved from this packaged_task");
                return lcos::future<void>();
            }

            future_obtained_ = true;
            return lcos::future<void>(impl_->get());
        }

        void set()
        {
            return (*impl_)->set_data(util::unused);
        }

        void invalidate(boost::exception_ptr const& e)
        {
            (*impl_)->set_error(e); // set the received error
        }

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Result, typename RemoteResult>
    struct component_type_database<lcos::detail::promise<Result, RemoteResult> >
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
