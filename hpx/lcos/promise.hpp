//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PROMISE_FULL_EMPTY_FEB_03_2009_0841AM)
#define HPX_LCOS_PROMISE_FULL_EMPTY_FEB_03_2009_0841AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/once.hpp>
#include <hpx/util/one_size_heap_list_base.hpp>
#include <hpx/util/static_reinit.hpp>

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

namespace hpx { namespace components
{
    namespace detail_adl_barrier
    {
        template <typename BackPtrTag>
        struct init;
    }

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Component, typename Derived>
        struct heap_factory;

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT boost::shared_ptr<util::one_size_heap_list_base> get_promise_heap(
            components::component_type type);

        ///////////////////////////////////////////////////////////////////////
        template <typename Promise>
        struct promise_heap_factory
        {
            typedef Promise component_type;
            typedef managed_component<Promise> derived_type;
            typedef derived_type value_type;

            struct wrapper_heap_tag {};

        private:
            static boost::shared_ptr<util::one_size_heap_list_base> heap_;
            static lcos::local::once_flag constructed_;

            static void destruct_heap()
            {
                heap_.reset();
            }

            // this will be called exactly once per runtime initialization
            static void construct_heap()
            {
                heap_ = get_promise_heap(get_component_type<component_type>());
            }

            static void create_heap()
            {
                heap_ = get_promise_heap(get_component_type<component_type>());
                util::reinit_register(&promise_heap_factory::construct_heap,
                    &promise_heap_factory::destruct_heap);
            }

            static util::one_size_heap_list_base& get_heap()
            {
                // ensure thread-safe initialization
                lcos::local::call_once(constructed_,
                    &promise_heap_factory::create_heap);
                return *heap_;
            }

        public:
            static void* alloc(std::size_t count = 1)
            {
                return get_heap().alloc(count);
            }
            static void free(void* p, std::size_t count = 1)
            {
                get_heap().free(p, count);
            }
            static naming::gid_type get_gid(void* p)
            {
                return get_heap().get_gid(p);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Promise>
        boost::shared_ptr<util::one_size_heap_list_base>
            promise_heap_factory<Promise>::heap_;

        template <typename Promise>
        lcos::local::once_flag
            promise_heap_factory<Promise>::constructed_;

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename RemoteResult>
        struct heap_factory<
                lcos::detail::promise<Result, RemoteResult>,
                managed_component<lcos::detail::promise<Result, RemoteResult> > >
          : promise_heap_factory<lcos::detail::promise<Result, RemoteResult> >
        {};
    }
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
        public lcos::detail::future_data<Result>
    {
    protected:
        typedef lcos::detail::future_data<Result> future_data_type;
        typedef typename future_data_type::result_type result_type;

    public:
        typedef typename future_data_type::completed_callback_type
            completed_callback_type;

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_promise };

        promise()
        {}

        promise(completed_callback_type const& data_sink)
          : future_data_type(data_sink)
        {}

        promise(BOOST_RV_REF(completed_callback_type) data_sink)
          : future_data_type(boost::move(data_sink))
        {}

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~promise()
        {
            this->finalize();
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_local_data(BOOST_FWD_REF(T) result)
        {
            return this->set_data(boost::forward<result_type>(result));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_value (BOOST_RV_REF(RemoteResult) result)
        {
            // set the received result, reset error status
            this->set_data(boost::move(result));
        }

        Result get_value()
        {
            return this->get_data();
        }

        Result move_value()
        {
            return this->move_data();
        }

        void set_exception(boost::exception_ptr const& e)
        {
            return this->future_data_type::set_exception(e);
        }

        // retrieve the gid of this promise
        naming::id_type get_gid() const
        {
            naming::gid_type::mutex_type::scoped_lock l(&gid_);

            // take all credits to avoid a self reference
            naming::gid_type gid = gid_;
            naming::detail::strip_credit_from_gid(
                const_cast<naming::gid_type&>(gid_));

            // we request the id of a future only once
            BOOST_ASSERT(0 != naming::detail::get_credit_from_gid(gid));

            //naming::detail::set_credit_split_mask_for_gid(gid);
            //if (0 == naming::detail::get_credit_from_gid(gid))
            //{
            //    naming::detail::retrieve_new_credits(gid, gid,
            //        HPX_INITIAL_GLOBALCREDIT, 0, l);
            //}

            return naming::id_type(gid, naming::id_type::managed);
        }

        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(gid_ != naming::invalid_gid);
            return gid_;
        }

    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<promise>* bp)
        {
            BOOST_ASSERT(bp);
            BOOST_ASSERT(gid_ == naming::invalid_gid);
            gid_ = bp->get_base_gid();
        }

        naming::gid_type gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void, util::unused_type>
      : public lcos::base_lco_with_value<void, util::unused_type>,
        public lcos::detail::future_data<void>
    {
    protected:
        typedef lcos::detail::future_data<void> future_data_type;
        typedef future_data_type::result_type result_type;

    public:
        typedef future_data_type::completed_callback_type completed_callback_type;

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_promise };

        promise()
        {}

        promise(completed_callback_type const& data_sink)
          : future_data_type(data_sink)
        {}

        promise(BOOST_RV_REF(completed_callback_type) data_sink)
          : future_data_type(boost::move(data_sink))
        {}

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~promise()
        {
            this->finalize();
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
        void set_value (BOOST_RV_REF(util::unused_type) result)
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

        void set_exception(boost::exception_ptr const& e)
        {
            return this->future_data_type::set_exception(e);
        }

        // retrieve the gid of this promise
        naming::id_type get_gid() const
        {
            naming::gid_type::mutex_type::scoped_lock l(&gid_);

            // take all credits to avoid a self reference
            naming::gid_type gid = gid_;
            naming::detail::strip_credit_from_gid(
                const_cast<naming::gid_type&>(gid_));

            // we request the id of a future only once
            BOOST_ASSERT(0 != naming::detail::get_credit_from_gid(gid));

            //naming::detail::set_credit_split_mask_for_gid(gid);
            //if (0 == naming::detail::get_credit_from_gid(gid))
            //{
            //    naming::detail::retrieve_new_credits(gid, gid,
            //        HPX_INITIAL_GLOBALCREDIT, 0, l);
            //}

            return naming::id_type(gid, naming::id_type::managed);
        }

        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(gid_ != naming::invalid_gid);
            return gid_;
        }

    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<promise>* bp)
        {
            BOOST_ASSERT(bp);
            BOOST_ASSERT(gid_ == naming::invalid_gid);
            gid_ = bp->get_base_gid();
        }

        naming::gid_type gid_;
    };
}}}

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    // This is a placeholder shim used for the type erased memory management
    // for all promise types
    struct managed_promise : boost::noncopyable
    {
        /// \brief Construct a managed_component instance holding a new wrapped
        ///        instance
        managed_promise()
          : promise_(0)
        {
            BOOST_ASSERT(false);        // this is never called
        }

        ~managed_promise()
        {
            intrusive_ptr_release(this);
        }

    private:
        friend void intrusive_ptr_add_ref(managed_promise* p)
        {
            intrusive_ptr_add_ref(p->promise_);
        }
        friend void intrusive_ptr_release(managed_promise* p)
        {
            intrusive_ptr_release(p->promise_);
        }

        lcos::detail::promise<void, util::unused_type>* promise_;
    };
}}

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
    ///     apply<some_action>(new continuation(f.get_gid()), ...);
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
    ///                  type this promise is expected to receive
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
        typedef typename wrapped_type::completed_callback_type
            completed_callback_type;

    public:
        /// Construct a new \a promise instance. The supplied
        /// \a thread will be notified as soon as the result of the
        /// operation associated with this future instance has been
        /// returned.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        promise()
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info) << "promise::promise(" << impl_->get_gid() << ")";
        }

        promise(completed_callback_type const& data_sink)
          : impl_(new wrapping_type(new wrapped_type(data_sink))),
            future_obtained_(false)
        {
            LLCO_(info) << "promise::promise(" << impl_->get_gid() << ")";
        }

        promise(BOOST_RV_REF(completed_callback_type) data_sink)
          : impl_(new wrapping_type(new wrapped_type(boost::move(data_sink)))),
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

        virtual ~promise()
        {}

        lcos::future<Result> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<Result>::get_future",
                    "future already has been retrieved from this packaged_action");
                return lcos::future<Result>();
            }

            future_obtained_ = true;
            return lcos::detail::make_future_from_data<Result>(impl_->get());
        }

        ///
        template <typename T>
        void set_value(BOOST_FWD_REF(T) result)
        {
            (*impl_)->set_data(boost::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);      // set the received error
        }

        template <typename T>
        void set_local_data(BOOST_FWD_REF(T) result)
        {
            (*impl_)->set_local_data(boost::forward<Result>(result));
        }

#ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // [N3722, 4.1] asks for this...
        explicit operator lcos::future<Result>()
        {
            return get_future();
        }
#endif

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };

#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
    // [N3722, 4.1] asks for this...
    template <typename Result>
    inline future<Result>::future(promise<Result>& promise)
    {
        promise.get_future().swap(*this);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void, util::unused_type>
    {
    public:
        typedef detail::promise<void, util::unused_type> wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;
        typedef wrapped_type::completed_callback_type completed_callback_type;

        /// Construct a new \a future instance. The supplied
        /// \a thread will be notified as soon as the result of the
        /// operation associated with this future instance has been
        /// returned.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        promise()
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info) << "promise<void>::promise(" << impl_->get_gid() << ")";
        }

        promise(completed_callback_type const& data_sink)
          : impl_(new wrapping_type(new wrapped_type(data_sink))),
            future_obtained_(false)
        {
            LLCO_(info) << "promise::promise(" << impl_->get_gid() << ")";
        }

        promise(BOOST_RV_REF(completed_callback_type) data_sink)
          : impl_(new wrapping_type(new wrapped_type(boost::move(data_sink)))),
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

        typedef util::unused_type result_type;

        ~promise()
        {}

        lcos::future<void> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<void>::get_future",
                    "future already has been retrieved from this packaged_action");
                return lcos::future<void>();
            }

            future_obtained_ = true;
            return lcos::detail::make_future_from_data<void>(impl_->get());
        }

        void set_value()
        {
            (*impl_)->set_data(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);
        }

#ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // [N3722, 4.1] asks for this...
        explicit operator lcos::future<void>()
        {
            return get_future();
        }
#endif

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };

#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
    // [N3722, 4.1] asks for this...
    inline future<void>::future(promise<void>& promise)
    {
        promise.get_future().swap(*this);
    }
#endif
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    namespace detail
    {
        HPX_EXPORT extern boost::detail::atomic_count unique_type;
    }

    template <typename Result, typename RemoteResult>
    struct component_type_database<lcos::detail::promise<Result, RemoteResult> >
    {
        HPX_ALWAYS_EXPORT static components::component_type value;

        static components::component_type get()
        {
            // Promises are never created remotely, their factories are not
            // registered with AGAS, so we can assign the component types locally.
            if (value == components::component_invalid)
            {
                value = derived_component_type(++detail::unique_type,
                    components::component_base_lco_with_value);
            }
            return value;
        }

        static void set(components::component_type t)
        {
            BOOST_ASSERT(false);
        }
    };

    template <typename Result, typename RemoteResult>
    HPX_ALWAYS_EXPORT components::component_type
    component_type_database<
        lcos::detail::promise<Result, RemoteResult>
    >::value = components::component_invalid;
}}

#endif
