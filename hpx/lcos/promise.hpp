//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PROMISE_FEB_03_2009_0841AM)
#define HPX_LCOS_PROMISE_FEB_03_2009_0841AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
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
#include <hpx/lcos/local/spinlock_pool.hpp>
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
            static boost::once_flag constructed_;

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
                // FIXME: The heap may be initialized during startup in a
                //        non-HPX thread
                // Bootstrapping should happen in HPX threads ...
                boost::call_once(constructed_,
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
        boost::once_flag
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
    template <typename Result, typename RemoteResult>
    class promise_base
      : public lcos::base_lco_with_value<Result, RemoteResult>
    {
        template <typename Result_, typename RemoteResult_>
        friend class lcos::promise;

    public:
        virtual void add_ref() = 0;
        virtual void release() = 0;
        virtual long count() const = 0;

        // retrieve the gid of this promise
        naming::id_type get_id() const
        {
            boost::unique_lock<naming::gid_type> l(gid_.get_mutex());
            // The GID needs to be split in order to keep the promise alive.
            return get_gid_locked(std::move(l), true);
        }

        naming::id_type get_unmanaged_id() const
        {
            boost::unique_lock<naming::gid_type> l(gid_.get_mutex());
            return naming::id_type(gid_, id_type::unmanaged);
        }

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        naming::id_type get_gid() const
        {
            return get_id();
        }
#endif

    protected:
        naming::id_type get_gid_locked(
            boost::unique_lock<naming::gid_type> l, bool split) const
        {
            if (split)
            {
                hpx::future<naming::gid_type> gid
                    = naming::detail::split_gid_if_needed_locked(l, gid_);
                l.unlock();
                return naming::id_type(gid.get(), naming::id_type::managed);
            }
            else
            {
                l.unlock();
                return naming::id_type(gid_, naming::id_type::managed);
            }
        }

    protected:
        naming::gid_type get_base_gid() const
        {
            HPX_ASSERT(gid_ != naming::invalid_gid);
            return gid_;
        }

        mutable naming::gid_type gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class promise;

    template <typename Result, typename RemoteResult>
    void intrusive_ptr_release(promise<Result, RemoteResult>* p);

    ///////////////////////////////////////////////////////////////////////////
    /// A promise can be used by a single thread to invoke a (remote)
    /// action and wait for the result.
    template <typename Result, typename RemoteResult>
    class promise
      : public promise_base<Result, RemoteResult>,
        public lcos::detail::future_data<Result>
    {
    protected:
        typedef lcos::detail::future_data<Result> future_data_type;
        typedef typename future_data_type::result_type result_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_promise };

        promise()
        {}

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~promise()
        {
            // This lock is needed to not interfere with the intrusive pointer
            // deletion
            this->finalize();
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_local_data(T && result)
        {
            return this->set_data(std::forward<result_type>(result));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_value (RemoteResult && result)
        {
            // set the received result, reset error status
            this->set_data(std::move(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            return this->future_data_type::set_exception(e);
        }

        Result get_value(error_code& ec = throws)
        {
            typedef typename future_data_type::result_type result_type;
            result_type* result = this->get_result();

            // no error has been reported, return the result
            return std::move(*result);
        }

        void add_ref()
        {
            intrusive_ptr_add_ref(this);
        }

        void release()
        {
            intrusive_ptr_release(this);
        }

        long count() const
        {
            return this->count_;
        }

    private:
        bool requires_delete()
        {
            boost::unique_lock<naming::gid_type> l(this->gid_.get_mutex());
            long counter = --this->count_;

            // if this promise was never asked for its id we need to take
            // special precautions for it to go out of scope
            if (1 == counter && naming::detail::has_credits(this->gid_))
            {
                // this will trigger normal destruction
                // don't split the GID to avoid self references.
                naming::id_type id = this->get_gid_locked(std::move(l), false);
                return false;
            }
            else if (0 == counter)
            {
                return true;
            }
            return false;
        }

        friend void intrusive_ptr_release(promise* p)
        {
            if (p->requires_delete())
            {
                delete p;
            }
        }

        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<promise>* bp)
        {
            HPX_ASSERT(bp);
            HPX_ASSERT(this->gid_ == naming::invalid_gid);
            this->gid_ = bp->get_base_gid();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void, util::unused_type>
      : public promise_base<void, util::unused_type>,
        public lcos::detail::future_data<void>
    {
    protected:
        typedef lcos::detail::future_data<void> future_data_type;
        typedef future_data_type::result_type result_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_promise };

        promise()
        {}

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~promise()
        {
            // This lock is needed to not interfere with the intrusive pointer
            // deletion
            this->finalize();
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_local_data(T && result)
        {
            return set_data(std::forward<result_type>(result));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // trigger the future, set the result
        void set_value (util::unused_type && result)
        {
            // set the received result, reset error status
            set_data(std::move(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            return this->future_data_type::set_exception(e);
        }

        util::unused_type get_value(error_code& /*ec*/ = throws)
        {
            this->get_result();
            return util::unused;
        }

        void add_ref()
        {
            intrusive_ptr_add_ref(this);
        }

        void release()
        {
            intrusive_ptr_release(this);
        }

        long count() const
        {
            return this->count_;
        }

    private:
        bool requires_delete()
        {
            boost::unique_lock<naming::gid_type> l(this->gid_.get_mutex());
            long counter = --this->count_;

            // if this promise was never asked for its id we need to take
            // special precautions for it to go out of scope
            if (1 == counter && naming::detail::has_credits(this->gid_))
            {
                // this will trigger normal destruction
                // don't split the GID to avoid self references.
                naming::id_type id = this->get_gid_locked(std::move(l), false);
                return false;
            }
            else if (0 == counter)
            {
                return true;
            }
            return false;
        }

        friend void intrusive_ptr_release(promise* p)
        {
            if (p->requires_delete())
            {
                delete p;
            }
        }

        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<promise>* bp)
        {
            HPX_ASSERT(bp);
            HPX_ASSERT(gid_ == naming::invalid_gid);
            this->gid_ = bp->get_base_gid();
        }
    };
}}}

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    // This is a placeholder shim used for the type erased memory management
    // for all promise types
    struct managed_promise : boost::noncopyable
    {
    private:
        struct tag {};
        typedef lcos::local::spinlock_pool<tag> mutex_type;

        typedef lcos::detail::promise_base<void, util::unused_type>
            promise_base_type;

    public:
        managed_promise()
          : promise_(0)
        {
            HPX_ASSERT(false);        // this is never called
        }

        ~managed_promise()
        {
            promise_->release();
        }

        long count() const
        {
            return promise_->count();
        }

    private:
        friend void intrusive_ptr_add_ref(managed_promise* p)
        {
            p->promise_->add_ref();
        }
        friend void intrusive_ptr_release(managed_promise* p)
        {
            p->promise_->release();
        }

        promise_base_type* promise_;
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
    ///     apply<some_action>(new continuation(f.get_id()), ...);
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

    public:
        /// Construct a new \a promise instance. The supplied
        /// \a thread will be notified as soon as the result of the
        /// operation associated with this future instance has been
        /// returned.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action.
        ///               The target for either of these actions has to be this
        ///               future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        promise()
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info)
                << "promise::promise("
                << impl_->get_unmanaged_id() << ")";
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
        naming::id_type get_id() const
        {
            return (*impl_)->get_id();
        }

        naming::id_type get_unmanaged_id() const
        {
            return (*impl_)->get_unmanaged_id();
        }

    private:
        // Return the global id of this \a future instance
        naming::gid_type get_base_gid() const
        {
            return (*impl_)->get_base_gid();
        }

    public:
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
                    "future already has been retrieved from this promise");
                return lcos::future<Result>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<future<Result> >::create(impl_->get());
        }

        ///
        template <typename T>
        void set_value(T && result)
        {
            (*impl_)->set_data(std::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);      // set the received error
        }

        template <typename T>
        void set_local_data(T && result)
        {
            (*impl_)->set_local_data(std::forward<Result>(result));
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
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action.
        ///               The target for either of these actions has to be this
        ///               future instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        promise()
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info)
                << "promise<void>::promise("
                << impl_->get_unmanaged_id() << ")";
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
        naming::id_type get_id() const
        {
            return (*impl_)->get_id();
        }

        naming::id_type get_unmanaged_id() const
        {
            return (*impl_)->get_unmanaged_id();
        }

    private:
        /// \brief Return the global id of this \a future instance
        naming::gid_type get_base_gid() const
        {
            return (*impl_)->get_base_gid();
        }

    public:
        /// Return whether or not the data is available for this
        /// \a promise.
        bool is_ready() const
        {
            return (*impl_)->is_ready();
        }

        typedef util::unused_type result_type;

        virtual ~promise()
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

            using traits::future_access;
            return future_access<future<void> >::create(impl_->get());
        }

        void set_value()
        {
            (*impl_)->set_data(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);
        }

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };
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
        static components::component_type value;

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
            HPX_ASSERT(false);
        }
    };

    template <typename Result, typename RemoteResult>
    components::component_type component_type_database<
        lcos::detail::promise<Result, RemoteResult>
    >::value = components::component_invalid;
}}

#endif
