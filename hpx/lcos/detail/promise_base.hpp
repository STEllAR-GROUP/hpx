//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2016      Thomas Heller
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DETAIL_PROMISE_BASE_HPP
#define HPX_LCOS_DETAIL_PROMISE_BASE_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/detail/promise_lco.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/intrusive_ptr.hpp>

#include <memory>
#include <utility>

namespace hpx {
namespace lcos {
    namespace detail {

        template <typename Result>
        struct promise_data : task_base<Result>
        {
            typedef typename task_base<Result>::init_no_addref init_no_addref;

            promise_data() {}

            promise_data(init_no_addref no_addref)
              : task_base<Result>(no_addref)
            {}

            void set_task(util::unique_function_nonser<void()>&& f)
            {
                f_ = std::move(f);
            }

        private:
            void do_run()
            {
                if (!f_)
                    return;         // do nothing if no deferred task is given

                try
                {
                    f_();           // trigger action
                    this->wait();   // wait for value to come back
                }
                catch (...)
                {
                    this->set_exception(compat::current_exception());
                }
            }

            util::unique_function_nonser<void()> f_;
        };

        template <typename Result, typename Allocator>
        struct promise_data_allocator : promise_data<Result>
        {
            typedef typename promise_data<Result>::init_no_addref init_no_addref;
            typedef typename
                    std::allocator_traits<Allocator>::template
                        rebind_alloc<promise_data_allocator>
                other_allocator;

            promise_data_allocator(other_allocator const& alloc)
              : alloc_(alloc)
            {}

            promise_data_allocator(init_no_addref no_addref,
                    other_allocator const& alloc)
              : promise_data<Result>(no_addref), alloc_(alloc)
            {}

        private:
            void destroy()
            {
                typedef std::allocator_traits<other_allocator> traits;

                other_allocator alloc(alloc_);
                traits::destroy(alloc, this);
                traits::deallocate(alloc, this, 1);
            }

        private:
            other_allocator alloc_;
        };

        ///////////////////////////////////////////////////////////////////////
        struct set_id_helper
        {
            template <typename SharedState>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                SharedState const& shared_state, id_type const& id,
                bool& id_retrieved)
            {
                // by default, do nothing
            }

            template <typename SharedState>
            HPX_FORCEINLINE static auto call(int,
                    SharedState const& shared_state, id_type const& id,
                    bool& id_retrieved)
            ->  decltype(shared_state->set_id(id))
            {
                shared_state->set_id(id);
                id_retrieved = true;
            }
        };

        template <typename SharedState>
        HPX_FORCEINLINE
        void call_set_id(SharedState const& shared_state, id_type const& id,
            bool& id_retrieved)
        {
            set_id_helper::call(0, shared_state, id, id_retrieved);
        }
    }
}

namespace traits {
    namespace detail {
        // specialize for promise_data to extract corresponding, allocator-
        // based shared state type
        template <typename R, typename Allocator>
        struct shared_state_allocator<lcos::detail::promise_data<R>, Allocator>
        {
            typedef lcos::detail::promise_data_allocator<R, Allocator> type;
        };
    }    // namespace detail
}    // namespace traits

namespace lcos {
    namespace detail {
        // Promise base contains the actual implementation for the remotely
        // settable
        // promise. It consists of two parts:
        //  1) The shared state, which is used to signal the future
        //  2) The LCO, which is used to set the promise remotely
        //
        // The LCO is a component, which lifetime is completely controlled by
        // the shared state. That is, in the GID that we send along as the
        // continuation can be unmanaged, AGAS doesn't participate in the
        // promise lifetime management. The LCO is being reset once the shared
        // state either contains a value or the data, which is set through the
        // LCO interface, which can go out of scope safely when the shared
        // state is set.
        template <typename Result, typename RemoteResult, typename SharedState>
        class promise_base
          : public hpx::lcos::local::detail::promise_base<Result, SharedState>
        {
            typedef hpx::lcos::local::detail::promise_base<
                    Result, SharedState
                > base_type;

        protected:
            typedef Result result_type;
            typedef SharedState shared_state_type;
            typedef boost::intrusive_ptr<shared_state_type> shared_state_ptr;

            typedef promise_lco<Result, RemoteResult> wrapped_type;
            typedef components::managed_component<wrapped_type> wrapping_type;

        public:
            promise_base()
              : base_type()
              , id_retrieved_(false)
            {
                init_shared_state();
            }

            template <typename Allocator>
            promise_base(std::allocator_arg_t, Allocator const& a)
              : base_type(std::allocator_arg, a)
              , id_retrieved_(false)
            {
                init_shared_state();
            }

            promise_base(promise_base&& other) noexcept
                : base_type(std::move(other)),
                  id_retrieved_(other.id_retrieved_),
                  id_(std::move(other.id_)),
                  addr_(std::move(other.addr_))
            {
                other.id_retrieved_ = false;
                other.id_ = naming::invalid_id;
                other.addr_ = naming::address();
            }

            ~promise_base()
            {
                check_abandon_shared_state(
                    "lcos::detail::promise_base<R>::~promise_base()");
                this->shared_state_.reset();
            }

            promise_base& operator=(promise_base&& other) noexcept
            {
                base_type::operator=(std::move(other));
                id_retrieved_ = other.id_retrieved_;
                id_ = std::move(other.id_);
                addr_ = std::move(other.addr_);

                other.id_retrieved_ = false;
                other.id_ = naming::invalid_id;
                other.addr_ = naming::address();
                return *this;
            }

            naming::id_type get_id(error_code& ec = throws) const
            {
                if (this->shared_state_ == nullptr)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "detail::promise_base<Result, RemoteResult>::get_id",
                        "this promise has no valid shared state");
                    return naming::invalid_id;
                }
                if (!addr_ || !id_)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "detail::promise_base<Result, RemoteResult>::get_id",
                        "this promise has no valid LCO");
                    return naming::invalid_id;
                }
                if (!this->future_retrieved_)
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "promise<Result>::get_id",
                        "future has not been retrieved from this promise yet");
                    return naming::invalid_id;
                }

                id_retrieved_ = true;

                return id_;
            }

            naming::id_type get_unmanaged_gid(error_code& ec = throws) const
            {
                return get_id(ec);
            }

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
            naming::id_type get_gid(error_code& ec = throws) const
            {
                return get_id(ec);
            }
#endif

            naming::address resolve(error_code& ec = throws) const
            {
                if (!addr_ || !id_)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "detail::promise_base<Result, RemoteResult>::get_id",
                        "this promise has no valid LCO");
                    return naming::address();
                }
                return addr_;
            }

        protected:
            void init_shared_state()
            {
                // The lifetime of the LCO (component) part is completely
                // handled by the shared state, we create the object to get our
                // gid and then attach it to the completion handler of the
                // shared state.
                typedef std::unique_ptr<wrapping_type> wrapping_ptr;
                wrapping_ptr lco_ptr(
                    new wrapping_type(new wrapped_type(this->shared_state_)));

                id_ = lco_ptr->get_unmanaged_id();
                addr_ = naming::address(hpx::get_locality(),
                    lco_ptr->get_component_type(),
                    lco_ptr.get());

                // Pass id to shared state if it exposes the set_id() function
                detail::call_set_id(this->shared_state_, id_, id_retrieved_);

                // This helper is used to keep the component alive until the
                // completion handler has been called. We need to manually free
                // the component here, since we don't rely on reference counting
                // anymore
                auto keep_alive = hpx::util::deferred_call(
                    [](wrapping_ptr ptr)
                    {
                        delete ptr->get();      // delete wrapped_type
                    },
                    std::move(lco_ptr));
                this->shared_state_->set_on_completed(std::move(keep_alive));
            }

            void check_abandon_shared_state(const char* fun)
            {
                if (this->shared_state_ != nullptr &&
                    this->future_retrieved_ &&
                    !(this->shared_state_->is_ready() || id_retrieved_))
                {
                    this->shared_state_->set_error(broken_promise, fun,
                        "abandoning not ready shared state");
                    return;
                }
            }
            mutable bool id_retrieved_;

            naming::id_type id_;
            naming::address addr_;
        };
    }
}
}

#endif
