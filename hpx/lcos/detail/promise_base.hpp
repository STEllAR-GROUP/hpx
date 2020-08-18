//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2016-2017 Thomas Heller
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/lcos/detail/promise_lco.hpp>
#include <hpx/lcos_local/promise.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace hpx {
namespace lcos {
    namespace detail {

        template <typename Result>
        struct promise_data : task_base<Result>
        {
            typedef typename task_base<Result>::init_no_addref init_no_addref;

            promise_data() = default;

            explicit promise_data(init_no_addref no_addref)
              : task_base<Result>(no_addref)
            {}

            void set_task(util::unique_function_nonser<void()>&& f)
            {
                f_ = std::move(f);
            }

            void mark_as_started()
            {
                this->task_base<Result>::started_test_and_set();
            }

        private:
            void do_run()
            {
                if (!f_)
                    return;    // do nothing if no deferred task is given

                std::exception_ptr p;

                try
                {
                    f_();            // trigger action
                    this->wait();    // wait for value to come back
                    return;
                }
                catch (...)
                {
                    p = std::current_exception();
                }

                // The exception is set outside the catch block since
                // set_exception may yield. Ending the catch block on a
                // different worker thread than where it was started may lead
                // to segfaults.
                this->set_exception(std::move(p));
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

            explicit promise_data_allocator(other_allocator const& alloc)
              : alloc_(alloc)
            {}

            promise_data_allocator(init_no_addref no_addref, in_place,
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
                SharedState const& /*shared_state*/, id_type const& /*id*/,
                bool& /*id_retrieved*/)
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
            typedef hpx::intrusive_ptr<shared_state_type> shared_state_ptr;

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
                : base_type(std::move(static_cast<base_type&&>(other))),
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
                base_type::operator=(std::move(static_cast<base_type&&>(other)));
                id_retrieved_ = other.id_retrieved_;
                id_ = std::move(other.id_);
                addr_ = std::move(other.addr_);

                other.id_retrieved_ = false;
                other.id_ = naming::invalid_id;
                other.addr_ = naming::address();
                return *this;
            }

            naming::id_type get_id(bool mark_as_started = true,
                error_code& ec = throws) const
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

                if (mark_as_started)
                {
                    this->shared_state_->mark_as_started();
                }

                id_retrieved_ = true;
                return id_;
            }

            naming::id_type get_id(error_code& ec) const
            {
                return get_id(true, ec);
            }

            naming::id_type get_unmanaged_gid(error_code& ec = throws) const
            {
                return get_id(false, ec);
            }

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
        private:
            static void wrapping_deleter(wrapping_type *ptr)
            {
                ptr->~wrapping_type();
                hpx::components::component_heap<wrapping_type>().free(ptr);
            }

            // This helper is used to keep the component alive until the
            // completion handler has been called. We need to manually free
            // the component here, since we don't rely on reference counting
            // anymore
            struct keep_alive
            {
                typedef std::unique_ptr<wrapping_type, void(*)(wrapping_type*)>
                    wrapping_ptr;

                wrapping_ptr ptr_;

                explicit keep_alive(wrapping_ptr& ptr)
                  : ptr_(ptr.release(), &wrapping_deleter)
                {}

                keep_alive(keep_alive&& o)
                  : ptr_(o.ptr_.release(), &wrapping_deleter)
                {}

                keep_alive& operator=(keep_alive&& o) = default;

                void operator()()
                {
                    delete ptr_->get(); // delete wrapped_type
                }
            };

        protected:
            void init_shared_state()
            {
                // The lifetime of the LCO (component) part is completely
                // handled by the shared state, we create the object to get our
                // gid and then attach it to the completion handler of the
                // shared state.
                typedef typename keep_alive::wrapping_ptr wrapping_ptr;
                auto ptr = hpx::components::component_heap<wrapping_type>().alloc();
                wrapping_ptr lco_ptr(new (ptr) wrapping_type(
                    new wrapped_type(this->shared_state_)), &wrapping_deleter);

                id_ = lco_ptr->get_unmanaged_id();
                addr_ = naming::address(hpx::get_locality(),
                    components::get_component_type<wrapped_type>(),
                    lco_ptr.get());

                // Pass id to shared state if it exposes the set_id() function
                detail::call_set_id(this->shared_state_, id_, id_retrieved_);

                this->shared_state_->set_on_completed(keep_alive(lco_ptr));
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

