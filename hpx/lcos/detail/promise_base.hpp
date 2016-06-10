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
            void set_task(util::unique_function_nonser<void()>&& f)
            {
                f_ = std::move(f);
            }

        private:
            void do_run()
            {
                if (!f_)
                    return;    // do nothing if no deferred task is given

                try
                {
                    f_();            // trigger action
                    this->wait();    // wait for value to come back
                }
                catch (...)
                {
                    this->set_exception(boost::current_exception());
                }
            }

            util::unique_function_nonser<void()> f_;
        };

        // Promise base contains the actual implementation for the remotely
        // settable
        // promise. It consists of two parts:
        //  1) The shared state, which is used to signal the future
        //  2) The LCO, which is used to set the promise remotely
        //
        // The LCO is a component, which lifetime is completely controlled by
        // the
        // shared state. That is, in the GID that we send along as the
        // continuation
        // can be unmanaged, AGAS doesn't participate in the promise lifetime
        // management. The LCO is being reset once the shared state either
        // contains
        // a value or the data, which is set through the LCO interface, which
        // can go
        // out of scope safely when the shared state is set.
        template <typename Result, typename RemoteResult>
        class promise_base
            : public hpx::lcos::local::detail::promise_base<Result,
                  promise_data<Result>>
        {
            HPX_MOVABLE_ONLY(promise_base);

            typedef hpx::lcos::local::detail::promise_base<Result,
                promise_data<Result>>
                base_type;

        protected:
            typedef Result                                  result_type;
            typedef lcos::detail::future_data<Result>       shared_state_type;
            typedef boost::intrusive_ptr<shared_state_type> shared_state_ptr;

            typedef promise_lco<Result, RemoteResult> wrapped_type;
            typedef components::managed_component<wrapped_type> wrapping_type;

        public:
            promise_base()
              : base_type()
              , id_retrieved_(false)
            {
                // The lifetime of the LCO (component) part is completely
                // handled by the shared state, we create the object to get our
                // gid and then attach it to the completion handler of the
                // shared
                // state.
                typedef std::unique_ptr<wrapping_type> wrapping_ptr;
                wrapping_ptr                           lco_ptr(
                    new wrapping_type(new wrapped_type(this->shared_state_)));

                id_   = lco_ptr->get_unmanaged_id();
                addr_ = naming::address(hpx::get_locality(),
                    lco_ptr->get_component_type(),
                    lco_ptr.get());

                // This helper is used to keep the component alive until the
                // completion handler has been called.
                auto keep_alive = hpx::util::deferred_call(
                    [](wrapping_ptr ptr) {}, std::move(lco_ptr));
                this->shared_state_->set_on_completed(std::move(keep_alive));
            }

            promise_base(promise_base&& other) HPX_NOEXCEPT
                : base_type(std::move(other)),
                  id_retrieved_(other.id_retrieved_),
                  id_(std::move(other.id_)),
                  addr_(std::move(other.addr_))
            {
                other.id_   = naming::invalid_id;
                other.addr_ = naming::address();
            }

            ~promise_base()
            {
                check_abandon_shared_state(
                    "lcos::detail::promise_base<R>::~promise_base()");
                this->shared_state_.reset();
            }

            promise_base& operator=(promise_base&& other) HPX_NOEXCEPT
            {
                base_type::operator=(std::move(other));
                id_retrieved_      = other.id_retrieved_;
                id_                = std::move(other.id_);
                addr_              = std::move(other.addr_);

                other.id_   = naming::invalid_id;
                other.addr_ = naming::address();
                return *this;
            }

            naming::id_type get_id(error_code& ec = throws) const
            {
                if (this->shared_state_ == 0)
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
            void check_abandon_shared_state(const char* fun)
            {
                if (this->shared_state_ != 0 && this->future_retrieved_ &&
                    !id_retrieved_)
                {
                    this->shared_state_->set_error(broken_promise,
                        fun,
                        "abandoning not ready shared state");
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
