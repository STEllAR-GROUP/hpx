//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2016      Thomas Heller
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/component_type_database.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace lcos {
    namespace detail {
        template <typename Result, typename RemoteResult>
        class promise_lco;
    }
}
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace traits {
    template <typename Result, typename RemoteResult>
    struct managed_component_dtor_policy<
        lcos::detail::promise_lco<Result, RemoteResult>>
    {
        typedef managed_object_is_lifetime_controlled type;
    };
}
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace lcos {
    namespace detail {
        template <typename Result, typename RemoteResult>
        class promise_lco_base
            : public lcos::base_lco_with_value<Result, RemoteResult>
        {
        protected:
            typedef lcos::detail::future_data<Result>       shared_state_type;
            typedef hpx::intrusive_ptr<shared_state_type> shared_state_ptr;

            typedef lcos::base_lco_with_value<Result, RemoteResult> base_type;

            typedef typename base_type::result_type result_type;

        public:
            explicit promise_lco_base(shared_state_ptr const& shared_state)
              : shared_state_(shared_state)
            {
            }

            void set_value(RemoteResult&& result)
            {
                HPX_ASSERT(shared_state_);
                shared_state_->set_remote_data(std::move(result));
            }

            void set_exception(std::exception_ptr const& e)
            {
                HPX_ASSERT(shared_state_);
                shared_state_->set_exception(e);
            }

            // This is the component id. Every component needs to have an
            // embedded enumerator 'value' which is used by the generic action
            // implementation to associate this component with a given action.
            enum
            {
                value = components::component_promise
            };

        protected:
            shared_state_ptr shared_state_;

        private:
            // intrusive reference counting, noop since we don't require
            // reference counting here.
            friend void intrusive_ptr_add_ref(promise_lco_base* /*p*/)
            {
            }

            // intrusive reference counting, noop since we don't require
            // reference counting here.
            friend void intrusive_ptr_release(promise_lco_base* /*p*/)
            {
            }
        };

        template <typename Result, typename RemoteResult>
        class promise_lco : public promise_lco_base<Result, RemoteResult>
        {
        protected:
            typedef lcos::detail::future_data<Result>       shared_state_type;
            typedef hpx::intrusive_ptr<shared_state_type> shared_state_ptr;

            typedef promise_lco_base<Result, RemoteResult> base_type;

            typedef typename base_type::result_type result_type;

        public:
            explicit promise_lco(shared_state_ptr const& shared_state)
              : base_type(shared_state)
            {
            }

            result_type get_value()
            {
                result_type* result = this->shared_state_->get_result();
                return std::move(*result);
            }
            result_type get_value(error_code& ec)
            {
                result_type* result = this->shared_state_->get_result(ec);
                return std::move(*result);
            }

        private:
            template <typename>
            friend struct components::detail_adl_barrier::init;

            void set_back_ptr(components::managed_component<promise_lco>* bp)
            {
                HPX_ASSERT(bp);
                HPX_UNUSED(bp);
            }
        };

        template <>
        class promise_lco<void, hpx::util::unused_type>
            : public promise_lco_base<void, hpx::util::unused_type>
        {
        protected:
            using shared_state_type = lcos::detail::future_data<void>;
            using shared_state_ptr = hpx::intrusive_ptr<shared_state_type>;
            using base_type = promise_lco_base<void, hpx::util::unused_type>;

        public:
            explicit promise_lco(shared_state_ptr const& shared_state)
              : base_type(shared_state)
            {
            }

            hpx::util::unused_type get_value()
            {
                this->shared_state_->get_result();
                return hpx::util::unused;
            }
            hpx::util::unused_type get_value(error_code& ec)
            {
                this->shared_state_->get_result(ec);
                return hpx::util::unused;
            }

        private:
            template <typename>
            friend struct components::detail_adl_barrier::init;

            void set_back_ptr(components::managed_component<promise_lco>* bp)
            {
                HPX_ASSERT(bp);
                HPX_UNUSED(bp);
            }
        };
    }
}
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace traits {
    namespace detail {
        HPX_EXPORT extern util::atomic_count unique_type;
    }

    template <typename Result, typename RemoteResult>
    struct component_type_database<
        lcos::detail::promise_lco<Result, RemoteResult> >
    {
        static components::component_type value;

        static components::component_type get()
        {
            // Promises are never created remotely, their factories are not
            // registered with AGAS, so we can assign the component types
            // locally.
            if (value == components::component_invalid)
            {
                value = derived_component_type(++detail::unique_type,
                    components::component_base_lco_with_value);
            }
            return value;
        }

        static void set(components::component_type /* t */)
        {
            HPX_ASSERT(false);
        }
    };

    template <typename Result, typename RemoteResult>
    components::component_type component_type_database<
        lcos::detail::promise_lco<Result, RemoteResult>>::value =
        components::component_invalid;
}
namespace components { namespace detail {
    // Forward declare promise_lco<void> to avoid duplicate instantiations
    template <> HPX_ALWAYS_EXPORT
    hpx::components::managed_component<
        lcos::detail::promise_lco<void, hpx::util::unused_type>>::heap_type&
    component_heap_helper<hpx::components::managed_component<
        lcos::detail::promise_lco<void, hpx::util::unused_type>>>(...);

    template <typename Result, typename RemoteResult>
    struct component_heap_impl<hpx::components::managed_component<
        lcos::detail::promise_lco<Result, RemoteResult>>>
    {
        typedef void valid;

        HPX_ALWAYS_EXPORT static
        typename hpx::components::managed_component<
            lcos::detail::promise_lco<Result, RemoteResult>>::heap_type& call()
        {
            util::reinitializable_static<typename hpx::components::managed_component<
                lcos::detail::promise_lco<Result, RemoteResult>>::heap_type> heap;
            return heap.get();
        }
    };
}}
}

