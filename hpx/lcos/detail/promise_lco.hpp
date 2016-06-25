//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2016      Thomas Heller
//  Copyright (c) 2011      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DETAIL_PROMISE_LCO_HPP
#define HPX_LCOS_DETAIL_PROMISE_LCO_HPP

#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/unused.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

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
            typedef boost::intrusive_ptr<shared_state_type> shared_state_ptr;

            typedef lcos::base_lco_with_value<Result, RemoteResult> base_type;

            typedef typename base_type::result_type result_type;

        public:
            promise_lco_base(shared_state_ptr const& shared_state)
              : shared_state_(shared_state)
            {
            }

            void set_value(RemoteResult&& result)
            {
                HPX_ASSERT(shared_state_);
                shared_state_->set_data(std::move(result));
            }

            void set_exception(boost::exception_ptr const& e)
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
            friend void intrusive_ptr_add_ref(promise_lco_base* p)
            {
            }

            // intrusive reference counting, noop since we don't require
            // reference counting here.
            friend void intrusive_ptr_release(promise_lco_base* p)
            {
            }
        };

        template <typename Result, typename RemoteResult>
        class promise_lco : public promise_lco_base<Result, RemoteResult>
        {
        protected:
            typedef lcos::detail::future_data<Result>       shared_state_type;
            typedef boost::intrusive_ptr<shared_state_type> shared_state_ptr;

            typedef promise_lco_base<Result, RemoteResult> base_type;

            typedef typename base_type::result_type result_type;

        public:
            promise_lco(shared_state_ptr const& shared_state)
              : base_type(shared_state)
            {
            }

            result_type get_value(error_code& ec = throws)
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
            }
        };

        template <>
        class promise_lco<void, hpx::util::unused_type>
            : public promise_lco_base<void, hpx::util::unused_type>
        {
        protected:
            typedef lcos::detail::future_data<void>         shared_state_type;
            typedef boost::intrusive_ptr<shared_state_type> shared_state_ptr;
            typedef promise_lco_base<void, hpx::util::unused_type> base_type;

        public:
            promise_lco(shared_state_ptr const& shared_state)
              : base_type(shared_state)
            {
            }

            hpx::util::unused_type get_value(error_code& ec = throws)
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
        lcos::detail::promise_lco<Result, RemoteResult>>
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

        static void set(components::component_type t)
        {
            HPX_ASSERT(false);
        }
    };

    template <typename Result, typename RemoteResult>
    components::component_type component_type_database<
        lcos::detail::promise_lco<Result, RemoteResult>>::value =
        components::component_invalid;
}
}

#endif
