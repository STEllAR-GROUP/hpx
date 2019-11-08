//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PINNED_PTR_JAN_22_2016_1027AM)
#define HPX_COMPONENTS_PINNED_PTR_JAN_22_2016_1027AM

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/naming_base.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/component_pin_support.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace components
{
    namespace detail
    {
        class pinned_ptr_base
        {
        public:
            HPX_NON_COPYABLE(pinned_ptr_base);

        public:
            pinned_ptr_base() noexcept
              : lva_(0)
            {}

            explicit pinned_ptr_base(naming::address_type lva) noexcept
              : lva_(lva)
            {}

            virtual ~pinned_ptr_base() {}

        protected:
            naming::address_type lva_;
        };

        template <typename Component>
        class pinned_ptr : public pinned_ptr_base
        {
        public:
            HPX_NON_COPYABLE(pinned_ptr);

        public:
            pinned_ptr() noexcept {}

            explicit pinned_ptr(naming::address_type lva) noexcept
              : pinned_ptr_base(lva)
            {
                HPX_ASSERT(0 != this->lva_);
                pin();
            }

            ~pinned_ptr()
            {
                unpin();
            }

        protected:
            void pin()
            {
                if (0 != this->lva_)
                {
                    traits::component_pin_support<Component>::pin(
                        get_lva<Component>::call(this->lva_));
                }
            }

            void unpin()
            {
                if (0 != this->lva_)
                {
                    traits::component_pin_support<Component>::unpin(
                        get_lva<Component>::call(this->lva_));
                }
                this->lva_ = 0;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    class pinned_ptr
    {
    private:
        template <typename T> struct id {};

        // created pinned_ptr does not pin object it refers to
        template <typename Component, typename Enable = void>
        struct create_helper
        {
            static pinned_ptr call(naming::address_type)
            {
                return pinned_ptr{};
            }
        };

        // created pinned_ptr actually pins object it refers to
        template <typename Component>
        struct create_helper<Component,
            typename std::enable_if<
                traits::component_decorates_action<Component>::value
            >::type>
        {
            static pinned_ptr call(naming::address_type lva)
            {
                return pinned_ptr(lva, id<Component>{});
            }
        };

        template <typename Component>
        pinned_ptr(naming::address_type lva, id<Component>)
          : data_(new detail::pinned_ptr<Component>(lva))
        {
        }

    public:
        pinned_ptr() = default;

        pinned_ptr(pinned_ptr const& rhs) = delete;
        pinned_ptr(pinned_ptr && rhs) = default;

        pinned_ptr& operator= (pinned_ptr const& rhs) = delete;
        pinned_ptr& operator= (pinned_ptr && rhs) = default;

        template <typename Component>
        static pinned_ptr create(naming::address_type lva)
        {
            using component_type = typename std::remove_cv<Component>::type;
            return create_helper<component_type>::call(lva);
        }

    private:
        std::unique_ptr<detail::pinned_ptr_base> data_;
    };
}}

#endif

