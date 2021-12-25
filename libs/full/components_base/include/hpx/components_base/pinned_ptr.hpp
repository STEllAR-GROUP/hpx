//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/get_lva.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/components_base/traits/component_pin_support.hpp>
#include <hpx/modules/naming_base.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace components {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        class pinned_ptr_base
        {
        public:
            pinned_ptr_base(pinned_ptr_base const&) = delete;
            pinned_ptr_base& operator=(pinned_ptr_base const&) = delete;

        public:
            constexpr pinned_ptr_base() noexcept = default;

            constexpr explicit pinned_ptr_base(
                naming::address_type lva) noexcept
              : lva_(lva)
            {
            }

            virtual ~pinned_ptr_base() = default;

        protected:
            naming::address_type lva_ = nullptr;
        };

        template <typename Component>
        class pinned_ptr : public pinned_ptr_base
        {
        public:
            explicit pinned_ptr(naming::address_type lva) noexcept
              : pinned_ptr_base(lva)
            {
                HPX_ASSERT(nullptr != this->lva_);

                // pin associated component instance
                traits::component_pin_support<Component>::pin(
                    get_lva<Component>::call(this->lva_));
            }

            ~pinned_ptr() override
            {
                // unpin associated component instance
                traits::component_pin_support<Component>::unpin(
                    get_lva<Component>::call(this->lva_));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    class pinned_ptr
    {
    private:
        template <typename T>
        struct id
        {
        };

        template <typename Component>
        pinned_ptr(naming::address_type lva, id<Component>)
          : data_(new detail::pinned_ptr<Component>(lva))
        {
        }

    public:
        constexpr pinned_ptr() = default;

        ~pinned_ptr()
        {
            delete data_;
        }

        pinned_ptr(pinned_ptr const&) = delete;
        pinned_ptr(pinned_ptr&& rhs) noexcept
          : data_(rhs.data_)
        {
            rhs.data_ = nullptr;
        }

        pinned_ptr& operator=(pinned_ptr const&) = delete;
        pinned_ptr& operator=(pinned_ptr&& rhs) noexcept
        {
            if (this != &rhs)
            {
                data_ = rhs.data_;
                rhs.data_ = nullptr;
            }
            return *this;
        }

        template <typename Component>
        static pinned_ptr create(naming::address_type lva)
        {
            using component_type = std::remove_cv_t<Component>;
            if constexpr (traits::component_decorates_action_v<component_type>)
            {
                // created pinned_ptr actually pins object it refers to
                return pinned_ptr(lva, id<component_type>{});
            }
            else
            {
                // created pinned_ptr does not pin object it refers to
                (void) lva;
                return pinned_ptr{};
            }
        }

    private:
        detail::pinned_ptr_base* data_ = nullptr;
    };
}}    // namespace hpx::components
