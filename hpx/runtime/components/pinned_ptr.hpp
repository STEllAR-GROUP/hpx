//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PINNED_PTR_JAN_22_2016_1027AM)
#define HPX_COMPONENTS_PINNED_PTR_JAN_22_2016_1027AM

#include <hpx/config.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/util/assert.hpp>

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

            explicit pinned_ptr_base(naming::address::address_type lva) noexcept
              : lva_(lva)
            {}

            virtual ~pinned_ptr_base() {}

        protected:
            naming::address::address_type lva_;
        };

        template <typename Component>
        class pinned_ptr : public pinned_ptr_base
        {
        public:
            HPX_NON_COPYABLE(pinned_ptr);

        public:
            pinned_ptr() noexcept {}

            explicit pinned_ptr(naming::address::address_type lva) noexcept
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
                    get_lva<Component>::call(this->lva_)->pin();
            }

            void unpin()
            {
                if (0 != this->lva_)
                    get_lva<Component>::call(this->lva_)->unpin();
                this->lva_ = 0;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    class pinned_ptr
    {
    private:
        template <typename T> struct id {};

    public:
        pinned_ptr() {}

        pinned_ptr(pinned_ptr && rhs)
          : data_(std::move(rhs.data_))
        {}

        pinned_ptr& operator= (pinned_ptr && rhs)
        {
            data_ = std::move(rhs.data_);
            return *this;
        }

        template <typename Component>
        static pinned_ptr create(naming::address::address_type lva)
        {
            return pinned_ptr(lva, id<Component>());
        }

    private:
        template <typename Component>
        pinned_ptr(naming::address::address_type lva, id<Component>)
          : data_(new detail::pinned_ptr<Component>(lva))
        {}

        std::unique_ptr<detail::pinned_ptr_base> data_;
    };
}}

#endif

