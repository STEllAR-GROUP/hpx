//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/serialization.hpp>

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset {

    //////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT locality
    {
        template <typename Impl>
        class impl;

        class impl_base
        {
        public:
            virtual ~impl_base() = default;

            virtual bool equal(impl_base const& rhs) const = 0;
            virtual bool less_than(impl_base const& rhs) const = 0;
            virtual bool valid() const = 0;
            virtual const char* type() const = 0;
            virtual std::ostream& print(std::ostream& os) const = 0;
            virtual void save(serialization::output_archive& ar) const = 0;
            virtual void load(serialization::input_archive& ar) = 0;
            virtual impl_base* clone() const = 0;
            virtual impl_base* move() = 0;

            template <typename Impl>
            Impl& get()
            {
                HPX_ASSERT(Impl::type() == type());
                return static_cast<impl<Impl>*>(this)->impl_;
            }

            template <typename Impl>
            Impl const& get() const
            {
                HPX_ASSERT(Impl::type() == type());
                return static_cast<const impl<Impl>*>(this)->impl_;
            }
        };

    public:
        locality() = default;

        template <typename Impl,
            typename Enable1 =
                std::enable_if_t<!std::is_same_v<locality, std::decay_t<Impl>>>,
            typename Enable2 = std::enable_if_t<!traits::is_iterator_v<Impl>>>
        explicit locality(Impl&& i)
          : impl_(new impl<std::decay_t<Impl>>(HPX_FORWARD(Impl, i)))
        {
        }

        locality(locality const& other);
        locality(locality&& other) noexcept;

        locality& operator=(locality const& other);
        locality& operator=(locality&& other) noexcept;

        ///////////////////////////////////////////////////////////////////////
        explicit operator bool() const noexcept
        {
            return impl_ ? impl_->valid() : false;
        }

        char const* type() const
        {
            return impl_ ? impl_->type() : "";
        }

        template <typename Impl>
        Impl& get()
        {
            HPX_ASSERT(impl_);
            return impl_->get<Impl>();
        }

        template <typename Impl>
        Impl const& get() const
        {
            HPX_ASSERT(impl_);
            return impl_->get<Impl>();
        }

    private:
        friend HPX_EXPORT bool operator==(
            locality const& lhs, locality const& rhs);
        friend HPX_EXPORT bool operator!=(
            locality const& lhs, locality const& rhs);
        friend HPX_EXPORT bool operator<(
            locality const& lhs, locality const& rhs);
        friend HPX_EXPORT bool operator>(
            locality const& lhs, locality const& rhs);

        friend HPX_EXPORT std::ostream& operator<<(
            std::ostream& os, locality const& l);

        // serialization support
        friend class hpx::serialization::access;

        void save(serialization::output_archive& ar,
            const unsigned int version) const;

        void load(serialization::input_archive& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER();

        std::unique_ptr<impl_base> impl_;

        template <typename Impl>
        class impl : public impl_base
        {
        public:
            explicit impl(Impl&& i)
              : impl_(HPX_MOVE(i))
            {
            }
            explicit impl(Impl const& i)
              : impl_(i)
            {
            }

            bool equal(impl_base const& rhs) const override
            {
                if (type() != rhs.type())
                    return false;
                return impl_ == rhs.get<Impl>();
            }

            bool less_than(impl_base const& rhs) const override
            {
                return type() < rhs.type() ||
                    (type() == rhs.type() && impl_ < rhs.get<Impl>());
            }

            bool valid() const override
            {
                return !!impl_;
            }

            const char* type() const override
            {
                return Impl::type();
            }

            std::ostream& print(std::ostream& os) const override
            {
                os << impl_;
                return os;
            }

            void save(serialization::output_archive& ar) const override
            {
                impl_.save(ar);
            }

            void load(serialization::input_archive& ar) override
            {
                impl_.load(ar);
            }

            impl_base* clone() const override
            {
                return new impl<Impl>(impl_);
            }

            impl_base* move() override
            {
                return new impl<Impl>(HPX_MOVE(impl_));
            }

            Impl impl_;
        };
    };

    using endpoints_type = std::map<std::string, locality>;

    HPX_EXPORT std::ostream& operator<<(
        std::ostream& os, endpoints_type const& endpoints);
}}    // namespace hpx::parcelset

#include <hpx/config/warnings_suffix.hpp>
