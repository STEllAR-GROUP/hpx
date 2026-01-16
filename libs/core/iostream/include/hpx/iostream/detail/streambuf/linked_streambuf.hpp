//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>    // member template friends.

#include <cstdint>
#include <iosfwd>
#include <streambuf>
#include <string>
#include <typeinfo>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostream::detail {

    HPX_CXX_CORE_EXPORT template <typename Self, typename Ch, typename Tr,
        typename Alloc, typename Mode>
    class chain_base;

    HPX_CXX_CORE_EXPORT template <typename Chain, typename Access,
        typename Mode>
    class chainbuf;

    HPX_CXX_CORE_EXPORT template <typename Ch,
        typename Tr = std::char_traits<Ch>>
    class linked_streambuf : public std::basic_streambuf<Ch, Tr>
    {
        using base_type = std::basic_streambuf<Ch, Tr>;

    protected:
        linked_streambuf() = default;

        constexpr void set_true_eof(bool eof) noexcept
        {
            flags_ = (flags_ & ~flags::true_eof) |
                (eof ? static_cast<int>(flags::true_eof) : 0);
        }

    public:
        // Should be called only after receiving an ordinary EOF indication, to
        // confirm that it represents EOF rather than WOULD_BLOCK.
        [[nodiscard]] constexpr bool true_eof() const noexcept
        {
            return (flags_ & flags::true_eof) != 0;
        }

    protected:
        //----------grant friendship to chain_base and chainbuf-------------------//
        template <typename Self, typename ChT, typename TrT, typename Alloc,
            typename Mode>
        friend class chain_base;
        template <typename Chain, typename Mode, typename Access>
        friend class chainbuf;
        template <typename U>
        friend class member_close_operation;

        using base_type::eback;
        using base_type::egptr;
        using base_type::epptr;
        using base_type::gbump;
        using base_type::gptr;
        using base_type::overflow;
        using base_type::pbackfail;
        using base_type::pbase;
        using base_type::pbump;
        using base_type::pptr;
        using base_type::seekoff;
        using base_type::seekpos;
        using base_type::setg;
        using base_type::setp;
        using base_type::sync;
        using base_type::underflow;
        using base_type::xsgetn;
        using base_type::xsputn;

        void close(std::ios_base::openmode const which)
        {
            if (which == std::ios_base::in &&
                (flags_ & flags::input_closed) == 0)
            {
                flags_ = flags_ | flags::input_closed;
                close_impl(which);
            }
            if (which == std::ios_base::out &&
                (flags_ & flags::output_closed) == 0)
            {
                flags_ = flags_ | flags::output_closed;
                close_impl(which);
            }
        }

        constexpr void set_needs_close() noexcept
        {
            flags_ = flags_ & ~(flags::input_closed | flags::output_closed);
        }

        virtual void set_next(linked_streambuf<Ch, Tr>*) {}
        virtual void close_impl(std::ios_base::openmode) = 0;
        [[nodiscard]] virtual bool auto_close() const = 0;
        virtual void set_auto_close(bool) = 0;
        virtual bool strict_sync() = 0;
        [[nodiscard]] virtual std::type_info const& component_type() const = 0;
        virtual void* component_impl() = 0;

    private:
        enum class flags : std::uint8_t
        {
            true_eof = 1,
            input_closed = true_eof << 1,
            output_closed = input_closed << 1
        };

        friend constexpr int operator&(int const lhs, flags rhs) noexcept
        {
            return lhs & static_cast<int>(rhs);
        }
        friend constexpr int operator|(flags lhs, flags rhs) noexcept
        {
            return static_cast<int>(lhs) | static_cast<int>(rhs);
        }
        friend constexpr int operator|(int const lhs, flags rhs) noexcept
        {
            return lhs | static_cast<int>(rhs);
        }
        friend constexpr int operator~(flags rhs) noexcept
        {
            return ~static_cast<int>(rhs);
        }

        int flags_ = 0;
    };
}    // namespace hpx::iostream::detail

#include <hpx/config/warnings_prefix.hpp>
