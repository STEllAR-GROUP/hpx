//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2023 Hartmut Kaiser
//  Copyright (c) 2010 Artyom Beilis (Tonkikh)
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    HPX_CXX_EXPORT HPX_CXX_EXTERN class backtrace
    {
    public:
        HPX_CORE_EXPORT explicit backtrace(
            std::size_t frames_no = HPX_HAVE_THREAD_BACKTRACE_DEPTH);

        backtrace(backtrace const&) = default;
        backtrace(backtrace&&) = default;
        backtrace& operator=(backtrace const&) = default;
        backtrace& operator=(backtrace&&) = default;

        virtual ~backtrace() = default;

        [[nodiscard]] std::size_t stack_size() const noexcept
        {
            return frames_.size();
        }

        [[nodiscard]] void* return_address(std::size_t frame_no) const noexcept
        {
            if (frame_no < stack_size())
                return frames_[frame_no];
            return nullptr;
        }

        HPX_CORE_EXPORT void trace_line(
            std::size_t frame_no, std::ostream& out) const;

        [[nodiscard]] HPX_CORE_EXPORT std::string trace_line(
            std::size_t frame_no) const;

        [[nodiscard]] HPX_CORE_EXPORT std::string trace() const;

        HPX_CORE_EXPORT void trace(std::ostream& out) const;

    private:
        std::vector<void*> frames_;
    };

    namespace detail {

        HPX_CXX_EXPORT HPX_CXX_EXTERN class trace_manip
        {
        public:
            explicit constexpr trace_manip(backtrace const* tr) noexcept
              : tr_(tr)
            {
            }
            HPX_CORE_EXPORT std::ostream& write(std::ostream& out) const;

        private:
            backtrace const* tr_;
        };

        HPX_CORE_MODULE_EXPORT std::ostream& operator<<(
            std::ostream& out, trace_manip const& t);
    }    // namespace detail

    HPX_CXX_EXPORT template <typename E>
    [[nodiscard]] detail::trace_manip trace(E const& e)
    {
        auto const* tr = dynamic_cast<backtrace const*>(&e);
        return detail::trace_manip(tr);
    }

    HPX_CORE_MODULE_EXPORT_NODISCARD std::string trace(
        std::size_t frames_no = HPX_HAVE_THREAD_BACKTRACE_DEPTH);
}    // namespace hpx::util
