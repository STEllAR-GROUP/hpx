//
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2010 Artyom Beilis (Tonkikh)
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    namespace stack_trace {
        HPX_CORE_EXPORT std::size_t trace(void** addresses, std::size_t size);
        HPX_CORE_EXPORT void write_symbols(
            void* const* addresses, std::size_t size, std::ostream&);
        HPX_CORE_EXPORT std::string get_symbol(void* address);
        HPX_CORE_EXPORT std::string get_symbols(
            void* const* address, std::size_t size);
    }    // namespace stack_trace

    class backtrace
    {
    public:
        explicit backtrace(
            std::size_t frames_no = HPX_HAVE_THREAD_BACKTRACE_DEPTH)
        {
            if (frames_no == 0)
                return;
            frames_no += 2;    // we omit two frames from printing
            frames_.resize(frames_no, nullptr);
            std::size_t size = stack_trace::trace(&frames_.front(), frames_no);
            if (size != 0)
                frames_.resize(size);
        }

        virtual ~backtrace() noexcept {}

        std::size_t stack_size() const
        {
            return frames_.size();
        }

        void* return_address(std::size_t frame_no) const
        {
            if (frame_no < stack_size())
                return frames_[frame_no];
            return nullptr;
        }

        void trace_line(std::size_t frame_no, std::ostream& out) const
        {
            if (frame_no < frames_.size())
                stack_trace::write_symbols(&frames_[frame_no], 1, out);
        }

        std::string trace_line(std::size_t frame_no) const
        {
            if (frame_no < frames_.size())
                return stack_trace::get_symbol(frames_[frame_no]);
            return std::string();
        }

        std::string trace() const
        {
            if (frames_.empty())
                return std::string();
            return stack_trace::get_symbols(&frames_.front(), frames_.size());
        }

        void trace(std::ostream& out) const
        {
            if (frames_.empty())
                return;
            stack_trace::write_symbols(&frames_.front(), frames_.size(), out);
        }

    private:
        std::vector<void*> frames_;
    };

    namespace details {
        class trace_manip
        {
        public:
            trace_manip(backtrace const* tr)
              : tr_(tr)
            {
            }
            std::ostream& write(std::ostream& out) const
            {
                if (tr_)
                    tr_->trace(out);
                return out;
            }

        private:
            backtrace const* tr_;
        };

        inline std::ostream& operator<<(
            std::ostream& out, details::trace_manip const& t)
        {
            return t.write(out);
        }
    }    // namespace details

    template <typename E>
    inline details::trace_manip trace(E const& e)
    {
        backtrace const* tr = dynamic_cast<backtrace const*>(&e);
        return details::trace_manip(tr);
    }

    inline std::string trace(
        std::size_t frames_no = HPX_HAVE_THREAD_BACKTRACE_DEPTH)    //-V659
    {
        return backtrace(frames_no).trace();
    }
}}    // namespace hpx::util
