//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/type_support/base_from_member.hpp>

#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <type_traits>

namespace hpx::iostreams {

    struct flush_type
    {
    };

    struct endl_type
    {
    };

    struct async_flush_type
    {
    };

    struct async_endl_type
    {
    };

    inline std::ostream& operator<<(std::ostream& os, flush_type const&)
    {
        return os << std::flush;
    }

    inline std::ostream& operator<<(std::ostream& os, endl_type const&)
    {
        return os << std::endl;
    }

    inline std::ostream& operator<<(std::ostream& os, async_flush_type const&)
    {
        return os << std::flush;
    }

    inline std::ostream& operator<<(std::ostream& os, async_endl_type const&)
    {
        return os << std::endl;
    }

    namespace detail {

        class local_streambuf : public std::streambuf
        {
        public:
            explicit local_streambuf(std::ostream& target)
              : target_(std::addressof(target))
            {
                setp(buffer_, buffer_ + buffer_size_);
            }

            local_streambuf(local_streambuf const&) = delete;
            local_streambuf(local_streambuf&&) = delete;
            local_streambuf& operator=(local_streambuf const&) = delete;
            local_streambuf& operator=(local_streambuf&&) = delete;

        protected:
            int sync() override
            {
                std::lock_guard<hpx::spinlock> l(mtx_);
                flush_buffer_locked();
                target_->flush();
                return target_->good() ? 0 : -1;
            }

            int_type overflow(int_type ch) override
            {
                std::lock_guard<hpx::spinlock> l(mtx_);
                flush_buffer_locked();

                if (!traits_type::eq_int_type(ch, traits_type::eof()))
                {
                    char const c = traits_type::to_char_type(ch);
                    target_->write(&c, 1);
                }

                return target_->good() ? traits_type::not_eof(ch) :
                                         traits_type::eof();
            }

            std::streamsize xsputn(
                char const* s, std::streamsize count) override
            {
                std::lock_guard<hpx::spinlock> l(mtx_);
                flush_buffer_locked();
                target_->write(s, count);
                return target_->good() ? count : 0;
            }

        private:
            void flush_buffer_locked()
            {
                std::streamsize const count = pptr() - pbase();
                if (count > 0)
                {
                    target_->write(pbase(), count);
                    setp(buffer_, buffer_ + buffer_size_);
                }
            }

            static constexpr std::size_t buffer_size_ = 1024;

            hpx::spinlock mtx_{"iostreams::detail::local_streambuf"};
            std::ostream* target_;
            char buffer_[buffer_size_];
        };

        inline std::ostream& get_coutstream() noexcept
        {
            return std::cout;
        }

        inline std::ostream& get_cerrstream() noexcept
        {
            return std::cerr;
        }

        inline std::stringstream& get_consolestream() noexcept
        {
            static std::stringstream console_stream;
            return console_stream;
        }
    }    // namespace detail

    template <typename Char = char, typename Sink = void>
    struct ostream
      : private hpx::util::base_from_member<detail::local_streambuf>
      , std::ostream
    {
    private:
        using streambuf_base =
            hpx::util::base_from_member<detail::local_streambuf>;

    public:
        explicit ostream(std::ostream& target)
          : streambuf_base(target)
          , std::ostream(&this->member)
        {
            // Keep the distributed template surface for source compatibility,
            // but make the local fallback's supported instantiation explicit.
            static_assert(std::is_same_v<Char, char>,
                "The local iostream fallback only supports Char = char.");
            static_assert(std::is_same_v<Sink, void>,
                "The local iostream fallback only supports the default Sink.");
        }

        ostream(ostream const&) = delete;
        ostream(ostream&&) = delete;
        ostream& operator=(ostream const&) = delete;
        ostream& operator=(ostream&&) = delete;
    };
}    // namespace hpx::iostreams

namespace hpx {

    // The local fallback is header-only, so these manipulators do not need the
    // out-of-line object definitions used by the distributed component.
    inline constexpr iostreams::flush_type flush{};
    inline constexpr iostreams::endl_type endl{};
    inline constexpr iostreams::async_flush_type async_flush{};
    inline constexpr iostreams::async_endl_type async_endl{};

    inline iostreams::ostream<> cout(iostreams::detail::get_coutstream());
    inline iostreams::ostream<> cerr(iostreams::detail::get_cerrstream());
    inline iostreams::ostream<> consolestream(
        iostreams::detail::get_consolestream());

    // This exposes the shared backing stringstream directly. Read it only
    // after local output has stopped, e.g. after the runtime has finished or
    // after all threads writing to consolestream have been joined.
    inline std::stringstream const& get_consolestream()
    {
        return iostreams::detail::get_consolestream();
    }
}    // namespace hpx
