// logger.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#pragma once

#include <hpx/config.hpp>
#include <hpx/logging/format/named_write.hpp>
#include <hpx/logging/level.hpp>
#include <hpx/modules/format.hpp>

#include <utility>
#include <vector>

namespace hpx::util::logging {

    /**
    @brief The logger class. Every log from your application is an instance of
    this (see @ref workflow_processing "workflow")

    As described in @ref workflow_processing "workflow",
    processing the message is composed of 2 things:
    - @ref workflow_2a "Gathering the message"
    - @ref workflow_2b "Processing the message"

    The logger class has 2 template parameters:


    @param write_msg This is the object that does
    the @ref workflow_2b "second step" - the writing of the message.
    It can be a simple functor.
    Or, it can be a more complex object that contains
    logic of how the message is to be further formatted,
    and written to multiple destinations.
    You can implement your own @c write_msg class,
    or it can be any of the classes defined in writer namespace.
    Check out writer::format_write - which allows you to use
    several formatters to further format the message, and then write it to destinations.

    \n\n
    You will seldom need to use the logger class directly.
    You can use @ref defining_your_logger "other wrapper classes".


    \n\n
    Once all message is gathered, it's passed on to the writer.
    This is usually done through a @ref macros_use "macro".

    @code
    HPX_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts )
    HPX_DECLARE_LOG(g_l, logger)

    #define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() )

    // usage
    L_ << "this is so cool " << i++;

    @endcode



    \n\n
    To understand more on the workflow that involves %logging:
    - check out the gather namespace
    - check out the writer namespace

    */
    class logger
    {
        HPX_NON_COPYABLE(logger);

        struct gather_holder : message
        {
            HPX_NON_COPYABLE(gather_holder);

            explicit gather_holder(logger& p_this)
              : m_this(p_this)
            {
            }

            ~gather_holder()
            {
                if (!empty())
                {
                    m_this.write(HPX_MOVE(static_cast<message&&>(*this)));
                }
            }

        private:
            logger& m_this;
        };

    public:
        logger() noexcept
          : m_level(level::enable_all)
        {
        }
        explicit logger(level default_level) noexcept
          : m_level(default_level)
        {
        }

        ~logger()
        {
            // force writing all messages from cache,
            // if cache hasn't been turned off yet
            turn_cache_off();
        }

        /**
            reads all data about a log message (gathers all the data about it)
        */
        gather_holder gather()
        {
            return gather_holder{*this};
        }

        [[nodiscard]] writer::named_write& writer() noexcept
        {
            return m_writer;
        }
        [[nodiscard]] writer::named_write const& writer() const noexcept
        {
            return m_writer;
        }

        [[nodiscard]] bool is_enabled(level level) const noexcept
        {
            return level >= m_level;
        }

        void set_enabled(level level) noexcept
        {
            m_level = level;
        }

        /** @brief Marks this logger as initialized

        You might log messages before the logger is initialized.
        In this case, they are cached, and will be written to the logger
        only when you mark it as "initialized"

        Example:

        @code
        // the logger
        HPX_DEFINE_LOG(g_l, logger_type)

        // marking the logger as initialized
        g_l()->mark_as_initialized();
        @endcode
        */
        void mark_as_initialized()
        {
            turn_cache_off();
        }

    public:
        HPX_CORE_EXPORT void turn_cache_off();

        // called after all data has been gathered
        void write(message msg) const
        {
            if (m_is_caching_off)
            {
                m_writer(msg);
            }
            else
            {
                m_cache.emplace_back(HPX_MOVE(msg));
            }
        }

    private:
        mutable std::vector<message> m_cache;
        mutable bool m_is_caching_off = false;
        writer::named_write m_writer;
        level m_level;
    };
}    // namespace hpx::util::logging
