//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCEL_COALESCING)
#include <hpx/assert.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/threading_base.hpp>

#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/parcelset_base/locality.hpp>
#include <hpx/parcelset_base/parcelport.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace hpx::plugins::parcel::detail {

    class message_buffer
    {
    public:
        enum message_buffer_append_state
        {
            normal = 0,
            first_message = 1,
            buffer_now_full = 2,
            singleton_buffer = 3
        };

        message_buffer()
          : max_messages_(0)
        {
        }

        explicit message_buffer(std::size_t max_messages)
          : max_messages_(max_messages)
        {
            messages_.reserve(max_messages);
            handlers_.reserve(max_messages);
        }

        message_buffer(message_buffer&& rhs) noexcept
          : dest_(HPX_MOVE(rhs.dest_))
          , messages_(HPX_MOVE(rhs.messages_))
          , handlers_(HPX_MOVE(rhs.handlers_))
          , max_messages_(rhs.max_messages_)
        {
        }

        message_buffer& operator=(message_buffer&& rhs)
        {
            if (&rhs != this)
            {
                max_messages_ = rhs.max_messages_;
                dest_ = HPX_MOVE(rhs.dest_);
                messages_ = HPX_MOVE(rhs.messages_);
                handlers_ = HPX_MOVE(rhs.handlers_);
            }
            return *this;
        }

        void operator()(parcelset::parcelport* pp)
        {
            if (!messages_.empty())
            {
                if (nullptr == threads::get_self_ptr())
                {
                    // reschedule this call on a new HPX thread
                    using parcelset::parcelport;
                    void (parcelport::*put_parcel_ptr)(
                        parcelset::locality const&,
                        std::vector<parcelset::parcel>,
                        std::vector<parcelset::write_handler_type>) =
                        &parcelport::put_parcels;

                    threads::thread_init_data data(
                        threads::make_thread_function_nullary(
                            util::deferred_call(put_parcel_ptr, pp, dest_,
                                HPX_MOVE(messages_), HPX_MOVE(handlers_))),
                        "parcelhandler::put_parcel",
                        threads::thread_priority::boost,
                        threads::thread_schedule_hint(),
                        threads::thread_stacksize::default_,
                        threads::thread_schedule_state::pending, true);
                    threads::register_thread(data);
                    return;
                }

                pp->put_parcels(
                    dest_, HPX_MOVE(messages_), HPX_MOVE(handlers_));
            }
        }

        message_buffer_append_state append(parcelset::locality const& dest,
            parcelset::parcel p, parcelset::write_handler_type f)
        {
            int result = normal;
            if (messages_.empty())
            {
                HPX_ASSERT(handlers_.empty());

                result = first_message;
                dest_ = dest;
            }
            else
            {
                HPX_ASSERT(messages_.size() == handlers_.size());
                HPX_ASSERT(dest_ == dest);
            }

            messages_.push_back(HPX_MOVE(p));
            handlers_.push_back(HPX_MOVE(f));

            if (messages_.size() >= max_messages_)
                result = buffer_now_full;

            return message_buffer_append_state(result);
        }

        bool empty() const
        {
            HPX_ASSERT(messages_.size() == handlers_.size());
            return messages_.empty();
        }

        void clear()
        {
            dest_ = parcelset::locality();
            messages_.clear();
            handlers_.clear();

            messages_.reserve(max_messages_);
            handlers_.reserve(max_messages_);
        }

        std::size_t size() const
        {
            HPX_ASSERT(messages_.size() == handlers_.size());
            return messages_.size();
        }

        double fill_ratio() const
        {
            return double(messages_.size()) / double(max_messages_);
        }

        void swap(message_buffer& o)
        {
            std::swap(max_messages_, o.max_messages_);
            std::swap(dest_, o.dest_);
            std::swap(messages_, o.messages_);
            std::swap(handlers_, o.handlers_);
        }

        std::size_t capacity() const
        {
            return max_messages_;
        }

    private:
        parcelset::locality dest_;
        std::vector<parcelset::parcel> messages_;
        std::vector<parcelset::write_handler_type> handlers_;
        std::size_t max_messages_;
    };
}    // namespace hpx::plugins::parcel::detail

#endif
