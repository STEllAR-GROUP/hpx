//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_BUFFER_MAR_07_2013_1250PM)
#define HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_BUFFER_MAR_07_2013_1250PM

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace hpx { namespace plugins { namespace parcel { namespace detail
{
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
        {}

        message_buffer(std::size_t max_messages)
          : max_messages_(max_messages)
        {
            messages_.reserve(max_messages);
            handlers_.reserve(max_messages);
        }

        message_buffer(message_buffer && rhs)
          : dest_(std::move(rhs.dest_)),
            messages_(std::move(rhs.messages_)),
            handlers_(std::move(rhs.handlers_)),
            max_messages_(rhs.max_messages_)
        {}

        message_buffer& operator=(message_buffer && rhs)
        {
            if (&rhs != this) {
                max_messages_ = rhs.max_messages_;
                dest_ = std::move(rhs.dest_);
                messages_ = std::move(rhs.messages_);
                handlers_ = std::move(rhs.handlers_);
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
                    void (parcelport::*put_parcel_ptr) (
                            parcelset::locality const&,
                            std::vector<parcelset::parcel>,
                            std::vector<parcelset::write_handler_type>
                        ) = &parcelport::put_parcels;

                    threads::register_thread_nullary(
                        util::deferred_call(put_parcel_ptr, pp,
                            dest_, std::move(messages_), std::move(handlers_)),
                        "parcelhandler::put_parcel", threads::pending, true,
                        threads::thread_priority_boost);
                    return;
                }

                pp->put_parcels(dest_, std::move(messages_), std::move(handlers_));
            }
        }

        message_buffer_append_state append(parcelset::locality const & dest,
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

            messages_.push_back(std::move(p));
            handlers_.push_back(std::move(f));

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

        std::size_t capacity() const { return max_messages_; }

    private:
        parcelset::locality dest_;
        std::vector<parcelset::parcel> messages_;
        std::vector<parcelset::write_handler_type> handlers_;
        std::size_t max_messages_;
    };
}}}}

#endif
