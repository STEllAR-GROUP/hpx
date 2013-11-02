//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_BUFFER_MAR_07_2013_1250PM)
#define HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_BUFFER_MAR_07_2013_1250PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/move.hpp>

#include <vector>

#include <boost/noncopyable.hpp>

namespace hpx { namespace plugins { namespace parcel { namespace detail
{
    class message_buffer
    {
        BOOST_COPYABLE_AND_MOVABLE(message_buffer);

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
        {}

        message_buffer(message_buffer const& rhs)
          : messages_(rhs.messages_),
            handlers_(rhs.handlers_),
            max_messages_(rhs.max_messages_)
        {}

        message_buffer(BOOST_RV_REF(message_buffer) rhs)
          : messages_(boost::move(rhs.messages_)),
            handlers_(boost::move(rhs.handlers_)),
            max_messages_(rhs.max_messages_)
        {}

        message_buffer& operator=(BOOST_COPY_ASSIGN_REF(message_buffer) rhs)
        {
            if (&rhs != this) {
                max_messages_ = rhs.max_messages_;
                messages_ = rhs.messages_;
                handlers_ = rhs.handlers_;
            }
            return *this;
        }

        message_buffer& operator=(BOOST_RV_REF(message_buffer) rhs)
        {
            if (&rhs != this) {
                max_messages_ = rhs.max_messages_;
                messages_ = boost::move(rhs.messages_);
                handlers_ = boost::move(rhs.handlers_);
            }
            return *this;
        }

        void operator()(parcelset::parcelport* set)
        {
            if (!messages_.empty())
                set->put_parcels(messages_, handlers_);
        }

        message_buffer_append_state append(parcelset::parcel const& p,
            parcelset::parcelport::write_handler_type const& f)
        {
            BOOST_ASSERT(messages_.size() == handlers_.size());

            int result = normal;
            if (messages_.empty())
                result = first_message;

            messages_.push_back(p);
            handlers_.push_back(f);

            if (messages_.size() >= max_messages_)
                result = buffer_now_full;

            return message_buffer_append_state(result);
        }

        bool empty() const
        {
            BOOST_ASSERT(messages_.size() == handlers_.size());
            return messages_.empty();
        }

        void clear()
        {
            messages_.clear();
            handlers_.clear();
        }

        std::size_t size() const
        {
            BOOST_ASSERT(messages_.size() == handlers_.size());
            return messages_.size();
        }

        double fill_ratio() const
        {
            return double(messages_.size()) / double(max_messages_);
        }

        void swap(message_buffer& o)
        {
            std::swap(max_messages_, o.max_messages_);
            std::swap(messages_, o.messages_);
            std::swap(handlers_, o.handlers_);
        }

        std::size_t capacity() const { return max_messages_; }

    private:
        std::vector<parcelset::parcel> messages_;
        std::vector<parcelset::parcelport::write_handler_type> handlers_;
        std::size_t max_messages_;
    };
}}}}

#endif
