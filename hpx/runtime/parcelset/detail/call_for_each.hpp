//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_PARCELSET_DETAIL_CALL_FOR_EACH_HPP
#define HPX_RUNTIME_PARCELSET_DETAIL_CALL_FOR_EACH_HPP

#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/move.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct call_for_each
        {
            HPX_MOVABLE_BUT_NOT_COPYABLE(call_for_each);
        public:
            typedef void result_type;

            typedef std::vector<parcelport::write_handler_type> handlers_type;
            typedef std::vector<parcel> parcels_type;
            handlers_type handlers_;
            parcels_type parcels_;

            call_for_each(handlers_type&& handlers, parcels_type && parcels)
              : handlers_(std::move(handlers))
              , parcels_(std::move(parcels))
            {}

            call_for_each(call_for_each &&other)
              : handlers_(std::move(other.handlers_))
              , parcels_(std::move(other.parcels_))
            {}

            call_for_each& operator=(call_for_each &&other)
            {
                handlers_ = std::move(other.handlers_);
                parcels_ = std::move(other.parcels_);

                return *this;
            }

            result_type operator()(boost::system::error_code const& e) const
            {
                HPX_ASSERT(parcels_.size() == handlers_.size());
                for(std::size_t i = 0; i < parcels_.size(); ++i)
                {
                    handlers_[i](e, parcels_[i]);
                }
            }
        };
    }
}}

#endif
