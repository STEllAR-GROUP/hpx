//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_PARCELSET_DETAIL_CALL_FOR_EACH_HPP
#define HPX_RUNTIME_PARCELSET_DETAIL_CALL_FOR_EACH_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assertion.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct call_for_each
        {
            typedef std::vector<parcelport::write_handler_type> handlers_type;
            typedef std::vector<parcel> parcels_type;
            handlers_type handlers_;
            parcels_type parcels_;

            call_for_each(handlers_type&& handlers, parcels_type&& parcels)
              : handlers_(std::move(handlers))
              , parcels_(std::move(parcels))
            {}

            call_for_each(call_for_each&& other)
              : handlers_(std::move(other.handlers_))
              , parcels_(std::move(other.parcels_))
            {}

            call_for_each& operator=(call_for_each&& other)
            {
                handlers_ = std::move(other.handlers_);
                parcels_ = std::move(other.parcels_);

                return *this;
            }

            void operator()(boost::system::error_code const& e)
            {
                HPX_ASSERT(parcels_.size() == handlers_.size());
                for(std::size_t i = 0; i < parcels_.size(); ++i)
                {
                    handlers_[i](e, parcels_[i]);
                    handlers_[i].reset();
                }
            }
        };
    }
}}

#endif
#endif
