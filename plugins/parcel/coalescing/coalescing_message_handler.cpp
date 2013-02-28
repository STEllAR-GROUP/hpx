//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/plugins/parcel/coalescing_message_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    coalescing_message_handler::coalescing_message_handler(std::size_t num)
      : buffer_size_(num)
    {
    }

    coalescing_message_handler::~coalescing_message_handler()
    {
    }

    void coalescing_message_handler::put_parcel(parcelset::parcelport* set,
        parcelset::parcel& p, write_handler_type f)
    {
        set->put_parcel(p, f);
    }
}}}
