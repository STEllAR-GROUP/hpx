//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/sender.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    struct sender;
    struct sender_connection
      : parcelset::parcelport_connection<sender_connection, std::vector<char>>
    {
    private:
        using sender_type = sender;
        using write_handler_type =
            hpx::function<void(std::error_code const&, parcel const&)>;
        using data_type = std::vector<char>;
        using base_type =
            parcelset::parcelport_connection<sender_connection, data_type>;
        using handler_type = hpx::move_only_function<void(error_code const&)>;
        using postprocess_handler_type = hpx::move_only_function<void(
            error_code const&, parcelset::locality const&,
            std::shared_ptr<sender_connection>)>;

    public:
        sender_connection(int dst, parcelset::parcelport* pp)
          : dst_rank(dst)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_rank)))
        {
        }

        ~sender_connection() {}

        parcelset::locality const& destination() const noexcept
        {
            return there_;
        }

        constexpr void verify_(
            parcelset::locality const& /* parcel_locality_id */) const noexcept
        {
        }

        void async_write(handler_type&& handler,
            postprocess_handler_type&& parcel_postprocess);

        bool can_be_eager_message(size_t max_header_size);

        void load(handler_type&& handler,
            postprocess_handler_type&& parcel_postprocess);

        bool isEager();
        bool send();

        void done();

        bool tryMerge(const std::shared_ptr<sender_connection>& other);

        int dst_rank;
        handler_type handler_;
        postprocess_handler_type postprocess_handler_;
        bool is_eager;
        LCI_mbuffer_t mbuffer;
        LCI_iovec_t iovec;
        std::shared_ptr<sender_connection>* sharedPtr_p;    // for LCI_putva
        parcelset::parcelport* pp_;
        parcelset::locality there_;
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        parcelset::data_point data_point_;
#endif
    };
}    // namespace hpx::parcelset::policies::lci

#endif
