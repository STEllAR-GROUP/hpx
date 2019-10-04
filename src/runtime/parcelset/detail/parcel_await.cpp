//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/detail/preprocess.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset { namespace detail
{
    template <typename Parcel, typename Handler, typename Derived>
    struct parcel_await_base : std::enable_shared_from_this<Derived>
    {
        typedef hpx::util::unique_function_nonser<void(Parcel&&, Handler&&)>
            put_parcel_type;

        parcel_await_base(Parcel&& parcel, Handler&& handler, int archive_flags,
            put_parcel_type pp)
          : put_parcel_(std::move(pp))
          , parcel_(std::move(parcel))
          , handler_(std::move(handler))
          , archive_(preprocess_, archive_flags)
          , overhead_(archive_.bytes_written())
        {}

        void done()
        {
            put_parcel_(std::move(parcel_), std::move(handler_));
        }

        bool apply_single(parcel &p)
        {
            archive_.reset();
            archive_ << p;

            // We are doing a fixed point iteration until we are sure that the
            // serialization process requires nothing more to wait on ...
            // Things where we need waiting:
            //  - (shared_)future<id_type>: when the future wasn't ready yet, we
            //      need to do another await round for the id splitting
            //  - id_type: we need to await, if and only if, the credit of the
            //      needs to split.
            if(preprocess_.has_futures())
            {
                auto this_ = this->shared_from_this();
                preprocess_([this_](){ this_->apply(); });
                return false;
            }
            archive_.flush();
            p.size() = preprocess_.size() + overhead_;
            p.num_chunks() = archive_.get_num_chunks();
            hpx::serialization::detail::preprocess::split_gids_map split_gids;
            std::swap(split_gids, preprocess_.split_gids_);
            p.set_split_gids(std::move(split_gids));

            return true;
        }

        put_parcel_type put_parcel_;
        Parcel parcel_;
        Handler handler_;
        hpx::serialization::detail::preprocess preprocess_;
        hpx::serialization::output_archive archive_;
        std::size_t overhead_;
    };

    struct parcel_await
      : parcel_await_base<parcel, write_handler_type, parcel_await>
    {
        typedef parcel_await_base<parcel, write_handler_type, parcel_await>
            base_type;
        parcel_await(parcel&& p, write_handler_type&& f, int archive_flags,
            put_parcel_type pp);

        void apply();
    };

    struct parcels_await
      : parcel_await_base<std::vector<parcel>, std::vector<write_handler_type>,
            parcels_await>
    {
        typedef parcel_await_base<std::vector<parcel>,
            std::vector<write_handler_type>, parcels_await>
            base_type;

        parcels_await(std::vector<parcel>&& p, std::vector<write_handler_type>&& f,
            int archive_flags, put_parcel_type pp);

        void apply();

        std::size_t idx_;
    };

    ///////////////////////////////////////////////////////////////////////////
    parcel_await::parcel_await(parcel&& p, write_handler_type&& f,
        int archive_flags, parcel_await::put_parcel_type pp)
    : base_type(std::move(p), std::move(f), archive_flags, std::move(pp))
    {
    }

    void parcel_await::apply()
    {
        if (apply_single(parcel_))
        {
            done();
        }
    }

    parcels_await::parcels_await(std::vector<parcel>&& p,
            std::vector<write_handler_type>&& f, int archive_flags,
            parcels_await::put_parcel_type pp)
      : base_type(std::move(p), std::move(f), archive_flags, std::move(pp)),
        idx_(0)
    {
    }

    void parcels_await::apply()
    {
        for (/*idx_*/; idx_ != parcel_.size(); ++idx_)
        {
            if(!apply_single(parcel_[idx_]))
                return;
        }
        done();
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcel_await_apply(parcel&& p, write_handler_type&& f,
        int archive_flags, put_parcel_type pp)
    {
        std::make_shared<parcel_await>(
                std::move(p), std::move(f), 0, std::move(pp)
            )->apply();
    }

    void parcels_await_apply(std::vector<parcel>&& p,
        std::vector<write_handler_type>&& f, int archive_flags,
        put_parcels_type pp)
    {
        std::make_shared<parcels_await>(
                std::move(p), std::move(f), 0, std::move(pp)
            )->apply();
    }
}}}

#endif
