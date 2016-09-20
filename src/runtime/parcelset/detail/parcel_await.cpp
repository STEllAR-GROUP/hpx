//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset { namespace detail {

    parcel_await::parcel_await(parcel&& p, int archive_flags,
        parcel_await::put_parcel_type pp)
      : put_parcel_(std::move(pp)),
        archive_(preprocess_, archive_flags),
        overhead_(archive_.bytes_written()),
        idx_(0)
    {
        parcels_.push_back(std::move(p));
    }

    parcel_await::parcel_await(std::vector<parcel>&& parcels, int archive_flags,
        parcel_await::put_parcel_type pp)
      : put_parcel_(std::move(pp)),
        parcels_(std::move(parcels)),
        archive_(preprocess_, archive_flags),
        overhead_(archive_.bytes_written()),
        idx_(0)
    {
    }

    void parcel_await::apply()
    {
        for (/*idx_*/; idx_ != parcels_.size(); ++idx_)
        {
            archive_.reset();
            archive_ << parcels_[idx_];

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
                return;
            }
            archive_.flush();
            parcels_[idx_].size() = preprocess_.size() + overhead_;
            parcels_[idx_].num_chunks() = archive_.get_num_chunks();
            parcels_[idx_].set_split_gids(std::move(preprocess_.split_gids_));
            put_parcel_(std::move(parcels_[idx_]));
        }
    }

}}}
