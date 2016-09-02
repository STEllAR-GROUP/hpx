//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCEL_AWAIT_HPP
#define HPX_PARCELSET_PARCEL_AWAIT_HPP

#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/serialization/detail/preprocess.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/output_container.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset { namespace detail {
    template <typename PutParcel>
    struct parcel_await
      : std::enable_shared_from_this<parcel_await<PutParcel>>
    {
        template <typename PutParcel_>
        parcel_await(parcel&& p, int archive_flags, PutParcel_&& pp)
          : put_parcel_(std::forward<PutParcel_>(pp)),
            p_(std::move(p)),
            archive_(preprocess_, archive_flags, &chunks_),
            overhead_(archive_.bytes_written())
        {
        }

        void apply()
        {
            archive_.reset();
            archive_ << p_;

            // We are doing a fixed point iteration until we are sure that the
            // serialization process requires nothing more to wait on ...
            // Things where we need waiting:
            //  - (shared_)future<id_type>: when the future wasn't ready yet, we
            //      need to do another await round for the id splitting
            //  - id_type: we need to await, if and only if, the credit of the
            //      needs to split.
            if(preprocess_.has_futures())
            {
                HPX_ASSERT(!p_.action_args_bitwise());
                auto this_ = this->shared_from_this();
                preprocess_([this_](){ this_->apply(); });
                return;
            }
            p_.size() = preprocess_.size() + overhead_;
            p_.num_chunks() = chunks_.size();
            p_.set_splitted_gids(std::move(preprocess_.splitted_gids_));
            put_parcel_(std::move(p_));
        }

        typename hpx::util::decay<PutParcel>::type put_parcel_;
        parcel p_;
        hpx::serialization::detail::preprocess preprocess_;
        std::vector<hpx::serialization::serialization_chunk> chunks_;
        hpx::serialization::output_archive archive_;
        std::size_t overhead_;
    };
}}}

#endif
