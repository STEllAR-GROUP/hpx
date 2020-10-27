//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/actions/actions_fwd.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/lcos_local/detail/preprocess_future.hpp>
#include <hpx/naming/detail/preprocess_gid_types.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/serialization/detail/preprocess_container.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset { namespace detail {

    template <typename Parcel, typename Handler, typename Derived>
    struct parcel_await_base : std::enable_shared_from_this<Derived>
    {
        using put_parcel_type =
            hpx::util::unique_function_nonser<void(Parcel&&, Handler&&)>;

        parcel_await_base(Parcel&& parcel, Handler&& handler,
            std::uint32_t archive_flags, put_parcel_type pp)
          : put_parcel_(std::move(pp))
          , parcel_(std::move(parcel))
          , handler_(std::move(handler))
          , archive_(data_, archive_flags)
          , overhead_(archive_.bytes_written())
        {
        }

        void done()
        {
            put_parcel_(std::move(parcel_), std::move(handler_));
        }

        bool apply_single(parcel& p)
        {
            archive_.reset();

            archive_ << p;

            auto* handle_futures = archive_.try_get_extra_data<
                serialization::detail::preprocess_futures>();

            // We are doing a fixed point iteration until we are sure that the
            // serialization process requires nothing more to wait on ...
            // Things where we need waiting:
            //  - (shared_)future<id_type>: when the future wasn't ready yet, we
            //      need to do another await round for the id splitting
            //  - id_type: we need to await, if and only if, the credit of the
            //      needs to split.
            if (handle_futures && handle_futures->has_futures())
            {
                auto this_ = this->shared_from_this();
                (*handle_futures)([this_]() { this_->apply(); });
                return false;
            }

            archive_.flush();

            p.size() = data_.size() + overhead_;
            p.num_chunks() = archive_.get_num_chunks();

            auto* split_gids = archive_.try_get_extra_data<
                serialization::detail::preprocess_gid_types>();
            if (split_gids)
            {
                p.set_split_gids(split_gids->move_split_gids());
            }

            return true;
        }

        put_parcel_type put_parcel_;
        Parcel parcel_;
        Handler handler_;
        hpx::serialization::detail::preprocess_container data_;
        hpx::serialization::output_archive archive_;
        std::size_t overhead_;
    };

    struct parcel_await
      : parcel_await_base<parcel, write_handler_type, parcel_await>
    {
        using base_type =
            parcel_await_base<parcel, write_handler_type, parcel_await>;
        parcel_await(parcel&& p, write_handler_type&& f,
            std::uint32_t archive_flags, put_parcel_type pp);

        void apply();
    };

    struct parcels_await
      : parcel_await_base<std::vector<parcel>, std::vector<write_handler_type>,
            parcels_await>
    {
        using base_type = parcel_await_base<std::vector<parcel>,
            std::vector<write_handler_type>, parcels_await>;

        parcels_await(std::vector<parcel>&& p,
            std::vector<write_handler_type>&& f, std::uint32_t archive_flags,
            put_parcel_type pp);

        void apply();

        std::size_t idx_;
    };

    ///////////////////////////////////////////////////////////////////////////
    parcel_await::parcel_await(parcel&& p, write_handler_type&& f,
        std::uint32_t archive_flags, parcel_await::put_parcel_type pp)
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
        std::vector<write_handler_type>&& f, std::uint32_t archive_flags,
        parcels_await::put_parcel_type pp)
      : base_type(std::move(p), std::move(f), archive_flags, std::move(pp))
      , idx_(0)
    {
    }

    void parcels_await::apply()
    {
        for (/*idx_*/; idx_ != parcel_.size(); ++idx_)
        {
            if (!apply_single(parcel_[idx_]))
                return;
        }
        done();
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcel_await_apply(parcel&& p, write_handler_type&& f,
        std::uint32_t archive_flags, put_parcel_type pp)
    {
        auto ptr = std::make_shared<parcel_await>(
            std::move(p), std::move(f), archive_flags, std::move(pp));
        ptr->apply();
    }

    void parcels_await_apply(std::vector<parcel>&& p,
        std::vector<write_handler_type>&& f, std::uint32_t archive_flags,
        put_parcels_type pp)
    {
        auto ptr = std::make_shared<parcels_await>(
            std::move(p), std::move(f), archive_flags, std::move(pp));
        ptr->apply();
    }
}}}    // namespace hpx::parcelset::detail

#endif
