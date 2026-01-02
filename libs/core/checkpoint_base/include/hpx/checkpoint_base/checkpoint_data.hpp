// Copyright (c) 2018 Adrian Serio
// Copyright (c) 2018-2026 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/checkpoint_base/checkpoint_data.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/type_support.hpp>

#include <cstddef>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    // tag used to mark serialization archive during check-pointing
    HPX_CXX_CORE_EXPORT struct checkpointing_tag
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    /// save_checkpoint_data
    ///
    /// \tparam Container    Container used to store the check-pointed data.
    /// \tparam Ts           Types of variables to checkpoint
    ///
    /// \param data          Container instance used to store the checkpoint
    ///                      data
    /// \param ts            Variable instances to be inserted into the
    ///                      checkpoint.
    ///
    /// Save_checkpoint_data takes any number of objects which a user may wish
    /// to store in the given container.
    HPX_CXX_CORE_EXPORT template <typename Container, typename... Ts>
    void save_checkpoint_data(Container& data, Ts&&... ts)
    {
        // Create serialization archive from checkpoint data member
        hpx::serialization::output_archive ar(data);

        // force check-pointing flag to be created in the archive, the
        // serialization of id_type's checks for it
        ar.get_extra_data<checkpointing_tag>();

        // Serialize data
        (hpx::serialization::detail::serialize_one(ar, ts), ...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// prepare_checkpoint_data
    ///
    /// \tparam Ts           Types of variables to checkpoint
    ///
    /// \param ts            Variable instances to be inserted into the
    ///                      checkpoint.
    ///
    /// prepare_checkpoint_data takes any number of objects which a user may
    /// wish to store in a subsequent save_checkpoint_data operation. The
    /// function will return the number of bytes necessary to store the data
    /// that will be produced.
    HPX_CXX_CORE_EXPORT template <typename... Ts>
    std::size_t prepare_checkpoint_data(Ts const&... ts)
    {
        // Create serialization archive from special container that collects
        // sizes
        hpx::serialization::detail::preprocess_container data;
        hpx::serialization::output_archive ar(data);

        // force check-pointing flag to be created in the archive, the
        // serialization of id_type's checks for it
        ar.get_extra_data<checkpointing_tag>();

        // Serialize data
        (hpx::serialization::detail::serialize_one(ar, ts), ...);

        return data.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// restore_checkpoint_data
    ///
    /// \tparam Container    Container used to restore the check-pointed data.
    /// \tparam Ts           Types of variables to restore
    ///
    /// \param cont          Container instance used to restore the checkpoint
    ///                      data
    /// \param ts            Variable instances to be restored from the
    ///                      container
    ///
    /// restore_checkpoint_data takes any number of objects which a user may
    /// wish to restore from the given container. The sequence of objects has to
    /// correspond to the sequence of objects for the corresponding call to
    /// save_checkpoint_data that had used the given container instance.
    HPX_CXX_CORE_EXPORT template <typename Container, typename... Ts>
    void restore_checkpoint_data(Container const& cont, Ts&... ts)
    {
        // Create serialization archive
        hpx::serialization::input_archive ar(cont, cont.size());

        // De-serialize data
        (hpx::serialization::detail::serialize_one(ar, ts), ...);
    }

    /// \cond NOINTERNAL
    HPX_CXX_CORE_EXPORT template <typename Container, typename F, typename... Ts>
    void restore_checkpoint_data_func(Container const& cont, F&& f, Ts&... ts)
    {
        // Create serialization archive
        hpx::serialization::input_archive ar(cont, cont.size());

        // De-serialize data
        (f(ar, ts), ...);
    }
    /// \endcond
}    // namespace hpx::util

namespace hpx::util {

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    template <>
    struct extra_data_helper<checkpointing_tag>
    {
        HPX_CORE_EXPORT static extra_data_id_type id() noexcept;
        static constexpr void reset(checkpointing_tag*) noexcept {}
    };
}    // namespace hpx::util
