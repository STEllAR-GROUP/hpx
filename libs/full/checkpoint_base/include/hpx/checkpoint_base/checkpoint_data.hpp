// Copyright (c) 2018 Adrian Serio
// Copyright (c) 2018-2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/checkpoint_base/checkpoint_data.hpp

#pragma once

#include <hpx/serialization/detail/extra_archive_data.hpp>
#include <hpx/serialization/detail/preprocess_container.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    // tag used to mark serialization archive during check-pointing
    struct checkpointing_tag
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    /// save_checkpoint_data
    ///
    /// \tparam Container    Container used to store the check-pointed data.
    /// \tparam Ts           Types of variables to checkpoint
    ///
    /// \param cont          Container instance used to store the checkpoint data
    /// \param ts            Variable instances to be inserted into the checkpoint.
    ///
    /// Save_checkpoint_data takes any number of objects which a user may wish
    /// to store in the given container.
    template <typename Container, typename... Ts>
    void save_checkpoint_data(Container& data, Ts&&... ts)
    {
        // Create serialization archive from checkpoint data member
        hpx::serialization::output_archive ar(data);

        // force check-pointing flag to be created in the archive,
        // the serialization of id_type's checks for it
        ar.get_extra_data<checkpointing_tag>();

        // Serialize data

        // Trick to expand the variable pack, takes advantage of the
        // comma operator.
        int const sequencer[] = {0, (ar << ts, 0)...};
        (void) sequencer;    // Suppress unused param. warnings
    }

    ///////////////////////////////////////////////////////////////////////////
    /// prepare_checkpoint_data
    ///
    /// \tparam Ts           Types of variables to checkpoint
    ///
    /// \param ts            Variable instances to be inserted into the checkpoint.
    ///
    /// prepare_checkpoint_data takes any number of objects which a user may
    /// wish to store in a subsequent save_checkpoint_data operation. The
    /// function will return the number of bytes necessary to store the data
    /// that will be produced.
    template <typename... Ts>
    std::size_t prepare_checkpoint_data(Ts const&... ts)
    {
        // Create serialization archive from special container that collects
        // sizes
        hpx::serialization::detail::preprocess_container data;
        hpx::serialization::output_archive ar(data);

        // force check-pointing flag to be created in the archive,
        // the serialization of id_type's checks for it
        ar.get_extra_data<checkpointing_tag>();

        // Serialize data

        // Trick to expand the variable pack, takes advantage of the
        // comma operator.
        int const sequencer[] = {0, (ar << ts, 0)...};
        (void) sequencer;    // Suppress unused param. warnings

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
    /// \param ts            Variable instances to be restored from the container
    ///
    /// restore_checkpoint_data takes any number of objects which a user may wish
    /// to restore from the given container. The sequence of objects has to
    /// correspond to the sequence of objects for the corresponding call to
    /// save_checkpoint_data that had used the given container instance.
    template <typename Container, typename... Ts>
    void restore_checkpoint_data(Container const& cont, Ts&... ts)
    {
        // Create serialization archive
        hpx::serialization::input_archive ar(cont, cont.size());

        // De-serialize data

        // Trick to expand the variable pack, takes advantage of the comma
        // operator
        int const sequencer[] = {0, (ar >> ts, 0)...};
        (void) sequencer;    // Suppress unused variable warnings
    }

    /// \cond NOINTERNAL
    template <typename Container, typename F, typename... Ts>
    void restore_checkpoint_data_func(Container const& cont, F&& f, Ts&... ts)
    {
        // Create serialization archive
        hpx::serialization::input_archive ar(cont, cont.size());

        // De-serialize data

        // Trick to expand the variable pack, takes advantage of the comma
        // operator
        int const sequencer[] = {0, (f(ar, ts), 0)...};
        (void) sequencer;    // Suppress unused variable warnings
    }
    /// \endcond
}}    // namespace hpx::util

namespace hpx { namespace serialization { namespace detail {
    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    template <>
    struct extra_archive_data_helper<hpx::util::checkpointing_tag>
    {
        HPX_EXPORT static extra_archive_data_id_type id() noexcept;
        static constexpr void reset(hpx::util::checkpointing_tag*) noexcept {}
    };
}}}    // namespace hpx::serialization::detail
