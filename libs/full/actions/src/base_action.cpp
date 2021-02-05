//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    struct action_serialization_data
    {
        action_serialization_data()
          : parent_locality_(naming::invalid_locality_id)
          , parent_id_(static_cast<std::uint64_t>(0))
          , parent_phase_(0)
          , priority_(static_cast<threads::thread_priority>(0))
          , stacksize_(static_cast<threads::thread_stacksize>(0))
        {
        }

        action_serialization_data(std::uint32_t parent_locality,
            threads::thread_id_type parent_id, std::uint64_t parent_phase,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize)
          : parent_locality_(parent_locality)
          , parent_id_(reinterpret_cast<std::uint64_t>(parent_id.get()))
          , parent_phase_(parent_phase)
          , priority_(priority)
          , stacksize_(stacksize)
        {
        }

        std::uint32_t parent_locality_;
        std::uint64_t parent_id_;
        std::uint64_t parent_phase_;
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & parent_id_ & parent_phase_ & parent_locality_ & priority_ &
                stacksize_;
            // clang-format on
        }
    };
}}}    // namespace hpx::actions::detail

HPX_IS_BITWISE_SERIALIZABLE(hpx::actions::detail::action_serialization_data)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions {
    ///////////////////////////////////////////////////////////////////////////
    base_action::~base_action() {}

    ///////////////////////////////////////////////////////////////////////////
    base_action_data::base_action_data(
        threads::thread_priority priority, threads::thread_stacksize stacksize)
      : priority_(priority)
      , stacksize_(stacksize)
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
      , parent_locality_(base_action_data::get_locality_id())
      , parent_id_(threads::get_parent_id())
      , parent_phase_(threads::get_parent_phase())
#endif
    {
    }

    // serialization support
    // loading ...
    void base_action_data::load_base(hpx::serialization::input_archive& ar)
    {
        // Always serialize the parent information to maintain binary
        // compatibility on the wire.

        detail::action_serialization_data data;
        ar >> data;

#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        parent_locality_ = data.parent_locality_;
        parent_id_ = threads::thread_id_type(
            reinterpret_cast<threads::thread_data*>(data.parent_id_));
        parent_phase_ = data.parent_phase_;
#endif
        priority_ = data.priority_;
        stacksize_ = data.stacksize_;
    }

    // saving ...
    void base_action_data::save_base(hpx::serialization::output_archive& ar)
    {
        // Always serialize the parent information to maintain binary
        // compatibility on the wire.

#if !defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        std::uint32_t parent_locality_ = naming::invalid_locality_id;
        threads::thread_id_type parent_id_;
        std::uint64_t parent_phase_ = 0;
#endif
        detail::action_serialization_data data(
            parent_locality_, parent_id_, parent_phase_, priority_, stacksize_);
        ar << data;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::uint32_t base_action_data::get_locality_id()
    {
        error_code ec(lightweight);    // ignore any errors
        return hpx::get_locality_id(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
#if !defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
    /// Return the locality of the parent thread
    std::uint32_t base_action_data::get_parent_locality_id() const
    {
        return naming::invalid_locality_id;
    }

    /// Return the thread id of the parent thread
    threads::thread_id_type base_action_data::get_parent_thread_id() const
    {
        return threads::invalid_thread_id;
    }

    /// Return the phase of the parent thread
    std::uint64_t base_action_data::get_parent_thread_phase() const
    {
        return 0;
    }
#else
    /// Return the locality of the parent thread
    std::uint32_t base_action_data::get_parent_locality_id() const
    {
        return parent_locality_;
    }

    /// Return the thread id of the parent thread
    threads::thread_id_type base_action_data::get_parent_thread_id() const
    {
        return parent_id_;
    }

    /// Return the phase of the parent thread
    std::uint64_t base_action_data::get_parent_thread_phase() const
    {
        return parent_phase_;
    }
#endif

    /// Return the thread priority this action has to be executed with
    threads::thread_priority base_action_data::get_thread_priority() const
    {
        return priority_;
    }

    /// Return the thread stacksize this action has to be executed with
    threads::thread_stacksize base_action_data::get_thread_stacksize() const
    {
        return stacksize_;
    }
}}    // namespace hpx::actions

#endif
