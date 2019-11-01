// Copyright (c) 2018 Adrian Serio
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
/// This header defines the save_checkpoint and restore_checkpoint functions.
/// These functions are designed to help HPX application developer's checkpoint
/// their applications. Save_checkpoint serializes one or more objects and saves
/// them as a byte stream. Restore_checkpoint converts the byte stream back into
/// instances of the objects.
//

/// \file hpx/util/checkpoint.hpp

#if !defined(CHECKPOINT_HPP_07262017)
#define CHECKPOINT_HPP_07262017

#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/traits/is_client.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    // Forward declarations
    class checkpoint;

    std::ostream& operator<<(std::ostream& ost, checkpoint const& ckp);
    std::istream& operator>>(std::istream& ist, checkpoint& ckp);

    namespace detail {
        struct save_funct_obj;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Checkpoint Object
    ///
    /// Checkpoint is the container object which is produced by save_checkpoint
    /// and is consumed by a restore_checkpoint. A checkpoint may be moved into
    /// the save_checkpoint object to write the byte stream to the pre-created
    /// checkpoint object.
    ///
    /// Checkpoints are able to store all containers which are able to be
    /// serialized including components.
    class checkpoint
    {
        std::vector<char> data_;

        friend std::ostream& operator<<(
            std::ostream& ost, checkpoint const& ckp);
        friend std::istream& operator>>(std::istream& ist, checkpoint& ckp);

        // Serialization Definition
        friend class hpx::serialization::access;
        template <typename Archive>
        void serialize(Archive& arch, const unsigned int version)
        {
            arch& data_;
        }

        friend struct detail::save_funct_obj;

        template <typename T, typename... Ts>
        friend void restore_checkpoint(checkpoint const& c, T& t, Ts&... ts);

    public:
        checkpoint() = default;
        checkpoint(checkpoint const& c)
          : data_(c.data_)
        {
        }
        checkpoint(checkpoint&& c) noexcept
          : data_(std::move(c.data_))
        {
        }
        ~checkpoint() = default;

        // Other Constructors
        checkpoint(std::vector<char> const& vec)
          : data_(vec)
        {
        }
        checkpoint(std::vector<char>&& vec)
          : data_(std::move(vec))
        {
        }

        // Overloads
        checkpoint& operator=(checkpoint const& c)
        {
            if (&c != this)
            {
                data_ = c.data_;
            }
            return *this;
        }
        checkpoint& operator=(checkpoint&& c) noexcept
        {
            if (&c != this)
            {
                data_ = std::move(c.data_);
            }
            return *this;
        }

        friend bool operator==(checkpoint const& lhs, checkpoint const& rhs)
        {
            return lhs.data_ == rhs.data_;
        }
        friend bool operator!=(checkpoint const& lhs, checkpoint const& rhs)
        {
            return !(lhs == rhs);
        }

        // Iterators
        //  expose iterators to access data held by checkpoint
        using const_iterator = std::vector<char>::const_iterator;

        const_iterator begin() const
        {
            return data_.begin();
        }
        const_iterator end() const
        {
            return data_.end();
        }

        // Functions
        size_t size() const
        {
            return data_.size();
        }
    };

    // Stream Overloads

    ///////////////////////////////////////////////////////////////////////////
    /// Operator<< Overload
    ///
    /// \param ost           Output stream to write to.
    ///
    /// \param ckp           Checkpoint to copy from.
    ///
    /// This overload is the main way to write data from a
    /// checkpoint to an object such as a file. Inside
    /// the function, the size of the checkpoint will
    /// be written to the stream before the checkpoint's
    /// data. The operator>> overload uses this to read
    /// the correct number of bytes. Be mindful of this
    /// additional write and read when you use different
    /// facilities to write out or read in data to a
    /// checkpoint!
    ///
    /// \returns Operator<< returns the ostream object.
    ///
    inline std::ostream& operator<<(std::ostream& ost, checkpoint const& ckp)
    {
        // Write the size of the checkpoint to the file
        std::int64_t size = ckp.size();
        ost.write(reinterpret_cast<char const*>(&size), sizeof(std::int64_t));

        // Write the file to the stream
        ost.write(ckp.data_.data(), ckp.size());
        return ost;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Operator>> Overload
    ///
    /// \param ist           Input stream to write from.
    ///
    /// \param ckp           Checkpoint to write to.
    ///
    /// This overload is the main way to read in data from an
    /// object such as a file to a checkpoint.
    /// It is important to note that inside
    /// the function, the first variable to be read is
    /// the size of the checkpoint.
    /// This size variable is written to
    /// the stream before the checkpoint's
    /// data in the operator<< overload.
    /// Be mindful of this
    /// additional read and write when you use different
    /// facilities to read in or write out data from a
    /// checkpoint!
    ///
    /// \returns Operator>> returns the ostream object.
    ///
    inline std::istream& operator>>(std::istream& ist, checkpoint& ckp)
    {
        // Read in the size of the next checkpoint
        std::int64_t length = 0;
        ist.read(reinterpret_cast<char*>(&length), sizeof(std::int64_t));
        ckp.data_.resize(length);

        // Read in the next checkpoint
        ist.read(ckp.data_.data(), length);
        return ist;
    }

    // Function objects for save_checkpoint
    namespace detail {

        // Properly handle non clients
        template <typename T,
            typename U = typename std::enable_if<!hpx::traits::is_client<
                typename std::decay<T>::type>::value>::type>
        T&& prep(T&& t)
        {
            return std::forward<T>(t);
        }

        // Properly handle Clients to components
        template <typename Client, typename Server>
        hpx::future<std::shared_ptr<typename hpx::components::client_base<
            Client, Server>::server_component_type>>
        prep(hpx::components::client_base<Client, Server> const& c)
        {
            // Use shared pointer to serialize server
            using server_type = typename hpx::components::client_base<Client,
                Server>::server_component_type;
            return hpx::get_ptr<server_type>(c.get_id());
        }

        struct save_funct_obj
        {
            template <typename... Ts>
            checkpoint operator()(checkpoint&& c, Ts&&... ts) const
            {
                // Create serialization archive from checkpoint data member
                hpx::serialization::output_archive ar(c.data_);

                // force check-pointing flag to be created in the archive,
                // the serialization of id_type's checks for it
                ar.get_extra_data<naming::checkpointing_tag>();

                // Serialize data

                // Trick to expand the variable pack, akes advantage of the
                // comma operator.
                int const sequencer[] = {0, (ar << ts, 0)...};
                (void) sequencer;    // Suppress unused param. warnings

                return std::move(c);
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Save_checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint
    ///                      to be serialized and placed into a
    ///                      checkpoint object.
    ///
    /// \tparam U            This parameter is used to make sure that T
    ///                      is not a launch policy or a checkpoint. This
    ///                      forces the compiler to choose the correct overload.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes any number of objects which a user may wish
    /// to store and returns a future to a checkpoint object.
    /// This function can also store a component either by passing a
    /// shared_ptr to the component or by passing a component's client
    /// instance to save_checkpoint.
    /// Additionally the function can take a policy as
    /// a first object which changes its behavior depending on the
    /// policy passed to it. Most notably, if a sync policy is used
    /// save_checkpoint will simply return a checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one
    ///          exception: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpoint will simply return
    ///          a checkpoint.
    template <typename T, typename... Ts,
        typename U =
            typename std::enable_if<!hpx::traits::is_launch_policy<T>::value &&
                !std::is_same<typename std::decay<T>::type,
                    checkpoint>::value>::type>
    hpx::future<checkpoint> save_checkpoint(T&& t, Ts&&... ts)
    {
        return hpx::dataflow(detail::save_funct_obj{}, checkpoint{},
            detail::prep(std::forward<T>(t)),
            detail::prep(std::forward<Ts>(ts))...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Save_checkpoint - Take a pre-initialized checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint
    ///                      to be serialized and placed into a
    ///                      checkpoint object.
    ///
    /// \param c             Takes a pre-initialized checkpoint to copy
    ///                      data into.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes any number of objects which a user may wish
    /// to store and returns a future to a checkpoint object.
    /// This function can also store a component either by passing a
    /// shared_ptr to the component or by passing a component's client
    /// instance to save_checkpoint.
    /// Additionally the function can take a policy as a first object which
    /// changes its behavior depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one
    ///          exception: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpoint will simply return
    ///          a checkpoint.

    template <typename T, typename... Ts>
    hpx::future<checkpoint> save_checkpoint(checkpoint&& c, T&& t, Ts&&... ts)
    {
        return hpx::dataflow(detail::save_funct_obj{}, std::move(c),
            detail::prep(std::forward<T>(t)),
            detail::prep(std::forward<Ts>(ts))...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Save_checkpoint - Policy overload
    ///
    /// \tparam T            Containers passed to save_checkpoint to be
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint
    ///                      to be serialized and placed into a
    ///                      checkpoint object.
    ///
    /// \param p             Takes an HPX launch policy. Allows the user
    ///                      to change the way the function is launched
    ///                      i.e. async, sync, etc.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes any number of objects which a user may wish
    /// to store and returns a future to a checkpoint object.
    /// This function can also store a component either by passing a
    /// shared_ptr to the component or by passing a component's client
    /// instance to save_checkpoint.
    /// Additionally the function can take a policy as a first object which
    /// changes its behavior depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one
    ///          exception: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpoint will simply return
    ///          a checkpoint.

    template <typename T, typename... Ts>
    hpx::future<checkpoint> save_checkpoint(hpx::launch p, T&& t, Ts&&... ts)
    {
        return hpx::dataflow(p, detail::save_funct_obj{}, checkpoint{},
            detail::prep(std::forward<T>(t)),
            detail::prep(std::forward<Ts>(ts))...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Save_checkpoint - Policy overload & pre-initialized checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint
    ///                      to be serialized and placed into a
    ///                      checkpoint object.
    ///
    /// \param p             Takes an HPX launch policy. Allows the user
    ///                      to change the way the function is launched
    ///                      i.e. async, sync, etc.
    ///
    /// \param c             Takes a pre-initialized checkpoint to copy
    ///                      data into.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes any number of objects which a user may wish
    /// to store and returns a future to a checkpoint object.
    /// This function can also store a component either by passing a
    /// shared_ptr to the component or by passing a component's client
    /// instance to save_checkpoint.
    /// Additionally the function can take a policy as a first object which
    /// changes its behavior depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one
    ///          exception: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpoint will simply return
    ///          a checkpoint.

    template <typename T, typename... Ts>
    hpx::future<checkpoint> save_checkpoint(
        hpx::launch p, checkpoint&& c, T&& t, Ts&&... ts)
    {
        return hpx::dataflow(p, detail::save_funct_obj{}, std::move(c),
            detail::prep(std::forward<T>(t)),
            detail::prep(std::forward<Ts>(ts))...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Save_checkpoint - Sync_policy overload
    ///
    /// \tparam T            Containers passed to save_checkpoint to be
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint
    ///                      to be serialized and placed into a
    ///                      checkpoint object.
    ///
    /// \tparam U            This parameter is used to make sure that T
    ///                      is not a checkpoint. This forces the compiler
    ///                      to choose the correct overload.
    ///
    /// \param sync_p        hpx::launch::sync_policy
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes any number of objects which a user may wish
    /// to store and returns a future to a checkpoint object.
    /// This function can also store a component either by passing a
    /// shared_ptr to the component or by passing a component's client
    /// instance to save_checkpoint.
    /// Additionally the function can take a policy as a first object which
    /// changes its behavior depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint which is passed hpx::launch::sync_policy
    ///          will return a checkpoint which contains the serialized
    ///          values checkpoint.

    template <typename T, typename... Ts,
        typename U = typename std::enable_if<!std::is_same<
            typename std::decay<T>::type, checkpoint>::value>::type>
    checkpoint save_checkpoint(
        hpx::launch::sync_policy sync_p, T&& t, Ts&&... ts)
    {
        hpx::future<checkpoint> f_chk =
            hpx::dataflow(sync_p, detail::save_funct_obj{}, checkpoint{},
                detail::prep(std::forward<T>(t)),
                detail::prep(std::forward<Ts>(ts))...);
        return f_chk.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Save_checkpoint - Sync_policy overload & pre-init. checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint
    ///                      to be serialized and placed into a
    ///                      checkpoint object.
    ///
    /// \param sync_p        hpx::launch::sync_policy
    ///
    /// \param c             Takes a pre-initialized checkpoint to copy
    ///                      data into.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes any number of objects which a user may wish
    /// to store and returns a future to a checkpoint object.
    /// This function can also store a component either by passing a
    /// shared_ptr to the component or by passing a component's client
    /// instance to save_checkpoint.
    /// Additionally the function can take a policy as a first object which
    /// changes its behavior depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint which is passed hpx::launch::sync_policy
    ///          will return a checkpoint which contains the serialized
    ///          values checkpoint.
    template <typename T, typename... Ts>
    checkpoint save_checkpoint(
        hpx::launch::sync_policy sync_p, checkpoint&& c, T&& t, Ts&&... ts)
    {
        hpx::future<checkpoint> f_chk =
            hpx::dataflow(sync_p, detail::save_funct_obj{}, std::move(c),
                detail::prep(std::forward<T>(t)),
                detail::prep(std::forward<Ts>(ts))...);
        return f_chk.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Properly handle non client/server restoration
        template <typename T,
            typename U = typename std::enable_if<
                !hpx::traits::is_client<T>::value>::type>
        void restore_impl(hpx::serialization::input_archive& ar, T& t)
        {
            ar >> t;
        }

        // Properly handle client/server restoration
        template <typename Client, typename Server>
        void restore_impl(hpx::serialization::input_archive& ar,
            hpx::components::client_base<Client, Server>& c)
        {
            // Revive server
            using server_component_type =
                typename hpx::components::client_base<Client,
                    Server>::server_component_type;

            hpx::future<std::shared_ptr<server_component_type>> f_server_ptr;
            ar >> f_server_ptr;
            c = hpx::new_<Client>(
                hpx::find_here(), std::move(*(f_server_ptr.get())));
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Restore_checkpoint
    ///
    /// Restore_checkpoint takes a checkpoint object as a first argument and
    /// the containers which will be filled from the byte stream (in the same
    /// order as they were placed in save_checkpoint). Restore_checkpoint can
    /// resurrect a stored component in two ways: by passing in a instance of
    /// a component's shared_ptr or by passing in an
    /// instance of the component's client.
    ///
    /// \tparam T           A container to restore.
    ///
    /// \tparam Ts          Other containers to restore. Containers
    ///                     must be in the same order that they were
    ///                     inserted into the checkpoint.
    ///
    /// \param c            The checkpoint to restore.
    ///
    /// \param t            A container to restore.
    ///
    /// \param ts           Other containers to restore Containers
    ///                     must be in the same order that they were
    ///                     inserted into the checkpoint.
    ///
    /// \returns Restore_checkpoint returns void.
    template <typename T, typename... Ts>
    void restore_checkpoint(checkpoint const& c, T& t, Ts&... ts)
    {
        // Create serialization archive
        hpx::serialization::input_archive ar(c.data_, c.size());

        // De-serialize data
        detail::restore_impl(ar, t);

        // Trick to expand the variable pack, takes advantage of the comma
        // operator
        int const sequencer[] = {0, (detail::restore_impl(ar, ts), 0)...};
        (void) sequencer;    // Suppress unused variable warnings
    }

}}    // namespace hpx::util

#endif
