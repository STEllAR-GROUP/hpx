//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/basename_registration.hpp

#if !defined(HPX_RUNTIME_BASENAME_REGISTRATION_AUG_17_2015_0432PM)
#define HPX_RUNTIME_BASENAME_REGISTRATION_AUG_17_2015_0432PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/naming/id_type.hpp>

#include <string>
#include <utility>
#include <vector>

namespace hpx
{
    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_API_EXPORT std::string name_from_basename(
            std::string const& basename, std::size_t idx);
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// Return all registered ids from all localities from the given base
    /// name.
    ///
    /// This function locates all ids which were registered with the given
    /// base name. It returns a list of futures representing those ids.
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param num_ids      [in] The number of registered ids to expect.
    ///
    /// \returns A list of futures representing the ids which were registered
    ///          using the given base name.
    ///
    /// \note   The futures will become ready even if the event (for instance,
    ///         binding the name to an id) has already happened in the past.
    ///         This is important in order to reliably retrieve ids from a
    ///         name, even if the name was already registered.
    ///
    HPX_API_EXPORT std::vector<hpx::future<hpx::id_type> >
        find_all_from_basename(std::string const& base_name, std::size_t num_ids);

    ///////////////////////////////////////////////////////////////////////////
    /// Return all registered clients from all localities from the given base
    /// name.
    ///
    /// This function locates all ids which were registered with the given
    /// base name. It returns a list of futures representing those ids.
    ///
    /// \tparam Client     The client type to return
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param num_ids      [in] The number of registered ids to expect.
    ///
    /// \returns A list of futures representing the ids which were registered
    ///          using the given base name.
    ///
    /// \note   The futures embedded in the returned client objects will become
    ///         ready even if the event (for instance,
    ///         binding the name to an id) has already happened in the past.
    ///         This is important in order to reliably retrieve ids from a
    ///         name, even if the name was already registered.
    ///
    template <typename Client>
    std::vector<Client>
    find_all_from_basename(std::string const& base_name, std::size_t num_ids)
    {
        return components::make_clients<Client>(
            find_all_from_basename(base_name, num_ids));
    }

    /// Return registered ids from the given base name and sequence numbers.
    ///
    /// This function locates the ids which were registered with the given
    /// base name and the given sequence numbers. It returns a list of futures
    /// representing those ids.
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param ids          [in] The sequence numbers of the registered ids.
    ///
    /// \returns A list of futures representing the ids which were registered
    ///          using the given base name and sequence numbers.
    ///
    /// \note   The futures will become ready even if the event (for instance,
    ///         binding the name to an id) has already happened in the past.
    ///         This is important in order to reliably retrieve ids from a
    ///         name, even if the name was already registered.
    ///
    HPX_API_EXPORT std::vector<hpx::future<hpx::id_type> >
        find_from_basename(std::string const& base_name,
            std::vector<std::size_t> const& ids);

    /// Return registered clients from the given base name and sequence numbers.
    ///
    /// This function locates the ids which were registered with the given
    /// base name and the given sequence numbers. It returns a list of futures
    /// representing those ids.
    ///
    /// \tparam Client     The client type to return
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param ids          [in] The sequence numbers of the registered ids.
    ///
    /// \returns A list of futures representing the ids which were registered
    ///          using the given base name and sequence numbers.
    ///
    /// \note   The futures embedded in the returned client objects will become
    ///         ready even if the event (for instance,
    ///         binding the name to an id) has already happened in the past.
    ///         This is important in order to reliably retrieve ids from a
    ///         name, even if the name was already registered.
    ///
    template <typename Client>
    std::vector<Client>
        find_from_basename(std::string const& base_name,
            std::vector<std::size_t> const& ids)
    {
        return components::make_clients<Client>(
            find_from_basename(base_name, ids));
    }

    /// \brief Return registered id from the given base name and sequence number.
    ///
    /// This function locates the id which was registered with the given
    /// base name and the given sequence number. It returns a future
    /// representing those id.
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param sequence_nr  [in] The sequence number of the registered id.
    ///
    /// \returns A representing the id which was registered using the given
    ///          base name and sequence numbers.
    ///
    /// \note   The future will become ready even if the event (for instance,
    ///         binding the name to an id) has already happened in the past.
    ///         This is important in order to reliably retrieve ids from a
    ///         name, even if the name was already registered.
    ///
    HPX_API_EXPORT hpx::future<hpx::id_type>
        find_from_basename(std::string const& base_name,
            std::size_t sequence_nr = ~0U);

    /// \brief Return registered id from the given base name and sequence number.
    ///
    /// This function locates the id which was registered with the given
    /// base name and the given sequence number. It returns a future
    /// representing those id.
    ///
    /// \tparam Client     The client type to return
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param sequence_nr  [in] The sequence number of the registered id.
    ///
    /// \returns A representing the id which was registered using the given
    ///          base name and sequence numbers.
    ///
    /// \note   The future embedded in the returned client object will become
    ///         ready even if the event (for instance,
    ///         binding the name to an id) has already happened in the past.
    ///         This is important in order to reliably retrieve ids from a
    ///         name, even if the name was already registered.
    ///
    template <typename Client>
    Client find_from_basename(std::string base_name,
        std::size_t sequence_nr = ~0U)
    {
        return components::make_client<Client>(
            find_from_basename(std::move(base_name), sequence_nr));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Register the given id using the given base name.
    ///
    /// The function registers the given ids using the provided base name.
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param id           [in] The id to register using the given base name.
    /// \param sequence_nr  [in, optional] The sequential number to use for the
    ///                     registration of the id. This number has to be
    ///                     unique system wide for each registration using the
    ///                     same base name. The default is the current locality
    ///                     identifier. Also, the sequence numbers have to be
    ///                     consecutive starting from zero.
    ///
    /// \returns A future representing the result of the registration operation
    ///          itself.
    ///
    /// \note    The operation will fail if the given sequence number is not
    ///          unique.
    ///
    HPX_API_EXPORT hpx::future<bool> register_with_basename(
        std::string const& base_name, hpx::id_type id,
        std::size_t sequence_nr = ~0U);

    /// Register the id wrapped in the given future using the given base name.
    ///
    /// The function registers the object the given future refers to using the
    /// provided base name.
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param f            [in] The future which should be registered using
    ///                     the given base name.
    /// \param sequence_nr  [in, optional] The sequential number to use for the
    ///                     registration of the id. This number has to be
    ///                     unique system wide for each registration using the
    ///                     same base name. The default is the current locality
    ///                     identifier. Also, the sequence numbers have to be
    ///                     consecutive starting from zero.
    ///
    /// \returns A future representing the result of the registration operation
    ///          itself.
    ///
    /// \note    The operation will fail if the given sequence number is not
    ///          unique.
    ///
    inline hpx::future<bool> register_with_basename(std::string const& base_name,
        hpx::future<hpx::id_type> f, std::size_t sequence_nr = ~0U)
    {
        return f.then(
            [=](hpx::future<hpx::id_type> && f) mutable
            {
                return register_with_basename(base_name, f.get(), sequence_nr);
            });
    }

    /// Register the id wrapped in the given client using the given base name.
    ///
    /// The function registers the object the given client refers to using the
    /// provided base name.
    ///
    /// \tparam Client      The client type to register
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param client       [in] The client which should be registered using
    ///                     the given base name.
    /// \param sequence_nr  [in, optional] The sequential number to use for the
    ///                     registration of the id. This number has to be
    ///                     unique system wide for each registration using the
    ///                     same base name. The default is the current locality
    ///                     identifier. Also, the sequence numbers have to be
    ///                     consecutive starting from zero.
    ///
    /// \returns A future representing the result of the registration operation
    ///          itself.
    ///
    /// \note    The operation will fail if the given sequence number is not
    ///          unique.
    ///
    template <typename Client, typename Stub>
    hpx::future<bool> register_with_basename(std::string const& base_name,
        components::client_base<Client, Stub>& client,
        std::size_t sequence_nr = ~0U)
    {
        return client.then(
            [=](components::client_base<Client, Stub> && c)
            {
                return register_with_basename(base_name, c.get_id(), sequence_nr);
            });
    }

    /// \brief Unregister the given id using the given base name.
    ///
    /// The function unregisters the given ids using the provided base name.
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param sequence_nr  [in, optional] The sequential number to use for the
    ///                     un-registration. This number has to be the same
    ///                     as has been used with \a register_with_basename
    ///                     before.
    ///
    /// \returns A future representing the result of the un-registration
    ///          operation itself.
    ///
    HPX_API_EXPORT hpx::future<hpx::id_type> unregister_with_basename(
        std::string const& base_name, std::size_t sequence_nr = ~0U);

    /// Unregister the given base name.
    ///
    /// The function unregisters the given ids using the provided base name.
    ///
    /// \tparam Client      The client type to return
    ///
    /// \param base_name    [in] The base name for which to retrieve the
    ///                     registered ids.
    /// \param sequence_nr  [in, optional] The sequential number to use for the
    ///                     un-registration. This number has to be the same
    ///                     as has been used with \a register_with_basename
    ///                     before.
    ///
    /// \returns A future representing the result of the un-registration
    ///          operation itself.
    ///
    template <typename Client>
    Client unregister_with_basename(
        std::string const& base_name, std::size_t sequence_nr = ~0U)
    {
        return components::make_client<Client>(
            unregister_with_basename(base_name, sequence_nr));
    }
}

#endif
