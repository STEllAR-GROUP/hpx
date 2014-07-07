//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011-2014 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_ENCODE_PARCELS_HPP
#define HPX_PARCELSET_ENCODE_PARCELS_HPP

#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/signed_type.hpp>
#include <hpx/components/security/hash.hpp>
#endif

#include <boost/integer/endian.hpp>

#if defined(HPX_HAVE_SECURITY)
namespace hpx
{
    HPX_API_EXPORT bool is_starting();

    /// \brief Sign the given parcel-suffix
    ///
    /// \param suffix         The parcel suffoix to be signed
    /// \param signed_suffix  The signed parcel suffix will be placed here
    ///
    HPX_API_EXPORT void sign_parcel_suffix(
        components::security::parcel_suffix const& suffix,
        components::security::signed_parcel_suffix& signed_suffix,
        error_code& ec = throws);
}
#endif

namespace hpx { namespace parcelset
{
    template <typename Connection>
    boost::shared_ptr<parcel_buffer<typename Connection::buffer_type> >
    encode_parcels(std::vector<parcel> const &, Connection & connection, int archive_flags_);

    template <typename Connection>
    boost::shared_ptr<parcel_buffer<typename Connection::buffer_type> >
    encode_parcels(parcel const &p, Connection & connection, int archive_flags_)
    {
        return encode_parcels(std::vector<parcel>(1, p), connection, archive_flags_);
    }

#if defined(HPX_HAVE_SECURITY)
    template <typename Archive, typename Connection>
    void serialize_certificate(Archive& archive, Connection & connection,
        std::set<boost::uint32_t>& localities, parcel const& p)
    {
        // We send the certificate corresponding to the originating locality
        // of the parcel if this is the first message over this connection
        // or if the originating locality is not the current one.
        boost::uint32_t locality_id =
            naming::get_locality_id_from_gid(p.get_parcel_id());
        error_code ec(lightweight);
        boost::uint32_t this_locality_id = get_locality_id(ec);
        if (ec) {
            // this should only happen during bootstrap
            HPX_ASSERT(hpx::is_starting());
            this_locality_id = locality_id;
        }

        bool has_certificate = false;
        if ((connection.first_message_ || locality_id != this_locality_id) &&
            localities.find(locality_id) == localities.end())
        {
            // the first message must originate from this locality
            HPX_ASSERT(!connection.first_message_ || locality_id == this_locality_id);

            components::security::signed_certificate const& certificate =
                hpx::get_locality_certificate(locality_id, ec);

            if (!ec) {
                has_certificate = true;
                if (locality_id == this_locality_id)
                    connection.first_message_ = false;
                archive << has_certificate << certificate;

                // keep track of all certificates already prepended for this message
                localities.insert(locality_id);
            }
            else {
                // if the certificate is not available we have to still be on
                // the 'first' message (it's too early for a certificate)
                HPX_ASSERT(connection.first_message_);
                archive << has_certificate;
            }
        }
        else {
            archive << has_certificate;
        }
    }

    template <typename ParcelBuffer>
    void create_message_suffix(ParcelBuffer & buffer,
        naming::gid_type const& parcel_id)
    {
        // mark start of security work
        util::high_resolution_timer timer_sec;

        // calculate hash of overall message
        components::security::hash hash(
            reinterpret_cast<unsigned char const*>(&buffer.data_.front()),
            buffer.data_.size());

        using components::security::parcel_suffix;
        using components::security::signed_parcel_suffix;

        signed_parcel_suffix suffix;
        sign_parcel_suffix(
            parcel_suffix(get_locality_id(), parcel_id, hash),
            suffix);

        // append the signed parcel suffix to the message
        buffer.data_.reserve(buffer.data_.size() + signed_parcel_suffix::size());

        std::copy(suffix.begin(), suffix.end(), std::back_inserter(buffer.data_));

        // store the time required for security
        buffer.data_point_.security_time_ = timer_sec.elapsed_nanoseconds();
    }
#endif

    template <typename Connection>
    boost::shared_ptr<parcel_buffer<typename Connection::buffer_type> >
    encode_parcels(std::vector<parcel> const & pv, Connection & connection,
        int archive_flags_, bool enable_security)
    {
        typedef parcel_buffer<typename Connection::buffer_type> parcel_buffer_type;

#if defined(HPX_DEBUG)
        // make sure that all parcels go to the same locality
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            HPX_ASSERT(locality_id == connection.destination());
        }
#endif
        // collect argument sizes from parcels
        std::size_t arg_size = 0;
        boost::uint32_t dest_locality_id = pv[0].get_destination_locality_id();

        boost::shared_ptr<parcel_buffer_type> buffer;

        // guard against serialization errors
        try {
            try {
                // preallocate data_
                BOOST_FOREACH(parcel const & p, pv)
                {
                    arg_size += traits::get_type_size(p);
                }

                buffer = connection.get_buffer(pv[0], arg_size);
                buffer->clear();

                // mark start of serialization
                util::high_resolution_timer timer;

                {
                    // Serialize the data
                    HPX_STD_UNIQUE_PTR<util::binary_filter> filter(
                        pv[0].get_serialization_filter());

                    int archive_flags = archive_flags_;
                    if (filter.get() != 0) {
                        filter->set_max_length(buffer->data_.capacity());
                        archive_flags |= util::enable_compression;
                    }

                    util::portable_binary_oarchive archive(
                        buffer->data_
                      , &buffer->chunks_
                      , dest_locality_id
                      , filter.get()
                      , archive_flags);

#if defined(HPX_HAVE_SECURITY)
                    std::set<boost::uint32_t> localities;
#endif
                    std::size_t count = pv.size();
                    archive << count; //-V128

                    BOOST_FOREACH(parcel const& p, pv)
                    {
#if defined(HPX_HAVE_SECURITY)
                        if (enable_security)
                            serialize_certificate(archive, connection, localities, p);
#endif
                        archive << p;
                    }

                    arg_size = archive.bytes_written();
                }

#if defined(HPX_HAVE_SECURITY)
                // calculate and sign the hash, but only after everything has
                // been initialized
                if (enable_security && !connection.first_message_)
                    create_message_suffix(*buffer, pv[0].get_parcel_id());
#endif
                // store the time required for serialization
                buffer->data_point_.serialization_time_ = timer.elapsed_nanoseconds();
            }
            catch (hpx::exception const& e) {
                LPT_(fatal)
                    << "encode_parcels: "
                       "caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
                return buffer;
            }
            catch (boost::system::system_error const& e) {
                LPT_(fatal)
                    << "encode_parcels: "
                       "caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
                return buffer;
            }
            catch (boost::exception const&) {
                LPT_(fatal)
                    << "encode_parcels: "
                       "caught boost::exception";
                hpx::report_error(boost::current_exception());
                return buffer;
            }
            catch (std::exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem, due to slicing.
                boost::throw_exception(boost::enable_error_info(
                    hpx::exception(serialization_error, e.what())));
            }
        }
        catch (...) {
            LPT_(fatal)
                    << "encode_parcels: "
                   "caught unknown exception";
            hpx::report_error(boost::current_exception());
            return buffer;
        }

        buffer->size_ = buffer->data_.size();
        buffer->data_size_ = arg_size;

        performance_counters::parcels::data_point& data = buffer->data_point_;
        data.num_parcels_ = pv.size();
        data.bytes_ = arg_size;
        data.raw_bytes_ = buffer->data_.size();

        // prepare chunk data for transmission, the transmission_chunks data
        // first holds all zero-copy, then all non-zero-copy chunk infos
        typedef typename parcel_buffer_type::transmission_chunk_type
            transmission_chunk_type;
        typedef typename parcel_buffer_type::count_chunks_type
            count_chunks_type;

        std::vector<transmission_chunk_type>& chunks =
            buffer->transmission_chunks_;

        chunks.clear();
        chunks.reserve(buffer->chunks_.size());

        std::size_t index = 0;
        BOOST_FOREACH(util::serialization_chunk& c, buffer->chunks_)
        {
            if (c.type_ == util::chunk_type_pointer) {
                chunks.push_back(transmission_chunk_type(index, c.size_));
            }
            ++index;
        }

        buffer->num_chunks_ = count_chunks_type(
                static_cast<boost::uint32_t>(chunks.size()),
                static_cast<boost::uint32_t>(buffer->chunks_.size() - chunks.size())
            );

        if (!chunks.empty()) {
            // the remaining number of chunks are non-zero-copy
            BOOST_FOREACH(util::serialization_chunk& c, buffer->chunks_)
            {
                if (c.type_ == util::chunk_type_index) {
                    chunks.push_back(
                        transmission_chunk_type(c.data_.index_, c.size_));
                }
            }
        }

        return buffer;
    }
}}

#endif
