//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/tcp/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/traits/type_size.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/signed_type.hpp>
#include <hpx/components/security/hash.hpp>
#endif

#include <stdexcept>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>
#include <boost/format.hpp>

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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace tcp
{
    parcelport_connection::parcelport_connection(boost::asio::io_service& io_service,
            naming::locality const& locality_id,
            performance_counters::parcels::gatherer& parcels_sent,
            boost::uint64_t max_outbound_size)
      : socket_(io_service), out_priority_(0), out_size_(0), out_data_size_(0)
      , max_outbound_size_(max_outbound_size)
#if defined(HPX_HAVE_SECURITY)
      , first_message_(true)
#endif
      , there_(locality_id), parcels_sent_(parcels_sent)
      , archive_flags_(boost::archive::no_header)
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
      , state_(state_initialized)
#endif
    {
#ifdef BOOST_BIG_ENDIAN
        std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
        std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
        if (endian_out == "little")
            archive_flags_ |= util::endian_little;
        else if (endian_out == "big")
            archive_flags_ |= util::endian_big;
        else {
            BOOST_ASSERT(endian_out =="little" || endian_out == "big");
        }

        std::string array_optimization =
            get_config_entry("hpx.parcel.array_optimization", "1");
        if (boost::lexical_cast<int>(array_optimization) == 0)
            archive_flags_ |= util::disable_array_optimization;
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_SECURITY)
    template <typename Archive>
    void parcelport_connection::serialize_certificate(Archive& archive,
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
            BOOST_ASSERT(hpx::is_starting());
            this_locality_id = locality_id;
        }

        bool has_certificate = false;
        if ((first_message_ || locality_id != this_locality_id) &&
            localities.find(locality_id) == localities.end())
        {
            // the first message must originate from this locality
            BOOST_ASSERT(!first_message_ || locality_id == this_locality_id);

            components::security::signed_certificate const& certificate =
                hpx::get_locality_certificate(locality_id, ec);

            if (!ec) {
                has_certificate = true;
                if (locality_id == this_locality_id)
                    first_message_ = false;
                archive << has_certificate << certificate;

                // keep track of all certificates already prepended for this message
                localities.insert(locality_id);
            }
            else {
                // if the certificate is not available we have to still be on
                // the 'first' message (it's too early for a certificate)
                BOOST_ASSERT(first_message_);
                archive << has_certificate;
            }
        }
        else {
            archive << has_certificate;
        }
    }

    void parcelport_connection::create_message_suffix(
        naming::gid_type const& parcel_id)
    {
        // mark start of security work
        util::high_resolution_timer timer_sec;

        // calculate hash of overall message
        components::security::hash hash(
            reinterpret_cast<unsigned char const*>(&out_buffer_.front()),
            out_buffer_.size());

        using components::security::parcel_suffix;
        using components::security::signed_parcel_suffix;

        signed_parcel_suffix suffix;
        sign_parcel_suffix(
            parcel_suffix(get_locality_id(), parcel_id, hash),
            suffix);

        // append the signed parcel suffix to the message
        out_buffer_.reserve(out_buffer_.size() + signed_parcel_suffix::size());

        std::copy(suffix.begin(), suffix.end(), std::back_inserter(out_buffer_));

        // store the time required for security
        send_data_.security_time_ = timer_sec.elapsed_nanoseconds();
    }
#endif

    void parcelport_connection::set_parcel(std::vector<parcel> const& pv)
    {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        state_ = state_set_parcel;
#endif

#if defined(HPX_DEBUG)
        // make sure that all parcels go to the same locality
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            BOOST_ASSERT(locality_id == destination());
        }
#endif
        // we choose the highest priority of all parcels for this message
        threads::thread_priority priority = threads::thread_priority_default;

        // collect argument sizes from parcels
        std::size_t arg_size = 0;

        // guard against serialization errors
        try {
            // clear and preallocate out_buffer_
            out_buffer_.clear();

            BOOST_FOREACH(parcel const & p, pv)
            {
                arg_size += traits::get_type_size(p);
                priority = (std::max)(p.get_thread_priority(), priority);
            }

            out_buffer_.reserve(arg_size*2);

            // mark start of serialization
            util::high_resolution_timer timer;

            {
                // Serialize the data
                HPX_STD_UNIQUE_PTR<util::binary_filter> filter(
                    pv[0].get_serialization_filter());

                int archive_flags = archive_flags_;
                if (filter.get() != 0) {
                    filter->set_max_length(out_buffer_.capacity());
                    archive_flags |= util::enable_compression;
                }

                util::portable_binary_oarchive archive(
                    out_buffer_, filter.get(), archive_flags);

#if defined(HPX_HAVE_SECURITY)
                std::set<boost::uint32_t> localities;
#endif
                std::size_t count = pv.size();
                archive << count; //-V128

                BOOST_FOREACH(parcel const& p, pv)
                {
#if defined(HPX_HAVE_SECURITY)
                    serialize_certificate(archive, localities, p);
#endif
                    archive << p;
                }

                arg_size = archive.bytes_written();
            }

#if defined(HPX_HAVE_SECURITY)
            // calculate and sign the hash, but only after everything has
            // been initialized
            if (!first_message_)
                create_message_suffix(pv[0].get_parcel_id());
#endif
            // store the time required for serialization
            send_data_.serialization_time_ = timer.elapsed_nanoseconds();
        }
        catch (boost::archive::archive_exception const& e) {
            // We have to repackage all exceptions thrown by the
            // serialization library as otherwise we will loose the
            // e.what() description of the problem.
            HPX_THROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::archive::archive_exception: %s") % e.what()));
            return;
        }
        catch (boost::system::system_error const& e) {
            HPX_THROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::system::system_error: %d (%s)") %
                        e.code().value() % e.code().message()));
            return;
        }
        catch (std::exception const& e) {
            HPX_THROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "std::exception: %s") % e.what()));
            return;
        }

        // make sure outgoing message is not larger than allowed
        if (out_buffer_.size() > max_outbound_size_)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization created message larger "
                    "than allowed (created: %ld, allowed: %ld), consider"
                    "configuring larger hpx.parcel.max_message_size") %
                        out_buffer_.size() % max_outbound_size_));
            return;
        }

        out_priority_ = boost::integer::ulittle8_t(priority);
        out_size_ = out_buffer_.size();
        out_data_size_ = arg_size;

        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = arg_size;
        send_data_.raw_bytes_ = out_buffer_.size();
    }
}}}

