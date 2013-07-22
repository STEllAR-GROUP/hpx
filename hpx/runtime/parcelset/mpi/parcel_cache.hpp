//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_MPI_PARCELCACHE_HPP)
#define HPX_PARCELSET_MPI_PARCELCACHE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/mpi/sender.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <set>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace mpi { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct parcel_cache
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        typedef std::vector<boost::shared_ptr<parcel_buffer> > parcel_buffers;
        typedef hpx::lcos::local::spinlock mutex_type;

        parcel_cache(std::size_t num_threads, boost::uint64_t max_outbound_size)
          : parcel_maps_mtx_(num_threads)
          , parcel_buffers_(num_threads)
          , archive_flags_(boost::archive::no_header)
          , max_outbound_size_(max_outbound_size)
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

            for(std::size_t i = 0; i < num_threads; ++i)
            {
                parcel_maps_mtx_[i] = new mutex_type;
            }
        }

        ~parcel_cache()
        {
            for(std::size_t i = 0; i < parcel_maps_mtx_.size(); ++i)
            {
                delete parcel_maps_mtx_[i];
            }
        }

        void set_parcel(parcel const& p, write_handler_type const & f, std::size_t idx = -1)
        {
            set_parcel(std::vector<parcel>(1, p), std::vector<write_handler_type>(1, f), idx);
        }

        void set_parcel(std::vector<parcel> const& pv,
            std::vector<write_handler_type> handlers, std::size_t idx = -1)
        {
            // collect argument sizes from parcels
            std::size_t arg_size = 0;

            // we choose the highest priority of all parcels for this message
            threads::thread_priority priority = threads::thread_priority_default;

            // collect outgoing data
            boost::shared_ptr<parcel_buffer> buffer(boost::make_shared<parcel_buffer>());
            buffer->handlers_ = boost::move(handlers);

            // guard against serialization errors
            try {
                BOOST_FOREACH(parcel const & p, pv)
                {
                    arg_size += traits::get_type_size(p);
                    priority = (std::max)(p.get_thread_priority(), priority);
                }

                buffer->buffer_.reserve(arg_size*2);

                // mark start of serialization
                util::high_resolution_timer timer;

                {
                    // Serialize the data
                    HPX_STD_UNIQUE_PTR<util::binary_filter> filter(
                        pv[0].get_serialization_filter());

                    int archive_flags = archive_flags_;
                    if (filter.get() != 0) {
                        filter->set_max_length(buffer->buffer_.capacity());
                        archive_flags |= util::enable_compression;
                    }

                    util::portable_binary_oarchive archive(
                        buffer->buffer_, filter.get(), archive_flags);

                    std::size_t count = pv.size();
                    archive << count;

                    BOOST_FOREACH(parcel const& p, pv)
                    {
                        archive << p;
                    }

                    arg_size = archive.bytes_written();

                    // store the time required for serialization
                    buffer->send_data_.serialization_time_ = timer.elapsed_nanoseconds();
                }
            }
            catch (boost::archive::archive_exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem.
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization failed, caught "
                        "boost::archive::archive_exception: %s") % e.what()));
                return;
            }
            catch (boost::system::system_error const& e) {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization failed, caught "
                        "boost::system::system_error: %d (%s)") %
                            e.code().value() % e.code().message()));
                return;
            }
            catch (std::exception const& e) {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization failed, caught "
                        "std::exception: %s") % e.what()));
                return;
            }

            // make sure outgoing message is not larger than allowed
            if (buffer->buffer_.size() > max_outbound_size_)
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization created message larger "
                        "than allowed (created: %ld, allowed: %ld), consider"
                        "configuring larger hpx.parcel.max_message_size") %
                            buffer->buffer_.size() % max_outbound_size_));
                return;
            }

            buffer->send_data_.num_parcels_ = pv.size();
            buffer->send_data_.bytes_ = arg_size;
            buffer->send_data_.raw_bytes_ = buffer->buffer_.size();

            buffer->rank_ = pv[0].get_destination_locality().get_rank();
            if (buffer->rank_ == -1)
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    "can't send over MPI without a known destination rank");
                return;
            }

            std::size_t i = (idx == std::size_t(-1)) ? hpx::get_worker_thread_num() : idx;

            {
                mutex_type::scoped_lock lk(*parcel_maps_mtx_[i]);
                parcel_buffers_[i].push_back(buffer);
            }
        }

        std::list<boost::shared_ptr<sender> > get_senders(
            HPX_STD_FUNCTION<int()> const& tag_generator, MPI_Comm communicator,
            parcelport& pp)
        {
            std::list<boost::shared_ptr<sender> > res;
            for(std::size_t i = 0; i < parcel_buffers_.size(); ++i)
            {
                mutex_type::scoped_lock lk(*parcel_maps_mtx_[i]);
                BOOST_FOREACH(boost::shared_ptr<parcel_buffer> b, parcel_buffers_[i])
                {
                    boost::shared_ptr<sender> s(boost::make_shared<sender>(
                        header(
                            b->rank_
                          , tag_generator()
                          , static_cast<int>(b->buffer_.size())
                        )
                      , b
                      , communicator
                    ));
                    if(s->done(pp)) continue;
                    res.push_back(
                        s
                    );
                }
                parcel_buffers_[i].clear();
            }
            return boost::move(res);
        }

        std::vector<mutex_type*> parcel_maps_mtx_;
        std::vector<parcel_buffers> parcel_buffers_;

        // archive flags
        int archive_flags_;

        boost::uint64_t max_outbound_size_;
    };
}}}}

#endif
