//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_MPI_PARCELCACHE_HPP)
#define HPX_PARCELSET_MPI_PARCELCACHE_HPP

#include <hpx/hpx_fwd.hpp>
/*
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/io_service_pool.hpp>
*/
#include <hpx/runtime/parcelset/mpi/sender.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/format.hpp>

#include <set>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace mpi { namespace detail
{
    struct parcel_buffer
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;
        int rank;
        std::vector<char> buffer;
        std::vector<write_handler_type> handlers;
    };

    struct parcel_cache
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        typedef std::vector<parcel_buffer> parcel_buffers;
        typedef hpx::lcos::local::spinlock mutex_type;

        parcel_cache(std::size_t num_threads)
          : parcel_maps_mtx(num_threads)
          , parcel_buffers_(num_threads)
          , archive_flags_(boost::archive::no_header)
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
                parcel_maps_mtx[i] = new mutex_type;
            }
        }

        ~parcel_cache()
        {
            for(std::size_t i = 0; i < parcel_maps_mtx.size(); ++i)
            {
                delete parcel_maps_mtx[i];
            }
        }

        void set_parcel(parcel const& p, write_handler_type const & f, std::size_t idx = -1)
        {
            set_parcel(std::vector<parcel>(1, p), std::vector<write_handler_type>(1, f), idx);
        }

        void set_parcel(std::vector<parcel> const& pv, std::vector<write_handler_type> const & handlers, std::size_t idx = -1)
        {
            // collect argument sizes from parcels
            std::size_t arg_size = 0;

            // we choose the highest priority of all parcels for this message
            threads::thread_priority priority = threads::thread_priority_default;

            // guard against serialization errors
            std::vector<char> buffer;
            try {
                BOOST_FOREACH(parcel const & p, pv)
                {
                    arg_size += traits::get_type_size(p);
                    priority = (std::max)(p.get_thread_priority(), priority);
                }

                buffer.reserve(arg_size*2);

                // mark start of serialization
                util::high_resolution_timer timer;

                {
                    // Serialize the data
                    HPX_STD_UNIQUE_PTR<util::binary_filter> filter(
                        pv[0].get_serialization_filter());

                    int archive_flags = archive_flags_;
                    if (filter.get() != 0) {
                        filter->set_max_length(buffer.capacity());
                        archive_flags |= util::enable_compression;
                    }

                    util::portable_binary_oarchive archive(
                        buffer, filter.get(), boost::archive::no_header);

                    std::size_t count = pv.size();
                    archive << count;

                    BOOST_FOREACH(parcel const& p, pv)
                    {
                        archive << p;
                    }

                    arg_size = archive.bytes_written();

                    // store the time required for serialization
                    send_data_.serialization_time_ = timer.elapsed_nanoseconds();
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

            /*
            // make sure outgoing message is not larger than allowed
            if (out_buffer_.size() > max_outbound_size_)
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization created message larger "
                        "than allowed (created: %ld, allowed: %ld), consider"
                        "configuring larger hpx.parcel.max_message_size") %
                            out_buffer_.size() % max_outbound_size_));
                return;
            }
            */
            int dest_rank = pv[0].get_destination_locality().get_rank();
            std::size_t i = (idx == std::size_t(-1)) ? hpx::get_worker_thread_num() : idx;
            {
                mutex_type::scoped_lock lk(*parcel_maps_mtx[i]);
                parcel_buffer b = {dest_rank, buffer, handlers};
                parcel_buffers_[i].push_back(b);
            }
        }

        std::list<boost::shared_ptr<sender> > get_senders(HPX_STD_FUNCTION<int()> tag_generator, MPI_Comm communicator)
        {
            std::list<boost::shared_ptr<sender> > res;
            for(std::size_t i = 0; i < parcel_buffers_.size(); ++i)
            {
                mutex_type::scoped_lock lk(*parcel_maps_mtx[i]);
                BOOST_FOREACH(parcel_buffer & b, parcel_buffers_[i])
                {
                    int size = static_cast<int>(b.buffer.size());
                    boost::shared_ptr<sender> s(new sender(
                        header(
                            b.rank
                          , tag_generator()
                          , size
                        )
                      , boost::move(b.buffer)
                      , boost::move(b.handlers)
                      , communicator
                    ));
                    if(s->done()) continue;
                    res.push_back(
                        s
                    );
                }
                parcel_buffers_[i].clear();
            }
            return res;
        }

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point send_data_;

        std::vector<mutex_type*> parcel_maps_mtx;
        std::vector<parcel_buffers> parcel_buffers_;

        // archive flags
        int archive_flags_;
    };
}}}}

#endif
