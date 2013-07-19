//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/mpi/parcelport.hpp>
#include <hpx/runtime/parcelset/detail/call_for_each.hpp>
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/logging.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/hash.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/signed_type.hpp>
#endif

#include <boost/version.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

namespace hpx
{
    /// \brief Verify the certificate in the given byte sequence
    ///
    /// \param data      The full received message buffer, assuming that it
    ///                  has a parcel_suffix appended.
    /// \param parcel_id The parcel id of the first parcel in side the message
    ///
    HPX_API_EXPORT bool verify_parcel_suffix(std::vector<char> const& data,
        naming::gid_type& parcel_id, error_code& ec = throws);

}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace mpi
{
    void decode_message(
        std::size_t num_parcel_chunks,
        std::vector<char> const & parcel_data,
        parcelport& pp);

    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : parcelset::parcelport(ini),
        io_service_pool_(1, on_start_thread, on_stop_thread, "parcel_pool_mpi", "-mpi"),
        parcel_cache_(ini.get_os_thread_count()),
        stopped(false),
        next_tag(2)
        /*
        acceptor_(NULL),
        connection_cache_(ini.get_max_connections(), ini.get_max_connections_per_loc())
        */
    {
        if (here_.get_type() != connection_mpi) {
            HPX_THROW_EXCEPTION(network_error, "mpi::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
    }

    parcelport::~parcelport()
    {
        std::cout << util::mpi_environment::rank() << " parcelport::~parcelport()\n";
    }

    util::io_service_pool* parcelport::get_thread_pool(char const* name)
    {
        if (0 == std::strcmp(name, io_service_pool_.get_name()))
            return &io_service_pool_;
        return 0;
    }

    bool parcelport::run(bool blocking)
    {
        //std::cout << util::mpi_environment::rank() <<  " running MPI parcelport\n";
        io_service_pool_.run(false);    // start pool

        boost::asio::io_service& io_service = io_service_pool_.get_io_service();

        io_service.post(HPX_STD_BIND(&parcelport::handle_messages, this->shared_from_this()));
        
        if (blocking)
        {
            std::cout << util::mpi_environment::rank() <<  " running MPI parcelport blocked\n";
            io_service_pool_.join();
        }

        return true;
    }

    void parcelport::stop(bool blocking)
    {
        // make sure no more work is pending, wait for service pool to get empty
        std::cout << util::mpi_environment::rank() << " parcelport::stop() start\n";
        io_service_pool_.stop();
        stopped = true;
        if (blocking) {
            io_service_pool_.join();

            io_service_pool_.clear();
        }
        //MPI_Barrier(util::mpi_environment::communicator());
        std::cout << util::mpi_environment::rank() << " parcelport::stop() finished\n";
    }

    void parcelport::handle_messages()
    {
        MPI_Comm communicator = util::mpi_environment::communicator();
        MPI_Request recv_header_request;
        boost::int64_t recv_header[] = {0, 0, 0};
        boost::int64_t send_header[] = {0, 0, 0};

        std::vector<MPI_Request> recv_requests;

        typedef 
            std::map<
                std::pair<
                    int // Source
                  , int // Tag
                >
              , std::pair<
                    std::size_t // Number of parcels
                  , std::vector<char> // Data
                >
            >
            recv_buffers_type;
        recv_buffers_type recv_buffers;


        std::vector<MPI_Request> send_header_requests;
        std::vector<int> send_header_tags;
        std::vector<MPI_Request> send_data_requests;
        std::vector<int> send_data_tags;
        typedef 
            std::map<
                int // Tag
              , HPX_STD_TUPLE<
                    int // Destination
                  , std::size_t // Number of parcels
                  , std::vector<char> // Data
                  , std::vector<write_handler_type> // Write Handlers
                >
            >
            send_buffers_type;
        send_buffers_type send_buffers;

        MPI_Irecv(recv_header, 3, MPI_INT64_T, MPI_ANY_SOURCE, 0, communicator, &recv_header_request);
        //std::cout << util::mpi_environment::rank() <<  " handle messages!\n";
        bool run = !stopped;
        while(run)
        {
            MPI_Status status;
            int completed;
            MPI_Test(&recv_header_request, &completed, &status);
            if(completed)
            {
                /*
                std::cout << util::mpi_environment::rank() <<  " received a message from " << recv_header_status.MPI_SOURCE << "!\n";
                std::cout << "Tag: " << recv_header[0] << " Number of parcels: " << recv_header[1] << " Size of message : " << recv_header[2] << "\n";
                */

                std::pair<int, int> recv_key(status.MPI_SOURCE, static_cast<int>(recv_header[0]));
                recv_buffers_type::iterator it = recv_buffers.find(recv_key);
                if(it == recv_buffers.end())
                {
                    it =
                        recv_buffers.insert(
                            it
                          , std::make_pair(
                                recv_key
                              , std::make_pair(
                                    recv_header[1]
                                  , std::vector<char>(recv_header[2])
                                )
                            )
                        );
                }
                else
                {
                    it->second.first = recv_header[1];
                    it->second.second.resize(recv_header[2]);
                }

                recv_requests.push_back(MPI_Request());
                MPI_Irecv(
                    &it->second.second[0],
                    static_cast<int>(recv_header[2]),
                    MPI_CHAR,
                    status.MPI_SOURCE,
                    static_cast<int>(recv_header[0]),
                    communicator,
                    &recv_requests.back()
                );
                MPI_Irecv(recv_header, 3, MPI_INT64_T, MPI_ANY_SOURCE, 0, communicator, &recv_header_request);
            }
            
            if(!recv_requests.empty())
            {
                int index;
                MPI_Testany(static_cast<int>(recv_requests.size()), &recv_requests[0], &index, &completed, &status);
                if(completed)
                {
                    //BOOST_ASSERT(status.MPI_SOURCE == requests.first);
                    //std::cout << "received full message from " << status.MPI_SOURCE << "\n";
                    std::pair<int, int> recv_key(status.MPI_SOURCE, status.MPI_TAG);
                    recv_buffers_type::iterator it = recv_buffers.find(recv_key);
                    BOOST_ASSERT(it != recv_buffers.end());
                    decode_message(it->second.first, it->second.second, *this);
                    it->second.first = 0;
                    it->second.second.clear();
                    recv_requests.erase(recv_requests.begin() + index);
                }
            }

            detail::parcel_cache::parcels_map parcels_map = parcel_cache_.get_parcels();
            if(!parcels_map.empty())
            {
                //std::cout << util::mpi_environment::rank() <<  " got " << parcels_map.size() << " parcel data to send!\n";
                BOOST_FOREACH(detail::parcel_cache::parcels_map::value_type const & parcels, parcels_map)
                {
                    //std::cout << util::mpi_environment::rank() <<  " sending header to " << parcels.first << "!\n";
                    int tag = get_next_tag();
                    send_header[0] = tag;
                    send_header[1] = parcels.second.get<0>();
                    send_header[2] = parcels.second.get<1>().size();
                    
                    send_header_requests.push_back(MPI_Request());
                    MPI_Isend(send_header, 3, MPI_INT64_T, parcels.first, 0, communicator, &send_header_requests.back());
                    
                    send_header_tags.push_back(tag);
                    send_buffers_type::iterator it = send_buffers.find(tag);
                    if(it == send_buffers.end())
                    {
                        it
                          = send_buffers.insert(
                                it
                              , std::make_pair(
                                    tag
                                  , hpx::util::make_tuple(
                                        parcels.first
                                      , parcels.second.get<0>()
                                      , boost::move(parcels.second.get<1>())
                                      , boost::move(parcels.second.get<2>())
                                    )
                                )
                            );
                    }
                    else
                    {
                        it->second.get<0>() = parcels.first;
                        it->second.get<1>() = parcels.second.get<0>();
                        it->second.get<2>() = boost::move(parcels.second.get<1>());
                        it->second.get<3>() = boost::move(parcels.second.get<2>());
                    }
                }
            }
            
            if(!send_header_requests.empty())
            {
                int index;
                MPI_Testany(static_cast<int>(send_header_requests.size()), &send_header_requests[0], &index, &completed, &status);
                if(completed)
                {
                    //std::cout << util::mpi_environment::rank() <<  " sent header with tag " << send_header_tags[index] << "!\n";
                    send_data_requests.push_back(MPI_Request());
                    send_buffers_type::iterator it = send_buffers.find(send_header_tags[index]);
                    BOOST_ASSERT(it != send_buffers.end());
                    MPI_Isend(
                        &it->second.get<2>()[0]
                      , static_cast<int>(it->second.get<2>().size())
                      , MPI_CHAR
                      , it->second.get<0>()
                      , send_header_tags[index]
                      , communicator
                      , &send_data_requests.back()
                    );
                    send_data_tags.push_back(send_header_tags[index]);
                    send_header_requests.erase(send_header_requests.begin() + index);
                    send_header_tags.erase(send_header_tags.begin() + index);
                }
            }
            if(!send_data_requests.empty())
            {
                int index;
                MPI_Testany(static_cast<int>(send_data_requests.size()), &send_data_requests[0], &index, &completed, &status);
                if(completed)
                {
                    //std::cout << util::mpi_environment::rank() <<  " sent data with tag " << send_header_tags[index] << "!\n";
                    send_buffers_type::iterator it = send_buffers.find(send_data_tags[index]);
                    BOOST_ASSERT(it != send_buffers.end());
                    error_code ec;
                    BOOST_FOREACH(write_handler_type & f, it->second.get<3>())
                    {
                        if(f)
                        {
                            f(ec, it->second.get<2>().size());
                        }
                    }
                    free_tags.push_back(send_header_tags[index]);
                    send_data_requests.erase(send_data_requests.begin() + index);
                    send_data_tags.erase(send_data_tags.begin() + index);
                }
            }

            if(stopped)
            {
                run = !(recv_requests.empty() && send_data_requests.empty() && send_header_requests.empty());
            }
            else
            {
                run = true;
            }
        }
        // cancel all remaining requests
        MPI_Cancel(&recv_header_request);
        BOOST_ASSERT(recv_requests.empty());
        BOOST_ASSERT(send_header_requests.empty());
        BOOST_ASSERT(send_data_requests.empty());
        std::cout << util::mpi_environment::rank() << " handle_messages stopped\n";
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::put_parcels(std::vector<parcel> const & parcels,
        std::vector<write_handler_type> const& handlers)
    {
        /*
        std::cout << util::mpi_environment::rank() <<  " MPI parcelport: put parcels\n";
        std::cout << util::mpi_environment::rank() << " destination: " << parcels[0].get_destination_locality() << "\n";
        */

        if (parcels.size() != handlers.size())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                "mismatched number of parcels and handlers");
            return;
        }

        naming::locality locality_id = parcels[0].get_destination_locality();

#if defined(HPX_DEBUG)
        // make sure all parcels go to the same locality
        for (std::size_t i = 1; i != parcels.size(); ++i)
        {
            BOOST_ASSERT(locality_id == parcels[i].get_destination_locality());
        }
#endif
        parcel_cache_.set_parcel(parcels, handlers);

    }

    void parcelport::put_parcel(parcel const& p, write_handler_type f)
    {
        /*
        std::cout << util::mpi_environment::rank() <<  " MPI parcelport: put parcel\n";
        std::cout << util::mpi_environment::rank() << " destination: " << p.get_destination_locality() << "\n";
        */
        parcel_cache_.set_parcel(p, f);
    }

    void parcelport::send_early_parcel(parcel& p)
    {
        /*
        std::cout << util::mpi_environment::rank() << " MPI parcelport: send early parcel\n";
        std::cout << util::mpi_environment::rank() << " destination: " << p.get_destination_locality() << "\n";
        */
        naming::locality const& l = p.get_destination_locality();
        error_code ec;

        parcel_cache_.set_parcel(p, write_handler_type(), 0);
    }

    struct mpi_parcel_data
    {
        const char * data_ptr;
        std::size_t size_;

        std::size_t size() const
        {
            return size_;
        }
        const char * data() const
        {
            return data_ptr;
        }
    };

    void decode_message(
        std::size_t num_parcel_chunks,
        std::vector<char> const & parcel_data,
        parcelport& pp
        //performance_counters::parcels::data_point receive_data,
    )
    {
        // protect from un-handled exceptions bubbling up
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                boost::int64_t overall_add_parcel_time = 0;

                mpi_parcel_data data = {&parcel_data[0], parcel_data.size()};
                std::size_t bytes_decoded = 0;
                for(std::size_t chunk = 0; chunk < num_parcel_chunks; ++chunk)
                {
                    // De-serialize the parcel data
                    util::portable_binary_iarchive archive(data,
                        parcel_data.size(), boost::archive::no_header);

                    std::size_t parcel_count = 0;
                    archive >> parcel_count;
                    for(std::size_t i = 0; i != parcel_count; ++i)
                    {
                        // de-serialize parcel and add it to incoming parcel queue
                        parcel p;
                        archive >> p;
                        // make sure this parcel ended up on the right locality
                        BOOST_ASSERT(p.get_destination_locality() == pp.here());

                        // be sure not to measure add_parcel as serialization time
                        boost::int64_t add_parcel_time = timer.elapsed_nanoseconds();

                        pp.add_received_parcel(p);
                        overall_add_parcel_time += timer.elapsed_nanoseconds() -
                            add_parcel_time;
                    }

                    // complete received data with parcel count
                    /*
                    receive_data.num_parcels_ = parcel_count;
                    receive_data.raw_bytes_ = archive.bytes_read();
                    */
                    std::size_t bytes_read = archive.bytes_read();
                    data.data_ptr += bytes_read;
                    data.size_ -= bytes_read;
                    bytes_decoded += bytes_read;
                }

                // store the time required for serialization
                /*
                receive_data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;
                */

                //pp.add_received_data(receive_data);
            }
            catch (hpx::exception const& e) {
                LPT_(error)
                    << "decode_message(mpi): caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(error)
                    << "decode_message(mpi): caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(error)
                    << "decode_message(mpi): caught boost::exception.";
                hpx::report_error(boost::current_exception());
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
            LPT_(error)
                << "decode_message(mpi): caught unknown exception.";
            hpx::report_error(boost::current_exception());
        }
    }
}}}
