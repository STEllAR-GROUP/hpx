//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/exception_list.hpp>
#include <hpx/plugins/parcelport/ibverbs/connection_handler.hpp>
#include <hpx/plugins/parcelport/ibverbs/acceptor.hpp>
#include <hpx/plugins/parcelport/ibverbs/sender.hpp>
#include <hpx/plugins/parcelport/ibverbs/receiver.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/asio_util.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/atomic.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/lexical_cast.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <utility>

#if (defined(__linux) || defined(linux) || defined(__linux__))
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    std::string get_ibverbs_address(util::runtime_configuration const & ini)
    {
        if(ini.has_section("hpx.parcel.ibverbs"))
        {
            util::section const * sec = ini.get_section("hpx.parcel.ibverbs");
            if(nullptr != sec) {
                std::string ibverbs_enabled(
                    sec->get_entry("enable", "0"));
                if(boost::lexical_cast<int>(ibverbs_enabled))
                {
#if (defined(__linux) || defined(linux) || defined(__linux__))
                    std::string ibverbs_ifname(
                        sec->get_entry("ifname", HPX_HAVE_PARCELPORT_IBVERBS_IFNAME));

                    ifaddrs *ifap;
                    getifaddrs(&ifap);
                    for(ifaddrs *cur = ifap; cur != nullptr; cur = cur->ifa_next)
                    {
                        if(std::string(cur->ifa_name) == ibverbs_ifname)
                        {
                            char buf[1024] = {0};
                            switch(cur->ifa_addr->sa_family)
                            {
                                case AF_INET:
                                    {
                                        inet_ntop(
                                            cur->ifa_addr->sa_family
                                          , &((sockaddr_in *)cur->ifa_addr)->sin_addr
                                          , buf
                                          , 1024
                                        );
                                        freeifaddrs(ifap);
                                        return buf;
                                    }
                                case AF_INET6:
                                    {
                                        inet_ntop(
                                            cur->ifa_addr->sa_family
                                          , &((sockaddr_in6 *)cur->ifa_addr)->sin6_addr
                                          , buf
                                          , 1024
                                        );
                                        freeifaddrs(ifap);
                                        return buf;
                                    }
                                default:
                                    break;
                            }
                        }
                    }
                    freeifaddrs(ifap);
#endif
                }
            }
        }
        return "";
    }

    parcelset::locality parcelport_address(util::runtime_configuration const & ini)
    {
        // load all components as described in the configuration information
        std::string ibverbs_address = get_ibverbs_address(ini);
        if (ini.has_section("hpx.parcel")) {
            util::section const* sec = ini.get_section("hpx.parcel");
            if (nullptr != sec) {
                return parcelset::locality(
                    locality(
                        ibverbs_address
                      , hpx::util::get_entry_as<boost::uint16_t>(
                            *sec, "port", HPX_INITIAL_IP_PORT)
                    )
                );
            }
        }
        return
            parcelset::locality(
                locality(
                    ibverbs_address
                  , HPX_INITIAL_IP_PORT
                )
            );
    }

    std::size_t connection_handler
        ::memory_chunk_size(util::runtime_configuration const& ini)
    {
        return hpx::util::get_entry_as<std::size_t>(
            ini, "hpx.parcel.ibverbs.memory_chunk_size",
            HPX_HAVE_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE);
    }

    std::size_t connection_handler
        ::max_memory_chunks(util::runtime_configuration const& ini)
    {
        return hpx::util::get_entry_as<std::size_t>(
            ini, "hpx.parcel.ibverbs.max_memory_chunks",
            HPX_HAVE_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS);
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread)
      : base_type(ini, parcelport_address(ini), on_start_thread, on_stop_thread)
      , memory_pool_(memory_chunk_size(ini), max_memory_chunks(ini))
      , mr_cache_size_(max_memory_chunks(ini) * 4) // <-- FIXME: Find better value here
      , stopped_(false)
      , handling_messages_(false)
      , handling_accepts_(false)
      , use_io_pool_(true)
    {
        if (here_.type() != std::string("ibverbs")) {
            HPX_THROW_EXCEPTION(network_error, "ibverbs::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + std::string(here_.type()));
        }

        // we never do zero copy optimization for this parcelport
        allow_zero_copy_optimizations_ = false;

        if(hpx::util::get_entry_as<int>(ini, "hpx.parcel.ibverbs.use_io_pool", "1") == 0)
        {
            use_io_pool_ = false;
        }

        device_list_ = 0;
        int num_devices = 0;
        device_list_ = ibv_get_device_list(&num_devices);
        for(int i = 0; i < num_devices; ++i)
        {
            ibv_context *ctx = ibv_open_device(device_list_[i]);
            get_pd(ctx, boost::system::throws);
            context_list_.push_back(ctx);
        }
    }

    connection_handler::~connection_handler()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
        for (pd_map_type::value_type& pd_pair : pd_map_)
        {
            ibv_dealloc_pd(pd_pair.second);
        }
        for (ibv_context* ctx : context_list_)
        {
            ibv_close_device(ctx);
        }
        ibv_free_device_list(device_list_);

    }

    bool connection_handler::do_run()
    {
        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        util::endpoint_iterator_type end = util::accept_end();
        for (util::endpoint_iterator_type it =
                util::accept_begin(here_.get<locality>(),
                    io_service_pool_.get_io_service(0));
             it != end; ++it, ++tried)
        {
            try {
                boost::asio::ip::tcp::endpoint ep = *it;

                acceptor_.bind(ep, boost::system::throws);
                ++tried;
                break;
            }
            catch (boost::system::system_error const&) {
                errors.add(boost::current_exception());
                continue;
            }
        }

        if (errors.size() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error,
                "ibverbs::connection_handler::run", errors.get_message());
            return false;
        }

        handling_accepts_ = true;
        boost::asio::io_service& io_service = io_service_pool_.get_io_service(1);
        io_service.post(util::bind(&connection_handler::handle_accepts, this));

        background_work();
        return true;
    }

    void connection_handler::do_stop()
    {
        // Mark stopped state
        stopped_ = true;
        // Wait until message handler returns
        std::size_t k = 0;
        while(handling_messages_)
        {
            hpx::lcos::local::spinlock::yield(k);
            ++k;
        }
        k = 0;
        while(handling_accepts_)
        {
            hpx::lcos::local::spinlock::yield(k);
            ++k;
        }

        // cancel all pending accept operations
        boost::system::error_code ec;
        acceptor_.close(ec);
    }

    // Make sure all pending requests are handled
    void connection_handler::background_work()
    {
        if (stopped_)
            return;

        // Atomically set handling_messages_ to true, if another work item hasn't
        // started executing before us.
        bool false_ = false;
        if (!handling_messages_.compare_exchange_strong(false_, true))
            return;

        if(!hpx::is_starting() && !use_io_pool_)
        {
            hpx::applier::register_thread_nullary(
                util::bind(&connection_handler::handle_messages, this),
                "ibverbs::connection_handler::handle_messages",
                threads::pending, true, threads::thread_priority_boost);
        }
        else
        {
            boost::asio::io_service& io_service = io_service_pool_.get_io_service(0);
            io_service.post(util::bind(&connection_handler::handle_messages, this));
        }
    }

    std::shared_ptr<sender> connection_handler::create_connection(
        parcelset::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service(0);
        std::shared_ptr<sender> sender_connection(new sender(*this, memory_pool_,
            l, parcels_sent_));

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;
        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            try {
                util::endpoint_iterator_type end = util::connect_end();
                for (util::endpoint_iterator_type it =
                        util::connect_begin(l.get<locality>(), io_service);
                      it != end; ++it)
                {
                    boost::asio::ip::tcp::endpoint const& ep = *it;

                    client_context& ctx = sender_connection->context();
                    ctx.close(ec);
                    ctx.connect(*this, ep, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                if (hpx::threads::get_self_ptr()) {
                    this_thread::suspend(hpx::threads::pending,
                        "connection_handler(ibverbs)::create_connection");
                }
                else {
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
            }
            catch (boost::system::system_error const& e) {
                sender_connection->context().close(ec);
                sender_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "ibverbs::parcelport::create_connection", e.what());
                return sender_connection;
            }
        }

        if (error) {
            sender_connection->context().close(ec);
            sender_connection.reset();

            std::ostringstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            HPX_THROWS_IF(ec, network_error,
                "ibverbs::parcelport::create_connection",
                strm.str());
            return sender_connection;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return sender_connection;
    }

    parcelset::locality connection_handler::agas_locality(
        util::runtime_configuration const & ini) const
    {
        // ibverbs can't be used for bootstrapping
        HPX_ASSERT(false);
        return parcelset::locality();
    }

    parcelset::locality connection_handler::create_locality() const
    {
        return parcelset::locality(locality());
    }

    void connection_handler::add_sender(
        std::shared_ptr<sender> const& sender_connection)
    {
        std::lock_guard<hpx::lcos::local::spinlock> l(senders_mtx_);
        senders_.push_back(sender_connection);
    }

    void add_sender(connection_handler & handler,
        std::shared_ptr<sender> const& sender_connection)
    {
        handler.add_sender(sender_connection);
    }

    ibv_pd *connection_handler::get_pd(ibv_context *context,
        boost::system::error_code & ec)
    {
        std::lock_guard<hpx::lcos::local::spinlock> l(pd_map_mtx_);
        typedef pd_map_type::iterator iterator;

        iterator it = pd_map_.find(context);
        if(it == pd_map_.end())
        {
            ibv_pd *pd = ibv_alloc_pd(context);
            if(pd == 0)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
                return 0;
            }
            pd_map_.insert(std::make_pair(context, pd));
            mr_map_.insert(std::make_pair(pd, mr_cache_type(mr_cache_size_)));
            return pd;
        }
        return it->second;

    }

    ibverbs_mr register_buffer(connection_handler & conn, ibv_pd * pd,
        char * buffer, std::size_t size, int access)
    {
        return conn.register_buffer(pd, buffer, size, access);
    }

    ibverbs_mr connection_handler::register_buffer(ibv_pd * pd,
        char * buffer, std::size_t size, int access)
    {

        chunk_pair chunk = memory_pool_.get_chunk_address(buffer, size);
        std::lock_guard<hpx::lcos::local::spinlock> l(mr_map_mtx_);
        typedef mr_map_type::iterator pd_iterator;
        pd_iterator it = mr_map_.find(pd);
        HPX_ASSERT(it != mr_map_.end());

        mr_cache_type & mr_cache = it->second;
        ibverbs_mr result;
        if(mr_cache.get_entry(chunk, result))
        {
            return result;
        }
        // register new one
        result =
            ibverbs_mr(
                pd
              , chunk.first
              , chunk.second
              , access
            );
        mr_cache.insert(chunk, result);

        return result;
    }

    namespace detail
    {
        struct handling_messages
        {
            handling_messages(boost::atomic<bool>& handling_messages_flag)
              : handling_messages_(handling_messages_flag)
            {}

            ~handling_messages()
            {
                handling_messages_.store(false);
            }

            boost::atomic<bool>& handling_messages_;
        };
    }

    void connection_handler::handle_messages()
    {
        detail::handling_messages hm(handling_messages_);       // reset on exit

        bool bootstrapping = hpx::is_starting();
        bool has_work = true;
        std::size_t k = 0;

        hpx::util::high_resolution_timer t;
        // We let the message handling loop spin for another 2 seconds to avoid the
        // costs involved with posting it to asio
        while(bootstrapping || (!stopped_ && has_work)
            || (!has_work && t.elapsed() < 2.0))
        {
            // handle all sends ...
            has_work = do_sends();
            // handle all receives ...
            if(do_receives())
            {
                has_work = true;
            }

            if (bootstrapping)
                bootstrapping = hpx::is_starting();

            if(has_work)
            {
                t.restart();
                k = 0;
            }
            else
            {
                if(enable_parcel_handling_)
                {
                    hpx::lcos::local::spinlock::yield(k);
                    ++k;
                }
            }
        }
    }

    bool connection_handler::do_sends()
    {
        hpx::util::high_resolution_timer t;
        std::lock_guard<hpx::lcos::local::spinlock> l(senders_mtx_);
        for(
            senders_type::iterator it = senders_.begin();
            !stopped_ && enable_parcel_handling_ && it != senders_.end();
            /**/)
        {
            if((*it)->done())
            {
                it = senders_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        return !senders_.empty();
    }

    bool connection_handler::do_receives()
    {
        hpx::util::high_resolution_timer t;
        std::lock_guard<hpx::lcos::local::spinlock> l(receivers_mtx_);

        for(
            receivers_type::iterator it = receivers_.begin();
            !stopped_ && enable_parcel_handling_ && it != receivers_.end();
            /**/)
        {
                boost::system::error_code ec;
                if((*it)->done(*this, ec))
                {
                    if(!ec)
                    {
                        HPX_IBVERBS_RESET_EC(ec)
                        (*it)->async_read(ec);
                    }
                }
                if(ec == boost::asio::error::eof
                || ec == boost::asio::error::operation_aborted)
                {
                    it = receivers_.erase(it);
                    continue;
                }
                ++it;
        }
        return !receivers_.empty();
    }

    void connection_handler::handle_accepts()
    {
        detail::handling_messages hm(handling_accepts_);       // reset on exit
        std::size_t k = 64;
        while(!stopped_)
        {
            hpx::util::high_resolution_timer t;
            std::shared_ptr<receiver> rcv = acceptor_.accept(
                *this, memory_pool_, boost::system::throws);
            if(rcv)
            {
                rcv->async_read(boost::system::throws);
                {
                    std::lock_guard<hpx::lcos::local::spinlock> l(receivers_mtx_);
                    receivers_.push_back(rcv);
                }
            }

            hpx::lcos::local::spinlock::yield(k);
        }
    }
}}}}

#endif
