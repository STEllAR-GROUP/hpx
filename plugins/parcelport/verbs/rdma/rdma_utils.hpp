// Copyright (c) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_VERBS_UTILS_HPP
#define HPX_PARCELSET_POLICIES_VERBS_UTILS_HPP

#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
//
#include <memory>
#include <string>
//
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs {

//! \brief InfiniBand device for RDMA operations.

struct rdma_utils
{
public:

    static std::string addressToString(const struct sockaddr_in *address)
    {
        std::ostringstream addr;
        char addrbuf[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(address->sin_addr.s_addr), addrbuf, INET_ADDRSTRLEN);
        addr << addrbuf << ":" << address->sin_port;
        return addr.str();
    }

    static int stringToAddress(const std::string addrString, const std::string portString,
        struct sockaddr_in *address)
    {
        struct addrinfo *res;
        int err = getaddrinfo(addrString.c_str(), portString.c_str(), nullptr, &res);
        if (err != 0) {
            LOG_ERROR_MSG("failed to get address info for " << addrString << ": " << rdma_error::error_string(err));
            return err;
        }

        if (res->ai_family != PF_INET) {
            LOG_ERROR_MSG("address family is not PF_INET");
            err = EINVAL;
        }

        else {
            memcpy(address, res->ai_addr, sizeof(struct sockaddr_in));
            address->sin_port = (in_port_t)atoi(portString.c_str());
        }

        freeaddrinfo(res);
        return err;
    }
};

}}}}

#endif

