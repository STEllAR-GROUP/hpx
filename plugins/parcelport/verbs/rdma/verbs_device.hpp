// Copyright (c) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_VERBS_DEVICE_HPP
#define HPX_PARCELSET_POLICIES_VERBS_DEVICE_HPP

#include <memory>
#include <string>
#include <sstream>
//
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include <ifaddrs.h>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    class verbs_device
    {
    public:

        // ---------------------------------------------------------------------------
        verbs_device(std::string device, std::string interface) : context_(nullptr)
    {
            // Get the list of InfiniBand devices.
            int numDevices = 0;
            devices_ = ibv_get_device_list(&numDevices);
            if (devices_ == nullptr) {
                LOG_ERROR_MSG("no InfiniBand devices available");
                rdma_error e(ENODEV, "no InfiniBand devices available");
                throw e;
            }

            // Search for the specified InfiniBand device.
            device_ = nullptr;
            for (int index = 0; index < numDevices; ++index) {
                LOG_DEBUG_MSG("checking device " << ibv_get_device_name(devices_[index])
                    << " against " << device.c_str());
                if (strcmp(ibv_get_device_name(devices_[index]), device.c_str()) == 0) {
                    device_ = devices_[index];
                    break;
                }
            }

            // See if the requested InfiniBand device was found.
            if (device_ == nullptr) {
                LOG_ERROR_MSG("InfiniBand device " << device << " was not found");
                std::ostringstream what;
                what << "InfiniBand device " << device << " not found";
                rdma_error e(ENODEV, what.str());
                throw e;
            }
            LOG_DEBUG_MSG("found InfiniBand device " << get_device_name());

            interface_  = nullptr;
            interfaces_ = nullptr;

            if (!interface.empty()) {
                // Get the list of network interfaces.
                if (getifaddrs(&interfaces_) != 0) {
                    rdma_error e(errno, "getifaddrs() failed");
                    LOG_ERROR_MSG("error getting list of network interfaces: "
                        << rdma_error::error_string(e.error_code()));
                    throw e;
                }

                // Search for the specified network interface.
                for (interface_ = interfaces_; interface_ != nullptr;
                    interface_ = interface_->ifa_next)
                {
                    //      std::cout << "checking interface "
                    // << interface_->ifa_name << " against "
                    // << interface.c_str() << std::endl;
                    if ((strcmp(interface_->ifa_name, interface.c_str()) == 0)
                        && (interface_->ifa_addr->sa_family == AF_INET))
                    {
                        break;
                    }
                }

                // See if the specified network interface was found.
                if (interface_ == nullptr) {
                    LOG_ERROR_MSG("network interface "
                        << interface << " was not found on the node");
                    std::ostringstream what;
                    what << "network interface " << interface << " not found";
                    rdma_error e(ENOENT, what.str());
                    throw e;
                }
                LOG_DEBUG_MSG("found network interface " << get_interface_name());
            }
    }

        // ---------------------------------------------------------------------------
        ~verbs_device()
        {
            // Free the list of InfiniBand devices.
            ibv_free_device_list(devices_);

            if (interfaces_) {
                // Free the list of network interfaces.
                freeifaddrs(interfaces_);
            }

            //        if (context_) {
            //            if (ibv_close_device(context_)!=0) {
            //                LOG_ERROR_MSG("Failed to close device");
            //            }
            //        }
        }

        // ---------------------------------------------------------------------------
        std::string get_device_name(void)
        {
            std::string result;
            if (device_ != nullptr) {
                const char *namep = ibv_get_device_name(device_);
                if (namep != nullptr) {
                    result = namep;
                }
            }
            return result;
        }


        // ---------------------------------------------------------------------------
        std::string get_interface_name(void)
        {
            std::string result;
            if (interface_ != nullptr) {
                result = interface_->ifa_name;
            }
            return result;
        }

        // ---------------------------------------------------------------------------
        in_addr_t get_address(void)
        {
            if (interface_ == nullptr) {
                return 0;
            }
            struct sockaddr_in *addr =
                reinterpret_cast<struct sockaddr_in *>(interface_->ifa_addr);
            return addr->sin_addr.s_addr;
        }

        // ---------------------------------------------------------------------------
        struct ibv_context *get_context()
        {
            if (device_ == nullptr) {
                return nullptr;
            }

            if (context_ == nullptr) {
                struct ibv_context *ctx = ibv_open_device(device_);
                if (!ctx) {
                    LOG_ERROR_MSG("Failed to open device");
                    return 0;
                }
                LOG_DEBUG_MSG("Created context " << ctx);
                context_ = ctx;
            }
            return context_;
        }

        // ---------------------------------------------------------------------------
        std::string get_device_info(bool verbose)
        {
            struct ibv_device_attr device_attr;
            using namespace std;
            std::stringstream info;
            //
            ibv_context *ctx = this->get_context();
            if (ibv_query_device(ctx, &device_attr)) {
                LOG_ERROR_MSG("ibv_query_device failed");
            }
            else {
                info << "hca_id : " << ibv_get_device_name(device_) << "\n";
                if (strlen(device_attr.fw_ver))
                    info << setw(20) << "fw_ver" << device_attr.fw_ver << "\n"
// << setw(20) << "node_guid " << guid_str(device_attr.node_guid, buf) << "\n"
// << setw(20) << "sys_image_guid " << guid_str(device_attr.sys_image_guid, buf) << "\n"
                    << setw(20) << "vendor_id " << device_attr.vendor_id << "\n"
                    << setw(20) << "vendor_part_id " << device_attr.vendor_part_id << "\n"
                    << setw(20) << "hw_ver " << device_attr.hw_ver << "\n"
                    << setw(20) << "phys_port_cnt " << device_attr.phys_port_cnt << "\n";

                if (verbose) {
                    info << setw(20) << "max_mr_size " <<
                        (unsigned long long) device_attr.max_mr_size << "\n"
                        << setw(20) << "page_size_cap " <<
                        (unsigned long long) device_attr.page_size_cap << "\n"
                        << setw(20) << "max_qp " <<  device_attr.max_qp << "\n"
                        << setw(20) << "max_qp_wr " <<  device_attr.max_qp_wr << "\n"
                        << setw(20) << "device_cap_flags "
                        << device_attr.device_cap_flags << "\n"
                        << setw(20) << "max_sge " <<  device_attr.max_sge << "\n"
                        << setw(20) << "max_sge_rd " <<  device_attr.max_sge_rd << "\n"
                        << setw(20) << "max_cq " << device_attr.max_cq << "\n"
                        << setw(20) << "max_cqe " << device_attr.max_cqe << "\n"
                        << setw(20) << "max_mr " << device_attr.max_mr << "\n"
                        << setw(20) << "max_pd " << device_attr.max_pd << "\n"
                        << setw(20) << "max_qp_rd_atom "
                        << device_attr.max_qp_rd_atom << "\n"
                        << setw(20) << "max_ee_rd_atom "
                        << device_attr.max_ee_rd_atom << "\n"
                        << setw(20) << "max_res_rd_atom "
                        << device_attr.max_res_rd_atom << "\n"
                        << setw(20) << "max_qp_init_rd_atom "
                        << device_attr.max_qp_init_rd_atom << "\n"
                        << setw(20) << "max_ee_init_rd_atom "
                        << device_attr.max_ee_init_rd_atom << "\n"
                        //                    << setw(20) << "atomic_cap "
                        // << atomic_cap_str(device_attr.atomic_cap) << "\n"
                        << setw(20) << "max_ee " << device_attr.max_ee << "\n"
                        << setw(20) << "max_rdd " << device_attr.max_rdd << "\n"
                        << setw(20) << "max_mw " << device_attr.max_mw << "\n"
                        << setw(20) << "max_raw_ipv6_qp "
                        << device_attr.max_raw_ipv6_qp << "\n"
                        << setw(20) << "max_raw_ethy_qp "
                        << device_attr.max_raw_ethy_qp << "\n"
                        << setw(20) << "max_mcast_grp "
                        << device_attr.max_mcast_grp << "\n"
                        << setw(20) << "max_mcast_qp_attach "
                        << device_attr.max_mcast_qp_attach << "\n"
                        << setw(20) << "max_total_mcast_qp_attach "
                        << device_attr.max_total_mcast_qp_attach << "\n"
                        << setw(20) << "max_ah " << device_attr.max_ah << "\n"
                        << setw(20) << "max_fmr " <<  device_attr.max_fmr << "\n";
                    if (device_attr.max_fmr) {
                        info << setw(20) << "max_map_per_fmr "
                            << device_attr.max_map_per_fmr << "\n";
                    }
                    info << setw(20) << "max_srq " << device_attr.max_srq << "\n";
                    if (device_attr.max_srq) {
                        info << setw(20) << "max_srq_wr "
                            << device_attr.max_srq_wr << "\n"
                            << setw(20) << "max_srq_sge "
                            << device_attr.max_srq_sge << "\n";
                    }
                    info << setw(20) << "max_pkeys " << device_attr.max_pkeys << "\n"
                        << setw(20) << "local_ca_ack_delay "
                        << device_attr.local_ca_ack_delay << "\n";
                }
            }
            return info.str();
        }

    private:
        // Pointer to list of InfiniBand devices.
        struct ibv_device **devices_;

        // Pointer to selected device in the list.
        struct ibv_device *device_;

        struct ibv_context *context_;

        // Pointer to list of network interfaces.
        struct ifaddrs *interfaces_;

        // Pointer to selected interface in the list.
        struct ifaddrs *interface_;

    };

    // Smart pointer for verbs_device object.
    typedef std::shared_ptr<verbs_device> verbs_device_ptr;

}}}}

#endif
