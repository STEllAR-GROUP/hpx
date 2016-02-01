//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/agas/interface.hpp>
#include <hpx/components/component_storage/server/component_storage.hpp>

#include <fstream>

namespace hpx { namespace components { namespace server
{
    component_storage::component_storage()
      : data_(container_layout(find_all_localities()))
    {}

    ///////////////////////////////////////////////////////////////////////////
    naming::gid_type component_storage::migrate_to_here(
        std::vector<char> const& data, naming::id_type const& id,
        naming::address const& current_lva)
    {
        naming::gid_type gid(naming::detail::get_stripped_gid(id.get_gid()));
        data_[gid] = std::make_pair(current_lva.type_, data);

        // rebind the object to this storage locality
        naming::address addr(current_lva);
        addr.locality_ = this->get_base_gid();
        addr.address_ = 0;       // invalidate lva

        if (!agas::bind_sync(gid, addr, this->gid_))
        {
            std::ostringstream strm;
            strm << "failed to rebind id " << id
                 << "to storage locality: " << gid_;

            HPX_THROW_EXCEPTION(duplicate_component_address,
                "component_storage::migrate_to_here",
                strm.str());
            return naming::invalid_gid;
        }

        id.make_unmanaged();            // we can now release the object
        return naming::invalid_gid;
    }

    std::vector<char> component_storage::migrate_from_here(
        naming::gid_type const& id)
    {
        // return the stored data and erase it from the map
        naming::gid_type gid(naming::detail::get_stripped_gid(id));
        return data_.get_value_sync(gid, true).second;
    }

    ///////////////////////////////////////////////////////////////////////////
    // store and load from disk
    void component_storage::write_to_disk(std::string const& filename) const
    {
        // open file
        std::ofstream out(filename.c_str());
        if (!out.is_open())
        {
            HPX_THROW_EXCEPTION(filesystem_error,
                "component_storage::write_to_disk", "file could not be opened");
            return;
        }

        typedef data_store_type::partition_data_type partition_data_type;

        // gather data
        std::size_t num_partitions = data_.get_num_partitions();

        std::vector<hpx::future<partition_data_type> > partdata;
        partdata.reserve(num_partitions);

        data_store_type::const_segment_iterator end = data_.segment_cend();
        for (data_store_type::const_segment_iterator it = data_.segment_cbegin();
                it != end; ++it)
        {
            partdata.push_back((*it).get_data());
        }

        hpx::wait_all(partdata);

        // serialize the unordered_map
        std::vector<char> data;

        {
            serialization::output_archive archive(data);

            // store number of partitions
            archive << num_partitions;

            // store the data from each of the partitions
            for (auto && f : partdata)
            {
                archive << f.get();
            }
        }

        out << data.size();
        out.write(data.data(), data.size());
    }

    void component_storage::read_from_disk(std::string const& filename)
    {
        // open file
        std::ifstream in(filename.c_str());
        if (!in.is_open())
        {
            HPX_THROW_EXCEPTION(filesystem_error,
                "component_storage::write_to_disk", "file could not be opened");
            return;
        }

        typedef data_store_type::partition_data_type partition_data_type;

        // read all the data
        std::size_t data_size = 0;
        in >> data_size;

        std::vector<char> data(data_size);
        in.read(data.data(), data.size());

        hpx::naming::gid_type storage_gid = this->get_unmanaged_id().get_gid();

        // de-serialize all of the data
        {
            serialization::input_archive archive(data, data.size(), 0);

            std::size_t num_partitions = 0;
            archive >> num_partitions;

            data_ = data_store_type(container_layout(num_partitions));

            data_store_type::segment_iterator it = data_.segment_begin();
            data_store_type::segment_iterator end = data_.segment_end();

            for (std::size_t i = 0; i != num_partitions; ++i)
            {
                HPX_ASSERT(it != end);

                // read partition data from archive
                partition_data_type partition;
                archive >> partition;

                // re-register all gids with AGAS
                std::vector<hpx::future<bool> > agas_requests;
                agas_requests.reserve(partition.size());

                partition_data_type::const_iterator dend = partition.end();
                for (partition_data_type::const_iterator dit = partition.begin();
                     dit != dend; ++dit)
                {
                    hpx::naming::address addr(storage_gid, (*dit).second.first);
                    agas_requests.push_back(
                            hpx::agas::bind((*dit).first, addr, storage_gid)
                        );
                }

                // write the whole partition to the archive
                (*it).set_data(std::move(partition));
                ++it;

                // wait for AGAS requests to finish
                hpx::wait_all(agas_requests);
            }
        }
    }
}}}

HPX_REGISTER_UNORDERED_MAP(hpx::naming::gid_type, hpx_component_storage_data_type)
