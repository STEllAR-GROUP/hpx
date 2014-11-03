//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/components/io/pxfs_file.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/program_options.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <string>
#include <vector>

using hpx::io::pxfs_file;
using std::string;
using std::vector;

///////////////////////////////////////////////////////////////////////////////
bool test_pxfs_file_component(string const& file_name)
{
    {
        // testing writing operations
        pxfs_file pf;

        pf.open_sync(file_name, O_WRONLY|O_CREAT);
        if (!pf.is_open())
        {
            hpx::cerr << "Faile to open pxfs file " << file_name <<
                " for write!" << hpx::endl;
            return false;
        }

        char content1[] = "To be or not to be, ";
        char content2[] = "that is the question. ";
        // -1 to get rid of "\0"
        vector<char> buf1(content1, content1 + sizeof(content1) / sizeof(char) - 1);
        vector<char> buf2(content2, content2 + sizeof(content2) / sizeof(char) - 1);

        ssize_t ss = pf.write_sync(buf1);
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf1.size()));
        ss = pf.write_sync(buf2);
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf2.size()));

        int rd = pf.lseek_sync(0, SEEK_SET);
        HPX_TEST_EQ(rd, 0);

        ss = pf.pwrite_sync(buf1, buf1.size() + buf2.size());
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf1.size()));

        ss = pf.pwrite_sync(buf2, 2 * buf1.size() + buf2.size());
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf2.size()));

        pf.close_sync();
        HPX_TEST(!pf.is_open());
    }

    {
        // testing reading operations
        pxfs_file pf;

        pf.open_sync(file_name, O_RDONLY);
        if (!pf.is_open())
        {
            hpx::cerr << "Faile to open file " << file_name <<
                " for read!" << hpx::endl;
            return false;
        }

        char content1[] = "To be or not to be, ";
        char content2[] = "that is the question. ";
        vector<char> buf1(content1, content1 + sizeof(content1) / sizeof(char) - 1);
        vector<char> buf2(content2, content2 + sizeof(content2) / sizeof(char) - 1);

        vector<char> buf = pf.read_sync(buf1.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf1.begin(), buf1.end()));

        buf = pf.read_sync(buf2.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf2.begin(), buf2.end()));

        int rd = pf.lseek_sync(0, SEEK_SET);
        HPX_TEST_EQ(rd, 0);

        buf = pf.pread_sync(buf1.size(), buf1.size() + buf2.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf1.begin(), buf1.end()));

        buf = pf.pread_sync(buf2.size(), 2 * buf1.size() + buf2.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf2.begin(), buf2.end()));

        pf.close_sync();
        HPX_TEST(!pf.is_open());

        // remove file
        HPX_TEST_EQ(pf.remove_file_sync(file_name), 0);
    }
    return true;
}


///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // default OrangeFS env or PVFS2TAB_FILE needs to be set
    namespace po = boost::program_options;

    // Configure application-specific options
    po::options_description
       desc_commandline("Usage: " + std::string(argv[0]) +
               " --path <pxfs_test_path>");

    desc_commandline.add_options()
        ( "path" , po::value<std::string>(), "PXFS test path.")
        ( "help" , "Print help message.")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc_commandline), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        hpx::cerr << desc_commandline << hpx::endl;
        return hpx::util::report_errors();
    }

    if (!vm.count("path"))
    {
        hpx::cerr << "need to specify PXFS test path!!" << hpx::endl;
        hpx::cerr << "Usage: " << argv[0] << " --path <path>" << hpx::endl;
        return hpx::util::report_errors();
    }

    std::string pxfs_path = vm["path"].as<std::string>();
    const string file_name = pxfs_path + "/test_pxfs_file";
    HPX_TEST(test_pxfs_file_component(file_name));

    return hpx::util::report_errors();
}
