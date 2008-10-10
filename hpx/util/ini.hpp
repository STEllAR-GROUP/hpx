//  Copyright (c) 2005-2007 Andre Merzky 
//  Copyright (c) 2005-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SECTION_SEP_17_2008_022PM)
#define HPX_UTIL_SECTION_SEP_17_2008_022PM

#include <map>
#include <iosfwd>
#include <boost/lexical_cast.hpp>

// suppress warnings about dependent classes not being exported from the dll
#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4091 4251 4231 4275 4660)
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT section
    {
    public:
        typedef std::map<std::string, std::string> entry_map;
        typedef std::map<std::string, section> section_map;

    private:
        section *this_() { return this; }

        section* root_;
        entry_map entries_;
        entry_map section_env_;
        section_map sections_;
        std::string name_;

    protected:
        bool regex_init();
        void line_msg(std::string const& msg, std::string const& file, 
            int lnum = 0);

    public:
        section();
        explicit section(std::string const& filename);
        section(section const& in);
        ~section() {}

        void parse(std::string const& sourcename, 
            std::vector<std::string> const& lines);
        void read(std::string const& filename);
        void merge(std::string const& second);
        void merge(section& second);
        void expand();
        void dump(int ind = 0, std::ostream& strm = std::cout) const;

        void add_section(std::string const& sec_name, section& sec, 
            section* root = NULL);
        bool has_section(std::string const& sec_name) const;

        section* get_section (std::string const& sec_name);
        section const* get_section (std::string const& sec_name) const;

        section_map const& get_sections() const
            { return sections_; }

        void add_entry(std::string const& key, std::string val);
        bool has_entry(std::string const& key) const;
        std::string get_entry(std::string const& key) const;
        std::string get_entry(std::string key, std::string const& dflt) const;
        template <typename T>
        std::string get_entry(std::string const& key, T dflt) const
        {
            return get_entry(key, boost::lexical_cast<std::string>(dflt));
        }

        entry_map const& get_entries() const
            { return entries_; }
        std::string expand_entry(std::string in) const;

        void expand_entry(std::string&, std::string::size_type) const;
        void expand_bracket(std::string&, std::string::size_type) const;
        void expand_brace(std::string&, std::string::size_type) const;

        void set_root(section* r) { root_ = r; }
        section* get_root() const { return root_; }
        std::string get_name() const { return name_; }

        section clone(section* root = NULL) const;
    };

}} // namespace hpx::util

#endif

