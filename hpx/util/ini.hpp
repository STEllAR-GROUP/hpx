//  Copyright (c) 2005-2007 Andre Merzky
//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SECTION_SEP_17_2008_022PM)
#define HPX_UTIL_SECTION_SEP_17_2008_022PM

#include <map>
#include <iostream>

#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <boost/lexical_cast.hpp>

// suppress warnings about dependent classes not being exported from the dll
#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4091 4251 4231 4275 4660)
#endif

///////////////////////////////////////////////////////////////////////////////
//  section serialization format version
#define HPX_SECTION_VERSION 0x10

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
        section_map sections_;
        std::string name_;
        std::string parent_name_;

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER()

    protected:
        bool regex_init();
        void line_msg(std::string msg, std::string const& file,
            int lnum = 0, std::string const& line = "");

        section& clone_from(section const& rhs, section* root = 0);

    public:
        section();
        explicit section(std::string const& filename, section* root = NULL);
        section(section const& in);
        ~section() {}

        section& operator=(section const& rhs);

        void parse(std::string const& sourcename,
            std::vector<std::string> const& lines, bool verify_existing = true,
            bool weed_out_comments = true);

        void parse(std::string const& sourcename,
            std::string const& line, bool verify_existing = true,
            bool weed_out_comments = true)
        {
            std::vector<std::string> lines;
            lines.push_back(line);
            parse(sourcename, lines, verify_existing, weed_out_comments);
        }

        void read(std::string const& filename);
        void merge(std::string const& second);
        void merge(section& second);
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
        std::string get_entry(std::string const& key, std::string const& dflt) const;
        template <typename T>
        std::string get_entry(std::string const& key, T dflt) const
        {
            return get_entry(key, boost::lexical_cast<std::string>(dflt));
        }

        entry_map const& get_entries() const
            { return entries_; }
        std::string expand(std::string in) const;

        void expand(std::string&, std::string::size_type) const;
        void expand_bracket(std::string&, std::string::size_type) const;
        void expand_brace(std::string&, std::string::size_type) const;

        std::string expand_only(std::string in,
            std::string const& expand_this) const;

        void expand_only(std::string&, std::string::size_type,
            std::string const& expand_this) const;
        void expand_bracket_only(std::string&, std::string::size_type,
            std::string const& expand_this) const;
        void expand_brace_only(std::string&, std::string::size_type,
            std::string const& expand_this) const;

        void set_root(section* r, bool recursive = false)
        {
            root_ = r;
            if (recursive) {
                section_map::iterator send = sections_.end();
                for (section_map::iterator si = sections_.begin(); si != send; ++si)
                    si->second.set_root(r, true);
            }
        }
        section* get_root() const { return root_; }
        std::string get_name() const { return name_; }
        std::string get_parent_name() const { return parent_name_; }
        std::string get_full_name() const
        {
            if (!parent_name_.empty())
                return parent_name_ + "." + name_;
            return name_;
        }

        void set_name(std::string const& name) { name_ = name; }
    };

}} // namespace hpx::util

#endif

