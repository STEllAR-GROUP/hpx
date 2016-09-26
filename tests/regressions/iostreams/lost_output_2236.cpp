//  Copyright (c) 2012016 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>
#include <sstream>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::string expected_output;
hpx::lcos::local::spinlock expected_output_mtx;

template <typename ... Ts>
void generate_output(Ts &&... ts)
{
    std::stringstream stream;
    int const sequencer[] =
    {
        0, (stream << ts, 0)...
    };
    (void)sequencer;

    stream << std::endl;

    std::string str = stream.str();

    {
        std::lock_guard<hpx::lcos::local::spinlock> l(expected_output_mtx);
        expected_output += str;
    }

    hpx::consolestream << str;
    HPX_TEST(hpx::consolestream);

    hpx::cout << str;
    HPX_TEST(hpx::cout);
}

template <typename ... Ts>
void generate_output_no_endl(Ts &&... ts)
{
    std::stringstream stream;
    int const sequencer[] =
    {
        0, (stream << ts, 0)...
    };
    (void)sequencer;

    std::string str = stream.str();

    {
        std::lock_guard<hpx::lcos::local::spinlock> l(expected_output_mtx);
        expected_output += str;
    }


    hpx::consolestream << str;
    HPX_TEST(hpx::consolestream);

    hpx::cout << str;
    HPX_TEST(hpx::cout);
}

///////////////////////////////////////////////////////////////////////////////
namespace gc {

struct collector_data
{
    hpx::id_type parent, cid;
    unsigned int minor_id, phantom_count, rcc, wc;
    bool rephantomize, start_over_recovery, rerecover, phantomized, recovered;

    // ctors
    collector_data()
      : parent(hpx::naming::invalid_id)
      , cid(hpx::naming::invalid_id)
      , minor_id(0)
      , phantom_count(0)
      , rcc(0)
      , wc(0)
      , rephantomize(false)
      , start_over_recovery(false)
      , rerecover(false)
      , phantomized(false)
      , recovered(false)
    {
    }
};

namespace server {

    extern int global_id;

    struct collectable : hpx::components::component_base<collectable>
    {
        int id = global_id++;
        unsigned int weight, max_weight, strong_count, weak_count;
        collector_data *cd;

        std::vector<hpx::id_type> outgoing();
        HPX_DEFINE_COMPONENT_ACTION(collectable, outgoing);

        // Assume a root is pointing to the object
        collectable()
          : weight(1)
          , max_weight(0)
          , strong_count(0)
          , weak_count(0)
          , cd(nullptr)
        {
        }
        collectable(const collectable &src)
          : weight(src.weight + 1)
          , max_weight(src.weight)
          , strong_count(1)
          , weak_count(0)
          , cd(nullptr)
        {
        }
        ~collectable()
        {
        }

        int add_ref(hpx::id_type id)
        {
            // use a mutex here?
            int n = out_refs.size();
            if (id != hpx::naming::invalid_id)
            {
                out_refs.push_back(id);
                incref_(weight, id);
            }
            return n;
        }
        HPX_DEFINE_COMPONENT_ACTION(collectable, add_ref);

        void set_ref(unsigned int index, hpx::naming::id_type id)
        {
            hpx::naming::id_type old_id = out_refs.at(index);
            if (id == old_id)
                return;
            out_refs[index] = id;
            if (old_id != hpx::naming::invalid_id)
                decref_(weight, old_id);
            if (id != hpx::naming::invalid_id)
                incref_(weight, id);
        }
        HPX_DEFINE_COMPONENT_ACTION(collectable, set_ref);

        void phantomize_ref(
            unsigned int weight, hpx::id_type parent, hpx::id_type cid);
        HPX_DEFINE_COMPONENT_ACTION(collectable, phantomize_ref);
        void incref(unsigned int weight);
        HPX_DEFINE_COMPONENT_ACTION(collectable, incref);
        void incref_(unsigned int weight, hpx::naming::id_type id);
        void decref(unsigned int weight);
        HPX_DEFINE_COMPONENT_ACTION(collectable, decref);
        void decref_(unsigned int weight, hpx::naming::id_type id);

        void phantom_wait_complete();

        void done(hpx::id_type child);
        HPX_DEFINE_COMPONENT_ACTION(collectable, done);

        void recover(hpx::id_type cid);
        HPX_DEFINE_COMPONENT_ACTION(collectable, recover);

        void spread(unsigned int weight);

        void recover_done();
        HPX_DEFINE_COMPONENT_ACTION(collectable, recover_done);

        void check_recover_done();

    private:
        void clean();
        void state();
        std::vector<hpx::id_type> out_refs;
    };
}
}

HPX_REGISTER_ACTION_DECLARATION(gc::server::collectable::phantomize_ref_action,
    gc_collectable_phantomize_ref_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::add_ref_action, gc_collectable_add_ref_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::set_ref_action, gc_collectable_set_ref_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::outgoing_action, gc_collectable_outgoing_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::incref_action, gc_collectable_incref_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::decref_action, gc_collectable_decref_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::done_action, gc_collectable_done_action);

HPX_REGISTER_ACTION_DECLARATION(
    gc::server::collectable::recover_action, gc_collectable_recover_action);

HPX_REGISTER_ACTION_DECLARATION(gc::server::collectable::recover_done_action,
    gc_collectable_recover_done_action);

namespace gc {
struct collectable
    : hpx::components::client_base<collectable, server::collectable>
{
    typedef hpx::components::client_base<collectable, server::collectable>
        base_type;

    collectable(hpx::future<hpx::id_type> f)
      : base_type(std::move(f))
    {
        incref(0);
    }

    collectable(hpx::id_type f)
      : base_type(std::move(f))
    {
        incref(0);
    }

    ~collectable()
    {
        decref(0);
    }

    hpx::future<int> add_ref(hpx::id_type id)
    {
        return hpx::async<server::collectable::add_ref_action>(
            this->get_id(), id);
    }
    hpx::future<void> set_ref(int index, hpx::id_type id)
    {
        return hpx::async<server::collectable::set_ref_action>(
            this->get_id(), index, id);
    }
    hpx::future<std::vector<hpx::id_type>> outgoing()
    {
        return hpx::async<server::collectable::outgoing_action>(this->get_id());
    }
    void incref(unsigned int w)
    {
        hpx::apply<server::collectable::incref_action>(this->get_id(), w);
    }
    void decref(unsigned int w)
    {
        hpx::apply<server::collectable::decref_action>(this->get_id(), w);
    }
};
}

namespace gc {
namespace server {

    int global_id = 0;

    std::vector<hpx::id_type> collectable::outgoing()
    {
        return out_refs;
    }

    void collectable::incref(unsigned int w)
    {
        if (w < weight)
        {
            strong_count++;
        }
        else
        {
            weak_count++;
            if (w > max_weight)
                max_weight = w;
        }
        state();
    }
    void collectable::phantomize_ref(
        unsigned int w, hpx::id_type parent, hpx::id_type cid)
    {
        if (cd == nullptr)
        {
            cd = new collector_data;
            cd->cid = cid;
            cd->parent = parent;
        }
        else if (cid > cd->cid)
        {
            cd->cid = cid;
            cd->parent = parent;
        }
        generate_output("Phantomized ref");
        state();
        if (w < weight)
        {
            HPX_ASSERT(strong_count > 0);
            strong_count--;
        }
        else
        {
            weak_count--;
            HPX_ASSERT(weak_count >= 0);
        }
        cd->phantom_count++;
        state();
        spread(weight);
        hpx::apply<collectable::done_action>(parent, this->get_id());
    }
    void collectable::phantom_wait_complete()
    {
        generate_output("Waiting complete ", id);
        if (strong_count > 0)
        {
            // we can build
            HPX_ASSERT("build not handled" == nullptr);
        }
        else
        {
            // we need to recover
            recover(cd->cid);
        }
    }
    void collectable::recover(hpx::id_type cid)
    {
        if (cid > cd->cid)
        {
            HPX_ASSERT(
                "change of owner during recovery not handled" == nullptr);
        }
        if (!cd->recovered)
        {
            cd->recovered = true;
            for (auto i : out_refs)
            {
                if (i != hpx::naming::invalid_id)
                {
                    hpx::apply<collectable::recover_action>(i, cd->cid);
                    cd->wc++;
                }
            }
        }
        check_recover_done();
    }
    void collectable::check_recover_done()
    {
        if (cd->wc == 0)
        {
            hpx::id_type this_id = get_id();
            if (cd->cid == this_id)
            {
                if (strong_count == 0)
                {
                    generate_output("Delete!");
                }
            }
            else
            {
                hpx::apply<collectable::recover_done_action>(cd->parent);
            }
        }
    }
    void collectable::recover_done()
    {
        cd->wc--;
        HPX_ASSERT(cd->wc > 0);
        check_recover_done();
    }
    void collectable::done(hpx::id_type child)
    {
        cd->wc--;
        HPX_ASSERT(cd->wc >= 0);
        if (cd->wc == 0 && cd->cid == this->get_id())
        {
            phantom_wait_complete();
            state();
        }
    }
    void collectable::state()
    {
        generate_output_no_endl("id=", id, " wt=", weight, " sc=", strong_count,
            " wc=", weak_count);
        if (cd != nullptr)
        {
            generate_output_no_endl(" pc=", cd->phantom_count);
        }
        generate_output();
    }
    void collectable::decref(unsigned int w)
    {
        if (w < weight)
        {
            strong_count--;
            if (strong_count == 0)
            {
                if (weak_count == 0)
                {
                    if (cd == nullptr || cd->phantom_count == 0)
                    {
                        clean();
                    }
                }
                else
                {
                    generate_output("Maybe garbage ", id);
                    unsigned int ow = weight;
                    weight = max_weight + 1;
                    strong_count = weak_count;
                    weak_count = 0;
                    if (cd == nullptr)
                    {
                        cd = new collector_data;
                        cd->cid = this->get_id();
                    }
                    spread(ow);
                }
            }
        }
        else
        {
            weak_count--;
            HPX_ASSERT(weak_count >= 0);
            if (weak_count == 0 && strong_count == 0)
            {
                if (cd == nullptr || cd->phantom_count == 0)
                {
                    clean();
                }
            }
        }
        state();
    }

    void collectable::spread(unsigned int ow)
    {
        if (!cd->phantomized)
        {
            cd->phantomized = true;
            for (auto i = out_refs.begin(); i != out_refs.end(); ++i)
            {
                if (*i != hpx::naming::invalid_id)
                {
                    cd->wc++;
                    hpx::async<collectable::phantomize_ref_action>(
                        *i, ow, this->get_id(), cd->cid);
                }
            }
            if (cd->wc == 0 && cd->cid == this->get_id())
            {
                phantom_wait_complete();
            }
        }
    }

    void collectable::clean()
    {
        generate_output("Definitely garbage ", id);
        int n = out_refs.size();
        for (int i = 0; i < n; i++)
        {
            set_ref(i, hpx::naming::invalid_id);
        }
    }

    void collectable::incref_(unsigned int weight, hpx::naming::id_type id)
    {
        hpx::async<server::collectable::incref_action>(id, weight).wait();
        state();
    }
    void collectable::decref_(unsigned int weight, hpx::naming::id_type id)
    {
        hpx::async<server::collectable::decref_action>(id, weight).wait();
        state();
    }
}
}

typedef hpx::components::component<gc::server::collectable> collectable_type;

HPX_REGISTER_COMPONENT(collectable_type, collectable);

HPX_REGISTER_ACTION(gc::server::collectable::phantomize_ref_action,
    gc_collectable_phantomize_ref_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::add_ref_action, gc_collectable_add_ref_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::set_ref_action, gc_collectable_set_ref_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::outgoing_action, gc_collectable_outgoing_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::incref_action, gc_collectable_incref_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::decref_action, gc_collectable_decref_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::done_action, gc_collectable_done_action);

HPX_REGISTER_ACTION(
    gc::server::collectable::recover_action, gc_collectable_recover_action);

HPX_REGISTER_ACTION(gc::server::collectable::recover_done_action,
    gc_collectable_recover_done_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    generate_output("Here1");
    {
        auto loc = hpx::find_here();
        gc::collectable c1(hpx::components::new_<gc::server::collectable>(loc));
        {
            gc::collectable c2(
                hpx::components::new_<gc::server::collectable>(loc));
            int index = c1.add_ref(c2.get_id()).get();
            c2.add_ref(c1.get_id()).wait();
        }
    }
    generate_output("Here2");
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    std::stringstream const& console_strm = hpx::get_consolestream();
    HPX_TEST_EQ(console_strm.str(), expected_output);

    return hpx::util::report_errors();
}
