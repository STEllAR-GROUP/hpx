//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NBODY_GLOBAL_TIME_SOLVE_SERVER)
#define HPX_NBODY_GLOBAL_TIME_SOLVE_SERVER

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using hpx::async;
using hpx::lcos::future;
using std::vector;

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{
    struct particle{
        double mass;
        double p[3];
        double v[3];
    };

    typedef particle** history;

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT global_solve
      : public hpx::components::managed_component_base<global_solve>{
    public:
        global_solve(){}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        //[global_solve_methods
        //first function loads all data from specified file
        int init(std::string filename, hpx::naming::id_type gid){
            int i, j;
            double dat[7];
            std::ifstream infile;

            gid_ = gid;
            infile.open(filename);
            if(!infile) return 1;
            infile>>numBodies>>numIters>>dtime;

            //allocate the necessary space for the history data
            partA = new particle*[numIters+1];
            partB = new particle*[numIters+1];
            for(i = 0; i <= numIters; i++){
                partA[i] = new particle[numBodies];
                partB[i] = new particle[numBodies];
            }

            for(i = 0; i < numBodies; i++){
                infile>>dat[0]>>dat[1]>>dat[2]>>dat[3]>>dat[4]>>dat[5]>>dat[6];
                for(j = 0; j < numIters+1; j++){
                    partB[j][i].mass = partA[j][i].mass = dat[0];
                    partA[j][i].p[0] = dat[1];
                    partA[j][i].p[1] = dat[2];
                    partA[j][i].p[2] = dat[3];
                    partA[j][i].v[0] = dat[4];
                    partA[j][i].v[1] = dat[5];
                    partA[j][i].v[2] = dat[6];
                    if(j == 0){
                        partB[j][i].p[0] = dat[1];
                        partB[j][i].p[1] = dat[2];
                        partB[j][i].p[2] = dat[3];
                        partB[j][i].v[0] = dat[4];
                        partB[j][i].v[1] = dat[5];
                        partB[j][i].v[2] = dat[6];
                    }
                }
            }
            infile.close();
            return 0;
        }

        //run the entire simulation
        void run(int iters, int bchunk, int tchunk){
            int i;
            vector<int> cargs(5,0);
            cargs[0] = tchunk; cargs[1] = bchunk;
            cargs[2] = cargs[3] = cargs[4] = 0;

            //allocate space for the grid of futures to be used
            calcFutures = new future<void>**[
                (int)std::ceil(numIters/(double)tchunk)];
            for(i = 0; i < std::ceil(numIters/tchunk); i++)
                calcFutures[i]=new future<void>*[
                    (int)std::ceil(numBodies/(double)bchunk)];

            //run the simulation
            for(i = 0; i < iters; i++){
                cargs[2] = i;
                if(i > 0) calcFutures[0][0]->get();
                if(i % 2 == 0) calcFutures[0][0] = new future<void>(async<calc_action>(
                    gid_, false, false, cargs));
                else           calcFutures[0][0] = new future<void>(async<calc_action>(
                    gid_, false, true, cargs));
                printf("dinner %d\n", i);
            }
            histories = iters;
        }

        //performs a block of computations
        void calculate(bool cont, bool odd, vector<int> const& cargs){
            printf("in calculate\n");
            int A = (int)std::ceil(cargs[4]/(double)cargs[1]);
            int B = (int)std::ceil(cargs[3]/(double)cargs[0]);
            vector<int> temp1(cargs);
            vector<int> temp2(cargs);
            temp1[4] += cargs[1];
            if(temp1[4] < numBodies && !cont){
                if(cargs[2] > 0) calcFutures[A+1][B]->get();
                calcFutures[A+1][B] = new future<void>(async<calc_action>(
                    gid_, cont, odd, temp1));
            }

            int i, j, k;
            double factor = 1;
            double rSq, tforce;
            double force[3], dif[3], difSq[3];
            double exf;
            particle temp;
            history in, out;

            //swap input and output with each history calculated
            if(odd){
                in = partB; out = partA;
            }
            else{
                in = partA; out = partB;
            }

            for(i = cargs[4]; i < std::min(cargs[4]+cargs[1], numBodies); i++){
                if(!cont)   temp = in[0][i];
                else        temp = out[cargs[3]-1][i];
                for(j=cargs[3]; j < std::min(cargs[3]+cargs[0], numIters); j++){
                    force[0] = force[1] = force[2] = 0;
                    for(k = 0; k < numBodies; k++){
                        if(k == i) continue;
                        dif[0] = in[j][k].p[0] - temp.p[0];
                        dif[1] = in[j][k].p[1] - temp.p[1];
                        dif[2] = in[j][k].p[2] - temp.p[2];
                        difSq[0] = pow(dif[0], 2);
                        difSq[1] = pow(dif[1], 2);
                        difSq[2] = pow(dif[2], 2);
                        rSq = difSq[0] + difSq[1] + difSq[2];
                        tforce = factor*(temp.mass + in[j][k].mass) / rSq;
                        force[0] += tforce*dif[0];
                        force[1] += tforce*dif[1];
                        force[2] += tforce*dif[2];
                    }
                    exf = dtime/(2*in[j][i].mass);
                    out[j+1][i].v[0] = temp.v[0] = temp.v[0] + force[0]*exf;          
                    out[j+1][i].v[1] = temp.v[1] = temp.v[1] + force[1]*exf;          
                    out[j+1][i].v[2] = temp.v[2] = temp.v[2] + force[2]*exf;          
                    out[j+1][i].p[0]=temp.p[0]=temp.p[0]+out[j][i].v[0]*dtime;
                    out[j+1][i].p[1]=temp.p[1]=temp.p[1]+out[j][i].v[1]*dtime;
                    out[j+1][i].p[2]=temp.p[2]=temp.p[2]+out[j][i].v[2]*dtime;
                    out[j+1][i].v[0] = temp.v[0] = temp.v[0] + force[0]*exf;          
                    out[j+1][i].v[1] = temp.v[1] = temp.v[1] + force[1]*exf;          
                    out[j+1][i].v[2] = temp.v[2] = temp.v[2] + force[2]*exf; 
                }
            }
            temp2[3] += cargs[0];
            if(temp2[3] < numIters){
                if(cargs[2] > 0) calcFutures[A][B+1]->get();
                calcFutures[A][B+1] = new future<void>(async<calc_action>(
                    gid_, true, odd, temp2));
            }
        }

        //ouput the final history calculated in a series of files
        void report(std::string directory){
            std::string outputFile;
            std::ofstream outfile;
            history reference;
            int i, j, max = 1;
            if(histories % 2 == 1) reference = partA;
            else                   reference = partB;
            while(max < numIters) max *= 10;
            for(i = 0; i < numIters; i++){
                std::stringstream o;
                o << i + max;
                outputFile = directory;
                outputFile.append("/out");
                outputFile.append(o.str());
                outputFile.append(".dat");
                outfile.open(outputFile.c_str());
                outfile<<"# X   Y   Z\n";
                for(j = 0; j < numBodies; j++)
                    outfile<<reference[i][j].p[0]<<" "<<reference[i][j].p[1]
                        <<" "<<reference[i][j].p[2]<<"\n";
                outfile.close();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes. 
        enum actions
        {
            solver_init = 0,
            solver_run = 1,
            solver_report = 2,
            solver_calculate = 3
        };

        //global_solve_action_types
        typedef hpx::actions::result_action2<
            global_solve, int, solver_init, std::string, hpx::naming::id_type,
            &global_solve::init> init_action;

        typedef hpx::actions::action3<
            global_solve, solver_run, int, int,
            int, &global_solve::run> run_action;

        typedef hpx::actions::action1<
            global_solve, solver_report, std::string,
            &global_solve::report> report_action;

        typedef hpx::actions::action3<
            global_solve, solver_calculate, bool, bool,
            vector<int> const&, &global_solve::calculate> calc_action;

    //global_solve_server_data_members
    private:
        history partA, partB;
        int numBodies, numIters, histories;
        double dtime;
        future<void>*** calcFutures;
        hpx::naming::id_type gid_;
    };
}}

//Non-intrusive serialization
namespace boost { namespace serialization {
    template <typename Archive>
    void serialize(Archive&, examples::server::particle&, unsigned int const);
}}

//[global_solve_registration_declarations
HPX_REGISTER_ACTION_DECLARATION_EX(
    examples::server::global_solve::init_action,
    global_solve_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    examples::server::global_solve::run_action,
    global_solve_run_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    examples::server::global_solve::report_action,
    global_solve_report_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    examples::server::global_solve::calc_action,
    global_solve_calculate_action);

//]
#endif

