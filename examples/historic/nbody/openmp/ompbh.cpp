#include <iostream>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include <omp.h>

static double dtime; // length of one time step
static double eps; // potential softening parameter
static double tolerance; // tolerance for stopping recursion, should be less than 0.57 for 3D case to bound error warren salmon 94
static double half_dt;
//http://www.cs.toronto.edu/~wayne/research/thesis/msc/node5.html
// force softening, i.e.,  replacing r^2 with (r ^ 2 + softening ^ 2) in the denominator of the gravitational force computation 
// for some small constant (softening factor epsilon (eps)) , usually chosen to approximate the average inter-particle separation. 
// This is done because it allows a smaller N to approximate a larger N, and also to eliminate the singularity at r=0
static double softening_2; // softening ^ 2 
double theta=0;

static double inv_tolerance_2;
static int iter;

using namespace std;

template <typename T>
T square(const T& value)
{
	return value * value;
}

/*
 * The N-Body tree is composed of two different kinds of particles
 * CELL - intermediate nodes
 * PAR - Terminal nodes which contain the body/particle.
 */
enum 
{
	CELL, PAR
};

///////// Main Class Definitions
class TreeLeaf;
static TreeLeaf **particles; // the n particles


class TreeNode 
{
	public:
		int node_type; /* Each node has a node type which is either NDE or PAR*/
		double mass;   /* Each node has a mass */
		double p[3];   /* position of each node is stored in a vector of type double named p */
};

class IntrTreeNode: public TreeNode /* Internal node inherits from base TreeNode */
{ 
	private:
		IntrTreeNode *edge_link; /* Each Internal node has a pointer to link itself with rest of the tree */
		static IntrTreeNode *tree_head, *free_node_list; /* Each also has two static links to the current root and the free list */
	public:
		static IntrTreeNode *newNode(const double pos_buf[]); /* Function to create a new node, returns an Internal Tree Node */
		void treeNodeInsert(TreeLeaf * const new_particle, const double sub_box_dim); /* Function to insert a particle (tree leaf) node  */
		void calculateCM(int &current_index); /* Recursive function to compute center of mass */

		static void treeReuse()      /* function to recycle tree */
		{
			free_node_list = tree_head; /* point the free List to the root of the tree */
		}
		TreeNode *branch[8];  /* Each Internal node has pointers to 8 branch nodes (since it is an octtree) of type TreeNode*/
};

class TreeLeaf: public TreeNode   /* Terminal / leaf nodes extend intermediate nodes */
{                                 /* in addition to the mass, px, py, pz the internal */
	public:                       /* nodes also have velocity vector of type double named v and */
		double v[3];              /* acceleration vector of type double named a  */
		double a[3];
		TreeLeaf()                /* TreeLeaf Constructor */
		{
			node_type = PAR;     /* Initialize TreeLeaf node to PARTICLE type */
			mass = 0.0;          /* Initialize mass to zero */
			p[0] = 0.0; /* Initialize position vector elements to 0 */
			p[1] = 0.0; /* Initialize position vector elements to 0 */
			p[2] = 0.0; /* Initialize position vector elements to 0 */
			v[0] = 0.0; /* Initialize velocity vector to 0 */
			v[1] = 0.0; /* Initialize velocity vector to 0 */
			v[2] = 0.0; /* Initialize velocity vector to 0 */
			a[0] = 0.0; /* Initialize acceleration vector to 0 */
			a[1] = 0.0; /* Initialize acceleration vector to 0 */
			a[2] = 0.0; /* Initialize acceleration vector to 0 */
		}
		void moveParticles(); /* moves particles according to acceleration and velocity calculated */
		void calculateForce(const IntrTreeNode * const root, const double box_size); /* calculates the acceleration and velocity of each particle in the tree */
		~TreeLeaf() { }       /* TreeLeaf Destructor */
	private:
		void forceCalcRe(const TreeNode * const n, double box_size_2); /* traverses tree recursively while calculating force for each particle */
};

//////////Main Class Methods

IntrTreeNode *IntrTreeNode::tree_head = NULL;  /* Initialize the head pointer to NULL */
IntrTreeNode *IntrTreeNode::free_node_list = NULL; /* Initialize the free list pointer to NULL */

IntrTreeNode *IntrTreeNode::newNode(const double pos_buf[]) 
{
	register IntrTreeNode *temp_node; /* Declare the temporary node of type TreeNodeI */

	if (free_node_list == NULL)           /* if the list of free nodes is Null then create */
	{                                     /* a new node and link it to the previous node of the tree */
		temp_node = new IntrTreeNode();   /* create a new node of type Tree Node internal and store it in temp_node */
		temp_node->edge_link = tree_head; /* initialize the link to point to the previous node of the tree */
		tree_head = temp_node;            /* set new subroot of the tree to be temp_node */
	} 
	else 
	{ 
		temp_node = free_node_list;       /* get a node from the free list of nodes ... re use previously allocated */
		free_node_list = free_node_list->edge_link; /* memory space. Helps in reuse of space */
	}

	temp_node->node_type = CELL;  /* Set the node type to CELL (since an intermediate node is being created */
	temp_node->mass = 0.0;        /* intermediate node have no mass. Center of Mass is populated here later*/
	temp_node->p[0] = pos_buf[0]; /* Set the position of the cell to the position passed in*/
	temp_node->p[1] = pos_buf[1];
	temp_node->p[2] = pos_buf[2];
	for (int i = 0; i < 8; ++i)        /* Intermediate nodes in a BHT can have upto 8 branch.  */
		temp_node->branch[i] = NULL;   /* Initialized to NULL */

	return temp_node;
}

/* Insert New Particle: This function inserts a new particle from the specified level 
 * values passed into the function are:
 *     the new particle to be inserted into the tree named new_particle (of type TreeLeaf)
 *     the value representing dimension of the sub_box  helps determine the 
 *     position of intermediate node
 */
void IntrTreeNode::treeNodeInsert(TreeLeaf * const new_particle, const double sub_box_dim) // builds the tree
{
	register int i = 0;                        /* Initialize stores the index  */ 
	register double temp[3] ;                  /* Initialize buffer position to 0.0 */
    temp[0] = 0.0;                    /* bufPos stores the value which helps derive*/
    temp[1] = 0.0;                    /* bufPos stores the value which helps derive*/
    temp[2] = 0.0;                    /* bufPos stores the value which helps derive
                                                * the coordinates of a new intermediate node
                                                * if one has to be created.
                                                */
    /*
     * If the curr.x > new.x & curr.y > new.y & curr.z > new.z then store at i=0 buf.x=0, buf.y=0, buf.z=0
     * If the curr.x < new.x & curr.y > new.y & curr.z > new.z then store at i=1 buf.x=sub_box, buf.y=0, buf.z=0
     * If the curr.x > new.x & curr.y < new.y & curr.z > new.z then store at i=2 buf.x=0, buf.y=sub_box, buf.z=0
     * If the curr.x < new.x & curr.y < new.y & curr.z > new.z then store at i=3 buf.x=sub_box, buf.y=sub_box, buf.z=0
     * If the curr.x > new.x & curr.y > new.y & curr.z < new.z then store at i=4 buf.x=0, buf.y=0, buf.z=sub_box
     * If the curr.x < new.x & curr.y > new.y & curr.z < new.z then store at i=5 buf.x=sub_box, buf.y=0, buf.z=sub_box
     * If the curr.x > new.x & curr.y < new.y & curr.z < new.z then store at i=6 buf.x=0, buf.y=sub_box, buf.z=sub_box
     * If the curr.x < new.x & curr.y < new.y & curr.z < new.z then store at i=7 buf.x=sub_box, buf.y=sub_box, buf.z=sub_box
     */ 
	if (p[0] < new_particle->p[0]) 
	{
		i = 1;
		temp[0] = sub_box_dim;
	}
	if (p[1] < new_particle->p[1]) 
	{
		i += 2;
		temp[1] = sub_box_dim;
	}
	if (p[2] < new_particle->p[2]) 
	{
		i += 4;
		temp[2] = sub_box_dim;
	}

    /* if the branch node at i as determined above is empty then
     * store current body at that position */
	if (branch[i] == NULL) 
	{
		branch[i] = new_particle;
	} 
	/* if the branch node at the current i is a CELL then
     * traverse to the branch nodes to see where the body 
     * can be inserted */ 
	else if (branch[i]->node_type == CELL) 
	{
		((IntrTreeNode *) (branch[i]))->treeNodeInsert(new_particle, 0.5 * sub_box_dim);
	} 

	/* if all branch node at the current position is a LEAF 
     * then create a node at the current position 
     * insert the current branch node into the subtree
     * insert the new node into the subtree */
	else 
	{
		/* since the cell that we are inserting in, is already a terminal node
         * we split the cell and create an intermediate node. In order to do this
         * we need to allocate an intermediate node */
        /* Since we are creating a new intermediate node we divide sub_box by 2 */
		register const double new_sub_box_dim = 0.5 * sub_box_dim;
		/* The coordinates of the intermediate which will point to the two branch nodes 
         * can be calculated 
         * the X coordinate of the intermediate node is p[0] - sub_box2 + bufPos[0]
         * the Y coordinate of the intermediate node is p[1] - sub_box2 + bufPos[1]
         * the Z coordinate of the intermediate node is p[2] - sub_box2 + bufPos[2]
         **/
		register const double pos_buf[] = { p[0] - new_sub_box_dim + temp[0], p[1] - new_sub_box_dim + temp[1], p[2] - new_sub_box_dim + temp[2] };
		register IntrTreeNode * const cell = newNode(pos_buf);
		/* Insert the new node at the new intermediate node */
		cell->treeNodeInsert(new_particle, new_sub_box_dim);
		/* Insert the current branch node at the new intermediate node */
		cell->treeNodeInsert((TreeLeaf *) (branch[i]), new_sub_box_dim);
		/* Set the current branch[i] to the new intermediate node that we just created */
		branch[i] = cell;
	}
}


static inline void computeRootPos(const int num_bodies, double &box_dim, double center_position[]) 
{
	/* Initialize the min and max position vectors to extreme values */
	double minPos[3];
	minPos[0] = 1.0E90;
	minPos[1] = 1.0E90;
	minPos[2] = 1.0E90;
	double maxPos[3];
	maxPos[0] = -1.0E90;
	maxPos[1] = -1.0E90;
	maxPos[2] = -1.0E90;
    
    /* loop through the all the bodies
     * compare each x, y, z and store
     * the max and min values for each coordinate
     * in the maxPos and minPos vector
     */
    for (int i = 0; i < num_bodies; ++i)
    {
//        std::cout << " x "<< particles[i]->p[0] <<" y " << particles[i]->p[1] << " z " << particles[i]->p[2] << std::endl;  
        if (minPos[0] > particles[i]->p[0])
            minPos[0] = particles[i]->p[0];
        if (minPos[1] > particles[i]->p[1])
            minPos[1] = particles[i]->p[1];
        if (minPos[2] > particles[i]->p[2])
            minPos[2] = particles[i]->p[2];
        if (maxPos[0] < particles[i]->p[0])
            maxPos[0] = particles[i]->p[0];
        if (maxPos[1] < particles[i]->p[1])
            maxPos[1] = particles[i]->p[1];
        if (maxPos[2] < particles[i]->p[2])
            maxPos[2] = particles[i]->p[2];
    }
    /* Maximum of the difference between 
     * max.X and min.X, max.Y and min.Y
     * and max.Z and min.Z 
     * is stored in the box_dim
     * The pass by reference box_dim variable helps determine
     * the bounding box of the particles 
     */
    box_dim = maxPos[0] - minPos[0];
    if (box_dim < (maxPos[1] - minPos[1]))
        box_dim = maxPos[1] - minPos[1];
    if (box_dim < (maxPos[2] - minPos[2]))
        box_dim = maxPos[2] - minPos[2];
       
    /* Calculate the center position as the average of
     * maxPos and minPos for each of the x, y and z 
     * coordinates and store it in a vector 
     */
    center_position[0] = (maxPos[0] + minPos[0]) / 2;
    center_position[1] = (maxPos[1] + minPos[1]) / 2;
    center_position[2] = (maxPos[2] + minPos[2]) / 2;
}


/* This function computes the center of mass
 * of the  tree upto a specified node. 
 * If the root is passed in as an argument
 * this function calculates the Center of Mass 
 * across the entire tree recursively. 
 */

void IntrTreeNode::calculateCM(int &current_node) // recursively summarizes info about subtrees
{
    /* initialize local buffer mass and position variables to 0 */
	register double m, px = 0.0, py = 0.0, pz = 0.0;
	/* create a buffer branch variable of the type tree node */
	register TreeNode *temp_branch;

    /* initialize var variable to 0 ... helps in cycling through 
     * the array index
     */	
	register int var = 0;
	/* this variable will store the total mass of the branch bodies */
	mass = 0.0;
	/* Cycle through each of the 8 children of the current node
     */
	for (int i = 0; i < 8; ++i) 
	{
		/* copy the current branch into the buffer */
        /* check to see if the current branch is NULL (empty) if so
         * then skip processing and move on to the next branch */
		temp_branch = branch[i];

		if (temp_branch != NULL) 
		{
			/* move the non null children to the front */
			branch[i] = NULL; // move non-NULL children to the front (needed to make other code faster)
			branch[var++] = temp_branch;
			/* if the current node is a leaf node ie contains a particle
             * then copy it into a new position in the particles[]
             * else it is an intermediate node, cycle through the 
             * branch nodes of the current intermediate node by recursively
             * calling the Compute Center of Mass function at the current 
             * intermediate node
             */
			if (temp_branch->node_type == PAR) 
			{ // Leaf Node
				particles[current_node++] = (TreeLeaf *) temp_branch; // sort particles in tree order (approximation of putting nearby nodes together for locality)
			} 
			else 
			{ // Intermediate Node
				((IntrTreeNode *) temp_branch)->calculateCM(current_node);
			}
			/* copy mass of the branch node into the temporary variable m */
			m = temp_branch->mass;
			/* Accumulate the mass of the current node into the variable
             * holding the total mass named "mass" */
			mass += m;
			/* Accumulate the position*mass of each branch coordinate 
             * and store the values in the temporary pos vector. 
             */
			px += temp_branch->p[0] * m;
			py += temp_branch->p[1] * m;
			pz += temp_branch->p[2] * m;
		}
	}
	/* Calculate center of mass for each intermediate node & root node
     * using the formula
     *                        ----- 
     *                 1      \  
     * CM_vector =  --------     m_i * pos_i
     *              tot_mass  /
     *                        -----
     */
	m = 1.0 / mass;
	p[0] = px * m;
	p[1] = py * m;
	p[2] = pz * m;
}


/* This function computes force on the particle from rest of the bodies.
 * The function receives root node and the box size as its arguments.
 */
void TreeLeaf::calculateForce(const IntrTreeNode * const root, const double box_size_buf) // computes the acceleration and velocity of a body
{
	/* create a buffer to store current acceleration */
	register double buf_acc[3];

	/* store the current body's acceleration into the buffer */
	for (int i = 0; i < 3; ++i)
		buf_acc[i] = a[i];
	/* zero out acceleration buffer*/ 
	a[0] = 0.0;
	a[1] = 0.0;
	a[2] = 0.0;

	/* call the recursive force calculator and pass in */ 
	forceCalcRe(root, box_size_buf * box_size_buf * inv_tolerance_2);

	if (iter > 0) 
	{
		v[0] += (a[0] - buf_acc[0]) * half_dt;
		v[1] += (a[1] - buf_acc[1]) * half_dt;
		v[2] += (a[2] - buf_acc[2]) * half_dt;
	}
}

void TreeLeaf::forceCalcRe(const TreeNode * const n, double box_size_2) // recursively walks the tree to compute the force on a body
{
	register double distance_r[3], distance_r_2, acceleration_factor, inv_distance_r;

	/* Calculating the distance r using
	 * x_cm - x, y_cm - y, z_cm - z 
	 */
	for (int i=0; i < 3; ++i)
		distance_r[i] = n->p[i] - p[i];

	/* r^2 = (x_cm - x)^2 + (y_cm - y)^2 + (z_cm - z)^2 */
	distance_r_2 = (distance_r[0]*distance_r[0]) + (distance_r[1]*distance_r[1]) + (distance_r[2]*distance_r[2]);

	/* Comparing r^2 < D^2  same as comparing D/r < theta=1
	 * if r^2 < D^2  it means D/r > Theta=1 : which means we expand the subtree and compute force impacted by each of the particles in the cell to the particle under consideration.
	 * if r^2 > D^2 it means D/r < Theta=1 : which means we use the center of mass of the box to compute force to the particle under consideration
	 **/
	if ((box_size_2/distance_r_2) < theta) 
	{
		if (n->node_type == CELL) 
		{
			register IntrTreeNode *temp_node = (IntrTreeNode *) n;
			box_size_2 *= 0.25;
			if (temp_node->branch[0] != NULL) 
			{
				forceCalcRe(temp_node->branch[0], box_size_2);
				if (temp_node->branch[1] != NULL) 
				{
					forceCalcRe(temp_node->branch[1], box_size_2);
					if (temp_node->branch[2] != NULL) 
					{
						forceCalcRe(temp_node->branch[2], box_size_2);
						if (temp_node->branch[3] != NULL) 
						{
							forceCalcRe(temp_node->branch[3], box_size_2);
							if (temp_node->branch[4] != NULL) 
							{
								forceCalcRe(temp_node->branch[4], box_size_2);
								if (temp_node->branch[5] != NULL) 
								{
									forceCalcRe(temp_node->branch[5], box_size_2);
									if (temp_node->branch[6] != NULL) 
									{
										forceCalcRe(temp_node->branch[6], box_size_2);
										if (temp_node->branch[7] != NULL) 
										{
											forceCalcRe(temp_node->branch[7], box_size_2);
										}
									}
								}
							}
						}
					}
				}
			}
		} 
		else 
		{ /* n is a body hence use its mass of the body to compute force on the particle under consideration */
			if (n != this) 
			{
				/* r^2 = r^2 + smoothing_factor^2 */
				distance_r_2 += softening_2;
				/* Calculate 1/r = 1/ (r)^0.5 */
				inv_distance_r = 1 / sqrt(distance_r_2);
				/* Calculating acceleration factor = M_cm / (R)^3 */
				acceleration_factor = n->mass * inv_distance_r * inv_distance_r * inv_distance_r;
				/* Calculating the acceleration vector for the current particle by aggregating to its 
				 * overall acceleration 
				 *    this.acc.x = r_x * acceleration factor
				 *	 this.acc.y = r_y * acceleration factor
				 *	 this.acc.z = r_z * acceleration factor
			     */ 
				for(int i=0; i<3; ++i)
					a[i] += distance_r[i] * acceleration_factor;
			}
		 }
	 } 
	 else 
	 { /* node is far enough away hence use center of mass of the box to compute force. */
		/* r^2 = r^2 + smoothing_factor^2 */
		distance_r_2 += softening_2;
		/* Calculate 1/r = 1/ (r)^0.5 */
		inv_distance_r = 1 / sqrt(distance_r_2);
		/* Calculating acceleration factor = M_cm / (R)^3 */
		acceleration_factor = n->mass * inv_distance_r * inv_distance_r * inv_distance_r;
		/* Calculating the acceleration vector for the current particle by aggregating to its 
		 * overall acceleration 
		 *    this.acc.x = r_x * acceleration factor
		 *	 this.acc.y = r_y * acceleration factor
		 *	 this.acc.z = r_z * acceleration factor
		 */ 
		for(int i=0; i<3; ++i)
			a[i] += distance_r[i] * acceleration_factor;
	 }
}

// REF: http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?bibcode=1995ApJ...443L..93H&db_key=AST&page_ind=0&data_type=GIF&type=SCREEN_VIEW&classic=YES 
// Move particle using leapfrog integrator / Verlet Integrator  
// r1 = r0 + (Vel_half * dt )
// v_3/2 = vel_half + (accel_1 * dt)
// advances a body's velocity and position by one time iter
void TreeLeaf::moveParticles() 
{
	double vel_dt_half [3];
	double v_half[3];
	/*Vel_t = dr/dt
	  acc_t = a[r_t] = dv/dt 
	  => dv_t = a[i] * dt 
	  => vel_dt/2 = a [i] * (dt/2) */
	for(int i=0; i < 3; ++i)
		vel_dt_half [i] = a[i] * half_dt;
	//vel_half = vel (t + (dt/2))
	//         = vel (t) + vel (dt/2) 
	for(int i=0; i < 3; ++i)
		v_half[i] = v[i] + vel_dt_half [i];  

	// r1 = r0 + (Vel_half * dt )
	for(int i=0; i < 3; ++i)
		p[i] += v_half[i] * dtime;  
	// v_3/2 = vel_half + (accel_1 * dt)
	for(int i=0; i < 3; ++i)
		v[i] = v_half[i] + vel_dt_half [i];  
}

static int num_bodies;     // number of particles in system
static int num_iterations; // number of time steps to run
static int grainSize;      // number of parallel tasks

static inline void processInput(string input_file) 
{
	/* 
     * Using C++ Filestream std library extensions
     * creating input file handle and output file handle
     */

	ifstream infile;
	infile.open(input_file.c_str());
	if (!infile)                                /* if there is a problem opening file */
	{                                           /* exit gracefully */
		cerr << "Can't open input file " << input_file << endl;
		exit (1);
	}

	infile >> num_bodies;                    /* Read value from first line of the file into num_bodies */
	infile >> num_iterations;                /* Read number of iterations */
	infile >> dtime;                         /* Integration timestep: It's convenient to use a timestep which has an exact representation as a floating-point number; for example, a value n/d, where n is an integer and d is a power of two. To simplify specification of such timesteps, the code accepts fractions as well as ordinary floating-point values. If dtime=0, treecode does a single force calculation, outputs the result, and exits. */
	infile >> eps;                           /* Density smoothing length used in the gravitational force calculation. In effect, the mass distribution is smoothed by replacing each body by a Plummer sphere with scale length eps, and the gravitational field of this smoothed distribution is calculated */ 
	infile >> tolerance;                  

	half_dt = 0.5 * dtime;
	softening_2 = (eps * eps);
	inv_tolerance_2 = 1.0 / (tolerance * tolerance);

	if (particles == NULL) 
	{
		/* create a n array container for particles each of type treeleaf */
		particles = new TreeLeaf*[num_bodies];
		/* cycle through the n array container and allocate memory space for each
         * particle */
		for (int i = 0; i < num_bodies; ++i)
			particles[i] = new TreeLeaf();
	}

	/* For each particle read data from file and 
     * populate the information in the particle datastructure 
     **/
	for(int i = 0; i < num_bodies ; ++i)
	{
		double dat[7] = {0,0,0,0,0,0,0};
		infile >> dat[0] >> dat[1] >> dat[2] >> dat[3] >> dat[4] >> dat[5] >> dat[6];
		particles[i]->mass = dat[0];
		particles[i]->p[0] = dat[1];
		particles[i]->p[1] = dat[2];
		particles[i]->p[2] = dat[3];
		particles[i]->v[0] = dat[4];
		particles[i]->v[1] = dat[5];
		particles[i]->v[2] = dat[6];
	}
	/* Close input file buffer */
	infile.close();
}

double chroner(timeval &t1, timeval &t2)
{
	double elapsedTime;
//	elapsedTime = (t2.tv_sec - t1.tv_sec);      // sec 
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	return elapsedTime;
}

static IntrTreeNode *local_root;
static double temp_box_size;

int main(int argc, char *argv[]) 
{
	// Time Measurement Variables and Related Calls
	timeval t1, t2;
	double iterChron[6];
	double progChron[6];
	double progompChron[8];

	// Variables for OpenMP 
	int nthreads, tid;

	// Parsing the commandline arguments
	// checking if the time step is passed in or not... if not then the 
	// program aborts gracefully
	// The program expects 4 arguments
	//  PROGRAM_NAME -i <input_file> -o <output_file>
	// If the user doesnot supply these arguments or a "-h" argument
	// The program aborts gracefully.
//	if(argc <= 4 && argv[1] != "-h")
        if(argc <= 4 && argv[1] != "-h")
        {
//              cout << "Usage: "<< argv[0] << " -i input_file -o output_file" << endl;
                cout << "Usage: "<< argv[0] << " -i input_file -c theta_val" << endl;
                exit (1);
        }

	/*
     * creating strings to store the passed in arguments into
     * input_file - stores the input file name from the parsed argv
     * output_file - stores the output file name from the parsed argv
     */
    string input_file;
//    string output_file;

    // Command Line parsing and processing
    int optind = 1;
    while((optind < argc) && (argv[optind][0] == '-'))
    {
        string sw = argv[optind];               /* store the current argv[i] into a string sw */ 
        if (sw == "-i")                         /* if the parsed string sw = -i */
        {                                       /* increment the counter by 1 */
            optind++;
            input_file = argv[optind];              /* store the input filename into ifname */ 
        }
        
        else if (sw == "-c")
        {
            optind++;
            theta = atof(argv[optind]);
        }        
//        else if (sw == "-o")                    /*  if the parsed string sw = -o */ 
//        {
//            optind++;                           /* increment the counter by 1 */
//            output_file = argv[optind];              /* store the output filename into ofname */
//        }
        else if (sw == "-h")                    /* if the parsed string sw = -h */
        {
            optind++;                           /* increment counter and print help information */
//            cout << "Usage: "<< argv[0] << " -i input_file -o output_file" << endl;
			cout << "Usage: "<< argv[0] << " -i input_file " << endl;
        }
        optind++;
    }

	particles = NULL;

	timeval start_time, end_time;
	register long run_time;

	gettimeofday(&t1, NULL);
	processInput(input_file);
	gettimeofday(&t2, NULL);
	progChron[0] = iterChron[0] = chroner(t1,t2);

	for (iter = 0; iter < num_iterations; ++iter) 
	{   // time-step the system
		/* declare the variables which will hold the dimensions of the
		 * box containing the particles and the array which will hold the 
		 * center coordinates of the box 
		 **/
		register double box_size, center_position[3];

		/* use the ComputeRootPos function to show calculate the 
		 * center coordinates and dimensions of the box. 
		 **/
		gettimeofday(&t1, NULL);
		computeRootPos(num_bodies, box_size, center_position);
		gettimeofday(&t2, NULL);
		iterChron[1] = chroner(t1,t2);

		/* Create the root node at the position determined above */
		IntrTreeNode *bht_root = IntrTreeNode::newNode(center_position); // create the tree's root

		const double sub_box_size = box_size * 0.5;
		gettimeofday(&t1, NULL);
		// Cycle through the particle array and insert each particle from 
		// the root position
		for (int i = 0; i < num_bodies; ++i) 
		{
			bht_root->treeNodeInsert(particles[i], sub_box_size); // grow the tree by inserting each body
		}
		gettimeofday(&t2, NULL);
		iterChron[2] = chroner(t1,t2);

		register int current_index = 0;

		gettimeofday(&t1, NULL);
		// Call a recursive compute mass function starting at the root node 
		bht_root->calculateCM(current_index); // summarize subtree info in each internal node (plus restructure tree and sort particles for performance reasons)
		gettimeofday(&t2, NULL);
		iterChron[3] = chroner(t1,t2);

		local_root = bht_root;
		temp_box_size = box_size;

		gettimeofday(&t1, NULL);
                
                double ompChron[8];
                timeval ot1, ot2;

#pragma omp parallel private (i, ot1, ot2, tid)
{
        
        gettimeofday(&ot1, NULL); 
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        
               
# pragma omp for
                
			for (int i = 0; i < num_bodies; ++i) 
			{
				particles[i]->calculateForce(local_root, temp_box_size);
			}
			
	gettimeofday(&ot2, NULL);
        ompChron[tid] = chroner(ot1,ot2);
}
		gettimeofday(&t2, NULL);
		iterChron[4] = chroner(t1,t2);

		IntrTreeNode::treeReuse(); // recycle the tree
		gettimeofday(&t1, NULL);
		for (int i = 0; i < num_bodies; ++i) 
		{ 
			particles[i]->moveParticles(); 
		}
		gettimeofday(&t2, NULL);
		iterChron[5] = chroner(t1,t2);
		/// following format of cout's is required to be able to visualize the output
		//                    NUM_Bodies
		//                    Iteration / timestep 
		//                    N Bodies position and velocity
		cout << num_bodies << endl;
		cout << iter << endl;
		//for (int i = 0; i < num_bodies; ++i) 
		//	cout <<particles[i]->p[0] <<" " << particles[i]->p[1] << " " << particles[i]->p[2] <<" " <<particles[i]->v[0] <<" " << particles[i]->v[1] << " " << particles[i]->v[2] <<endl; 
		// update program timers
		for (int i = 1; i < 6; ++i)
			progChron[i] += iterChron[i];
                
		for (int i = 0; i < 8; ++i)
			progompChron[i] += ompChron[i];
		cerr << "iter :  FileRead   RootPos    TreeCreate    CenterMass      ForceCalc      MovePart    " << endl;
		cerr << iter<<" :" << iterChron[0] << "\t    " << iterChron[1] << "  \t" << iterChron[2] << "\t\t" << iterChron[3] \
			<< "\t\t" << iterChron[4] << "\t\t" << iterChron[5] <<endl;
                cerr << "Summary:  T0 " << ompChron[0] << " T1 : "  << ompChron[1] << " T2 : " << ompChron[2] << " T3 " << ompChron[3] << " T4 " << ompChron[4] << " T5 " << ompChron[5]<< std::endl;
                cerr << "          T6 " << ompChron[6] << " T7 : " << ompChron[7] << std::endl;
	} // end of time step

	cerr << "Summary:  FileRead   RootPos    TreeCreate    CenterMass      ForceCalc      MovePart    " << endl;
	cerr << "Summary:  " << progChron[0] << "\t    " << progChron[1] << " \t" << progChron[2] << "\t\t" << progChron[3] \
		<< "\t\t" << progChron[4] << "\t\t" << progChron[5] <<endl;

        cerr << "OpenMP:  T0 " << progompChron[0] << " T1 : "  << progompChron[1] << " T2 : " << progompChron[2] << " T3 " << progompChron[3] << " T4 " << progompChron[4] << " T5 " << progompChron[5]<< std::endl;
        cerr << "          T6 " << progompChron[6] << " T7 : " << progompChron[7] << std::endl;

	return 0;
}
