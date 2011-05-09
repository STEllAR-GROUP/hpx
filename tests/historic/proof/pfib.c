/*  Copyright (c) 2009 Steven Brandt
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying 
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <assert.h>

typedef void *(*pfunc)(void*);

pthread_attr_t attr, *attrp = 0;

struct fib_data {
    int in,out;
};

struct fib_data *fib(struct fib_data *fd) {
    if(fd->in < 2) {
        fd->out = fd->in;
    } else {
        pthread_t t1,t2;
        struct fib_data fd1, fd2;
        fd1.in = fd->in-1;
        fd2.in = fd->in-2;
        pthread_create(&t1,attrp,(pfunc)fib,(void*)&fd1);
        pthread_create(&t2,attrp,(pfunc)fib,(void*)&fd2);
        pthread_join(t1,NULL);
        pthread_join(t2,NULL);
        fd->out = fd1.out + fd2.out;
    }
    return NULL;
}

int main(int argc,char **argv) {
    struct fib_data fd;
    struct timespec ts1, ts2;
    double t;
    size_t ssz = PTHREAD_STACK_MIN;
    int i;

    fd.in = 22;
    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2])
	{
            if (argv[i][1] == 's')
	    {
	        pthread_attr_init(&attr);
		if (pthread_attr_setstacksize(&attr, ssz))
		{
		    attrp = 0;
		    printf("Warning: setting stack size attribute failed\n");
		}
		else attrp = &attr;
	    }
	    else if (argv[i][1], 'v')
	    {
	        fd.in = atoi(argv[++i]);
	    }
	    else
	    {
	        printf("Error: unknown option: %s\n", argv[i]);
		exit(2);
	    }
	}
    }
    clock_gettime(CLOCK_REALTIME, &ts1);
    fib(&fd);
    clock_gettime(CLOCK_REALTIME, &ts2);
    t = ts2.tv_sec-ts1.tv_sec+1e-9*(ts2.tv_nsec-ts1.tv_nsec);
    printf("f=%d, time=%f\n",fd.out,t);
    return 0;
}
