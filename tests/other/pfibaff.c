#define _GNU_SOURCE

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <limits.h>
#include <assert.h>

typedef void *(*pfunc)(void*);

#define MAX_CPUS 4

pthread_attr_t attr, *attrp = 0;
int cpus = 0;
cpu_set_t cstab[MAX_CPUS];


struct fib_data {
    int in,out;
};

struct fib_data *fib(struct fib_data *fd) {
  if (cpus)
  {
    int mycpu = fd->in%cpus;
    sched_setaffinity(0, sizeof(cpu_set_t), &cstab[mycpu]);
  }
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
  size_t ssz = PTHREAD_STACK_MIN;
  double t;
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
      else if (argv[i][1] == 'v')
      {
	fd.in = atoi(argv[++i]);
      }
      else if (argv[i][1] == 't')
      {
	int k;
	cpus = atoi(argv[++i]);
	if (cpus < 0 || cpus > MAX_CPUS)
	{
	  printf("Error: number of cpus must be positive integer <=%d\n", MAX_CPUS);
	  exit(2);
	}
	for (k = 0; k < cpus; k++)
	{
	  CPU_ZERO(&cstab[k]); CPU_SET(k, &cstab[k]);
	}
      }
      else
      {
	printf("Error: unknown option: %s\n", argv[i]);
	exit(2);
      }
    }
    else
    {
      printf("Error: unexpeted argument: %s\n", argv[i]);
    }
  }
  clock_gettime(CLOCK_REALTIME, &ts1);
  fib(&fd);
  clock_gettime(CLOCK_REALTIME, &ts2);
  t = ts2.tv_sec-ts1.tv_sec+1e-9*(ts2.tv_nsec-ts1.tv_nsec);
  printf("f=%d, time=%f\n",fd.out,t);
  return 0;
}
