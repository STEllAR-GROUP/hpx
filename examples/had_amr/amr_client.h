
#if defined(__cplusplus)
extern "C" {
#endif

struct Par{
  double lambda;
  int allowedl;
  int loglevel;
  int stencilsize;
  int nx0;
  int nt0;
  double minx0;
  double maxx0;
  double dx0;
  double dt0;
};

#if defined(__cplusplus)
}
#endif
