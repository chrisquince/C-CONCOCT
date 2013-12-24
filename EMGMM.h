#ifndef NMGS_H
#define NMGS_H

typedef struct s_Params
{
 /*seed*/
  unsigned long int lSeed;
  /*csv input file*/
  char *szInputFile;
  /*output file stub*/
  char *szOutFileStub;
  /*start cluster size*/
  int nKStart;
  /*end cluster size*/
  int nKEnd;
  /*min contig length*/
  int nLMin;
} t_Params;


typedef struct s_Data
{
  int nN;

  int nD;

  double **aadX;

  char **aszDimNames;

  char **aszSampleNames;
} t_Data;

typedef struct s_Cluster
{
  /*start seed*/
  unsigned long lSeed;
  /*thread index*/
  int nThread;
  /*pointer to data*/
  t_Data *ptData;
  /*number of data points*/
  int nN;
  /*number of clusters*/
  int nK;
  /*number of dimensions*/
  int nD;
  /*log likelihood*/
  double dLL;
  /*Bayesian information criterion*/
  double dBIC;
  /*Means*/
  double **aadMu;
  /*Inverse variances*/
  gsl_matrix **aptSigma;
  /*Responsibilities*/
  double **aadZ;
  /*log-Matrix determinants*/
  double *adLDet;
  /*mixture weights*/
  double *adPi;
  /*assigned cluster for each data point*/
  int *anMaxZ;
  /*frequencies for each cluster*/
  int *anW;
} t_Cluster;


#define DELIM ",\n"
#define MAX_LINE_LENGTH   10000
#define MAX_WORD_LENGTH   128

#define TRUE  1
#define FALSE 0

#define NOT_SET -1
#define OPTION  0      /* optional */
#define ALWAYS  1      /* required */

/*Default parameters*/
#define DEF_KSTART       2
#define DEF_KEND         32
#define MAX_ITER         100
/*Algorithm constants*/
#define MIN_CHANGE_LL    0.005
#define MIN_COVAR        0.001

#define N_RESTARTS       8

#define N_RTHREADS       5
#define N_KTHREADS       8

#define K_PRIME          100003
#define R_PRIME          1009

#define DEF_SEED         1
#define DEF_LMIN         1000

#define OUT_FILE_STUB    "-out"
#define INPUT_FILE       "-in"
#define SEED       	 "-l"
#define VERBOSE          "-v"
#define KSTART           "-ks"
#define KEND             "-ke"
#define LMIN             "-lm"

void getCommandLineParams(t_Params *ptParams,int argc,char *argv[]);

void readInputData(const char *szFile, t_Data *ptData);

void destroyData(t_Data *ptData);

void allocateCluster(t_Cluster *ptCluster, int nN, int nK, int nD, t_Data *ptData, long lSeed);

void performMStep(t_Cluster *ptCluster, t_Data *ptData);

void updateMeans(t_Cluster *ptCluster, t_Data *ptData);

void gmmTrainEM(t_Cluster *ptCluster, t_Data *ptData);

void initRandom(gsl_rng *ptGSLRNG, t_Cluster *ptCluster, t_Data *ptData);

void initKMeans(gsl_rng *ptGSLRNG, t_Cluster *ptCluster, t_Data *ptData);

double calcDist(double* adX, double *adMu, int nD);

double calcLNP(t_Cluster* ptCluster, double *adX, double* adZ);

void writeClusters(char *szOutFile, t_Cluster *ptCluster, t_Data *ptData);

void destroyCluster(t_Cluster* ptCluster);

void* fitEM(void *pvCluster);

void* runRThreads(void *pvpDCluster);

void writeMeans(char *szOutFile, t_Cluster *ptCluster);

void writeSquareMatrix(char*szFile, gsl_matrix *ptMatrix, int nD);

void calcCovarMatrices(t_Cluster *ptCluster, t_Data *ptData);

#endif
