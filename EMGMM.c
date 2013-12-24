/*System includes*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <float.h>

/*GSL includes*/
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cblas.h>
#include <pthread.h>

/*User includes*/
#include "EMGMM.h"

static char *usage[] = {"EMGMM - Fits Gaussian mixture model using EM algorithm\n",
                        "Required parameters:\n",
			"\t-in\tfilename\tcsv file\n",
			"\t-out\toutfilestub\n",
                        "Optional:\n",	
			"\t-l\tinteger\tseed\n",
			"\t-lm\tinteger\tmin contig length\n",
			"\t-ks\tinteger\tstart cluster number k>=ks\n",
			"\t-ke\tinteger\tend cluster number k>=ke\n",
			"\t-v\t\tverbose\n"};

static int nLines   = 10;

static int bVerbose = FALSE;

int main(int argc, char* argv[])
{
  t_Params           tParams;
  t_Data             tData;
  gsl_rng            *ptGSLRNG     = NULL;
  const gsl_rng_type *ptGSLRNGType = NULL;
  int i = 0, j = 0, k = 0, r = 0, nK = 0, nNK = 0, nD = 0, nN = 0;
  t_Cluster **aptBestK = NULL;
  char szOFile[MAX_LINE_LENGTH];
  FILE *ofp = NULL;
  double dBestBIC = DBL_MAX;
  int    nKBest   = -1;

  /*initialise GSL RNG*/
  gsl_rng_env_setup();

  gsl_set_error_handler_off();
  
  ptGSLRNGType = gsl_rng_default;
  ptGSLRNG     = gsl_rng_alloc(ptGSLRNGType);

  gsl_set_error_handler_off();
  
  /*get command line params*/
  getCommandLineParams(&tParams, argc, argv);

  /*read in input data*/
  readInputData(tParams.szInputFile, &tData);

  nD = tData.nD;
  nN = tData.nN;
  nNK = tParams.nKEnd - tParams.nKStart;

  aptBestK = (t_Cluster **) malloc(nNK*sizeof(t_Cluster*));
  if(!aptBestK)
    goto memoryError;

  for(nK = tParams.nKStart; nK < tParams.nKEnd; nK++){
    aptBestK[nK - tParams.nKStart] = (t_Cluster *) malloc(sizeof(t_Cluster));
  }


  for(nK = tParams.nKStart; nK < tParams.nKEnd; nK += N_KTHREADS){
    t_Cluster atDCluster[N_KTHREADS];
    pthread_t atKThreads[N_KTHREADS];    
    int       kret[N_KTHREADS];
    int       nKO = nK - tParams.nKStart;
    int       nKR = tParams.nKEnd - nK;
    int       nNKThreads =  nKR < N_KTHREADS ? nKR : N_KTHREADS;

    for(k = 0; k < nNKThreads; k++){
      aptBestK[nKO + k]->nN = nN;
      aptBestK[nKO + k]->nK = nK + k;
      aptBestK[nKO + k]->nD = nD;
      aptBestK[nKO + k]->ptData = &tData;
      aptBestK[nKO + k]->lSeed = tParams.lSeed + k*K_PRIME;

      kret[k] = pthread_create(&atKThreads[k], NULL, runRThreads, (void*) &aptBestK[nKO + k]);
    }

    for(k = 0; k < nNKThreads; k++){
      pthread_join(atKThreads[k], NULL);
    }
    
  }


  mkdir(tParams.szOutFileStub,S_IRWXU);

  sprintf(szOFile,"%s/%s_bic.csv",tParams.szOutFileStub,tParams.szOutFileStub,tParams.szOutFileStub);

  ofp = fopen(szOFile,"w");
  if(ofp){
    for(nK = tParams.nKStart; nK < tParams.nKEnd; nK++){
      double dBIC = aptBestK[nK - tParams.nKStart]->dBIC;
      fprintf(ofp,"%d,%f\n",nK,dBIC);
      if(dBIC < dBestBIC){
	dBestBIC = dBIC;
	nKBest = nK;
      }
    }
    fclose(ofp);
  }
  else{
    fprintf(stderr,"Failed openining %s in main\n", szOFile);
    fflush(stderr);
  }

  sprintf(szOFile,"%s/%s_clustering_gt%d.csv",tParams.szOutFileStub,tParams.szOutFileStub,tParams.nLMin);
  writeClusters(szOFile,aptBestK[nKBest - tParams.nKStart],&tData);

  sprintf(szOFile,"%s/%s_pca_means_gt%d.csv",tParams.szOutFileStub,tParams.szOutFileStub,tParams.nLMin);
  writeMeans(szOFile,aptBestK[nKBest - tParams.nKStart]);
  
  calcCovarMatrices(aptBestK[nKBest - tParams.nKStart], &tData);

  for(k = 0; k < nKBest; k++){
    sprintf(szOFile,"%s/%s_pca_variances_gt%d_dim%d.csv",tParams.szOutFileStub,tParams.szOutFileStub,tParams.nLMin,k);
    
    writeSquareMatrix(szOFile, aptBestK[nKBest - tParams.nKStart]->aptSigma[k], nD);
  }

  /*free up memory in data object*/
  destroyData(&tData);

  /*free up best BIC clusters*/
  for(nK = tParams.nKStart; nK < tParams.nKEnd; nK++){
    destroyCluster(aptBestK[nK - tParams.nKStart]);
    free(aptBestK[nK - tParams.nKStart]);
  }

  free(aptBestK);
  exit(EXIT_SUCCESS);

 memoryError:
  fprintf(stderr, "Failed allocating memory in main\n");
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void writeUsage(FILE* ofp)
{
  int i = 0;
  char *line;

  for(i = 0; i < nLines; i++){
    line = usage[i];
    fputs(line,ofp);
  }
}

char *extractParameter(int argc, char **argv, char *param,int when)
{
  int i = 0;

  while((i < argc) && (strcmp(param,argv[i]))){
    i++;
  }

  if(i < argc - 1){
    return(argv[i + 1]);
  }

  if((i == argc - 1) && (when == OPTION)){
    return "";
  }

  if(when == ALWAYS){
    fprintf(stdout,"Can't find asked option %s\n",param);
  }

  return (char *) NULL;
}

void getCommandLineParams(t_Params *ptParams,int argc,char *argv[])
{
  char *szTemp = NULL;
  char *cError = NULL;

  /*get parameter file name*/
  ptParams->szInputFile  = extractParameter(argc,argv, INPUT_FILE,ALWAYS);  
  if(ptParams->szInputFile == NULL)
    goto error;
 
  /*get parameter file name*/
  ptParams->szOutFileStub  = extractParameter(argc,argv,OUT_FILE_STUB,ALWAYS);  
  if(ptParams->szOutFileStub == NULL)
    goto error;

  szTemp = extractParameter(argc,argv,VERBOSE,OPTION);
  if(szTemp != NULL){
    bVerbose=TRUE;
  }

  szTemp = extractParameter(argc,argv,SEED,OPTION);
  if(szTemp != NULL){
    ptParams->lSeed = (unsigned long int) strtol(szTemp,&cError,10);
    if(*cError != '\0'){
      goto error;
    }
  }
  else{
    ptParams->lSeed = (unsigned long int) DEF_SEED;
  }

  szTemp = extractParameter(argc,argv,LMIN,OPTION);
  if(szTemp != NULL){
    ptParams->nLMin = (int) strtol(szTemp,&cError,10);
    if(*cError != '\0'){
      goto error;
    }
  }
  else{
    ptParams->nLMin = DEF_LMIN;
  }

  szTemp = extractParameter(argc,argv,KSTART,OPTION);
  if(szTemp != NULL){
    ptParams->nKStart = strtol(szTemp,&cError,10);
    if(*cError != '\0'){
      goto error;
    }
  }
  else{
    ptParams->nKStart = DEF_KSTART;
  }

  szTemp = extractParameter(argc,argv,KEND,OPTION);
  if(szTemp != NULL){
    ptParams->nKEnd = strtol(szTemp,&cError,10);
    if(*cError != '\0'){
      goto error;
    }
  }
  else{
    ptParams->nKEnd = DEF_KEND;
  }

  return;

 error:
  writeUsage(stdout);
  exit(EXIT_FAILURE);
}

void readInputData(const char *szFile, t_Data *ptData)
{
  double  **aadX = NULL;
  int  i = 0, j = 0, nD = 0, nN = 0;
  char szLine[MAX_LINE_LENGTH];
  FILE* ifp = NULL;

  ifp = fopen(szFile, "r");

  if(ifp){
    char* szTok   = NULL;
    char* pcError = NULL;

    fgets(szLine, MAX_LINE_LENGTH, ifp);
    szTok = strtok(szLine, DELIM);
    /*count dimensions*/
    while(strtok(NULL, DELIM) != NULL){
      
      nD++;
    }
    /*count data points*/
    while(fgets(szLine, MAX_LINE_LENGTH, ifp) != NULL){
    	nN++;
    }
    fclose(ifp);

    /*reopen input file*/
    ifp = fopen(szFile, "r");	
    fgets(szLine, MAX_LINE_LENGTH, ifp);

    /*allocate memory for dimension names*/
    ptData->aszDimNames = (char **) malloc(nD*sizeof(char*));
    if(!ptData->aszDimNames)
      goto memoryError;

    szTok = strtok(szLine, DELIM);
    /*read in dim names*/
    for(i = 0; i < nD; i++){
      szTok = strtok(NULL, DELIM);
      ptData->aszDimNames[i] = strdup(szTok);
    }
	
    /*allocate memory for data matrix*/
    aadX = (double **) malloc(nN*sizeof(double*));
    if(!aadX)
      goto memoryError;
    for(i = 0; i < nN; i++){
      aadX[i] = (double *) malloc(nD*sizeof(double));
      if(!aadX[i])
	goto memoryError;
    }

    /*read in input data*/
    ptData->aszSampleNames = (char **) malloc(nN*sizeof(char*));
    if(!ptData->aszSampleNames)
      goto memoryError;

    for(i = 0; i < nN; i++){
    
      fgets(szLine, MAX_LINE_LENGTH, ifp);
      szTok = strtok(szLine, DELIM);
      ptData->aszSampleNames[i] = strdup(szTok);
      for(j = 0; j < nD; j++){
	szTok = strtok(NULL, DELIM);

	aadX[i][j] = strtod(szTok,&pcError);

	if(*pcError != '\0'){
	  goto formatError;
	}
      }
    }
  }
  else{
    fprintf(stderr, "Failed to open abundance data file %s aborting\n", szFile);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }

  ptData->nD = nD;
  ptData->nN = nN;
  ptData->aadX = aadX;
  return;

 memoryError:
  fprintf(stderr, "Failed allocating memory in readInputData\n");
  fflush(stderr);
  exit(EXIT_FAILURE);

 formatError:
  fprintf(stderr, "Incorrectly formatted abundance data file\n");
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void destroyData(t_Data *ptData)
{
  int nN = ptData->nN, nD = ptData->nD;
  int i = 0;

  for(i = 0; i < nD; i++){
    free(ptData->aszDimNames[i]);
  }
  free(ptData->aszDimNames);

  for(i = 0; i < nN; i++){
    free(ptData->aadX[i]);
  }
  free(ptData->aadX);

  for(i = 0; i < nN; i++){
    free(ptData->aszSampleNames[i]);
  }
  free(ptData->aszSampleNames);

  return;

}

void destroyCluster(t_Cluster* ptCluster)
{
  int i = 0, nN = ptCluster->nN, nK = ptCluster->nK;

  free(ptCluster->anMaxZ);

  free(ptCluster->anW); 

  for(i = 0; i < nN; i++){
    free(ptCluster->aadZ[i]); 
  }
  free(ptCluster->aadZ);

  free(ptCluster->adLDet); 
  free(ptCluster->adPi);  


  for(i = 0; i < nK; i++){
    free(ptCluster->aadMu[i]); 
  }
  free(ptCluster->aadMu);

  for(i = 0; i < nK ; i++){
    gsl_matrix_free(ptCluster->aptSigma[i]);
  }
  free(ptCluster->aptSigma);

  return;
 
 memoryError:
  fprintf(stderr, "Failed allocating memory in allocateCluster\n");
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void allocateCluster(t_Cluster *ptCluster, int nN, int nK, int nD, t_Data *ptData, long lSeed)
{
  int i = 0, j = 0, k = 0;

  ptCluster->lSeed = lSeed;
  ptCluster->ptData = ptData;

  ptCluster->nN = nN;
  ptCluster->nK = nK;
  ptCluster->nD = nD;

  ptCluster->dLL = 0.0;
  ptCluster->dBIC = 0.0;

  ptCluster->anMaxZ = (int *) malloc(nN*sizeof(int));
  if(!ptCluster->anMaxZ)
    goto memoryError;

  ptCluster->anW = (int *) malloc(nN*sizeof(int));
  if(!ptCluster->anW)
    goto memoryError;

  for(i = 0; i < nN; i++){
    ptCluster->anMaxZ[i] = NOT_SET;
  }

  for(i = 0; i < nK; i++){
    ptCluster->anW[i] = 0;
  }

  ptCluster->aadZ = (double **) malloc(nN*sizeof(double *));
  if(!ptCluster->aadZ)
    goto memoryError;

  for(i = 0; i < nN; i++){
    ptCluster->aadZ[i] = (double *) malloc(nK*sizeof(double));
    if(!ptCluster->aadZ[i])
      goto memoryError;

    for(j = 0; j < nK; j++){
      ptCluster->aadZ[i][j] = 0.0;
    }
  }

  ptCluster->adLDet = (double *) malloc(nK*sizeof(double));
  ptCluster->adPi   = (double *) malloc(nK*sizeof(double));

  if(!ptCluster->adLDet || !ptCluster->adPi)
    goto memoryError;

  for(k = 0; k < nK; k++){
    ptCluster->adLDet[k] = 0.0;
    ptCluster->adPi[k] = 0.0;
  }

  ptCluster->aadMu = (double **) malloc(nK*sizeof(double *));
  if(!ptCluster->aadMu)
    goto memoryError;

  for(i = 0; i < nK; i++){
    ptCluster->aadMu[i] = (double*) malloc (nD*sizeof(double));
    if(!ptCluster->aadMu[i])
      goto memoryError;
  }

  ptCluster->aptSigma = (gsl_matrix **) malloc(nK*sizeof(gsl_matrix *));
  if(!ptCluster->aptSigma)
    goto memoryError;

  for(i = 0; i < nK ; i++){
    ptCluster->aptSigma[i] = (gsl_matrix*) gsl_matrix_alloc (nD, nD);
  }

  return;
 
 memoryError:
  fprintf(stderr, "Failed allocating memory in allocateCluster\n");
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void writeSquareMatrix(char*szFile, gsl_matrix *ptMatrix, int nD)
{
  int i = 0, j = 0;
  FILE* ofp = fopen(szFile,"w");

  if(ofp){
    for(i = 0; i < nD; i++){
      for(j = 0; j < nD - 1; j++){
	fprintf(ofp,"%f,",gsl_matrix_get(ptMatrix,i,j));
      }
      fprintf(ofp,"%f\n",gsl_matrix_get(ptMatrix,i,j));
    }
  }
  else{
    fprintf(stderr,"Failed to open %s for writing in writeSquareMatrix\n", szFile);
    fflush(stderr);
  }
}

double decomposeMatrix(gsl_matrix *ptSigmaMatrix, int nD)
{
  double dDet = 0.0;
  int status;
  int l = 0;

  status = gsl_linalg_cholesky_decomp(ptSigmaMatrix);

  if(status == GSL_EDOM){
    fprintf(stderr,"Failed Cholesky decomposition in decomposeMatrix\n");
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
  else{
    for(l = 0; l < nD; l++){
      double dT = gsl_matrix_get(ptSigmaMatrix,l,l);
      dDet += 2.0*log(dT);
    }
    gsl_linalg_cholesky_invert(ptSigmaMatrix);
    return dDet;
  }
}

void calcCovarMatrices(t_Cluster *ptCluster, t_Data *ptData)
{
  int i = 0, j = 0, k = 0, l = 0, m = 0;
  int nN = ptData->nN, nK = ptCluster->nK, nD = ptData->nD;
  double **aadZ = ptCluster->aadZ,**aadX = ptData->aadX;
  double *adLDet = ptCluster->adLDet, *adPi = ptCluster->adPi;
  double **aadCovar = NULL;
  double dN = (double) nN;
  int    status;
  //gsl_matrix* ptSigmaMatrix = gsl_matrix_alloc(nD,nD);

  aadCovar = (double **) malloc(nD*sizeof(double*));
  if(!aadCovar)
    goto memoryError;

  for(i = 0; i < nD; i++){
    aadCovar[i] = (double *) malloc(nD*sizeof(double));
    if(!aadCovar[i])
      goto memoryError;
  }


  for(k = 0; k < nK; k++){ /*loop components*/    
    double*     adMu          = ptCluster->aadMu[k];
    gsl_matrix  *ptSigmaMatrix = ptCluster->aptSigma[k];
    /*recompute mixture weights and means*/
    for(j = 0; j < nD; j++){
      adMu[j] = 0.0;
      for(l = 0; l < nD; l++){
	aadCovar[j][l] = 0.0;
      }
      /*prevents singularities*/
      aadCovar[j][j] = MIN_COVAR;
    }

    /* compute weight associated with component k*/
    adPi[k] = 0.0;
    for(i = 0; i < nN; i++){
      adPi[k] += aadZ[i][k];
      for(j = 0; j < nD; j++){
	adMu[j] += aadZ[i][k]*aadX[i][j];
      }
    }
    /*normalise means*/
    for(j = 0; j < nD; j++){
      adMu[j] /= adPi[k];
    }
     
    /*calculate covariance matrices*/
    for(i = 0; i < nN; i++){
      double adDiff[nD];
      
      for(j = 0; j < nD; j++){
	adDiff[j] = aadX[i][j] - adMu[j];
      }

      for(l = 0; l < nD; l++){
	for(m = 0; m <=l ; m++){
	  aadCovar[l][m] += aadZ[i][k]*adDiff[l]*adDiff[m];
	}
      } 
    }
    
    for(l = 0; l < nD; l++){
      for(m = l + 1; m < nD; m++){
	aadCovar[l][m] = aadCovar[m][l];
      }
    }

    for(l = 0; l < nD; l++){
      for(m = 0; m < nD; m++){
	aadCovar[l][m] /= adPi[k];
	gsl_matrix_set(ptSigmaMatrix, l, m, aadCovar[l][m]);
      }
    }

    adPi[k] /= dN; /*normalise weights*/    
  }
  /*free up memory*/
  for(i = 0; i < nD; i++){
    free(aadCovar[i]);
  }

  //gsl_matrix_free(ptSigmaMatrix);
  free(aadCovar);

  return;
 memoryError:
  fprintf(stderr, "Failed allocating memory in performMStep\n");
  fflush(stderr);
  exit(EXIT_FAILURE); 
}

void performMStep(t_Cluster *ptCluster, t_Data *ptData){
  int i = 0, j = 0, k = 0, l = 0, m = 0;
  int nN = ptData->nN, nK = ptCluster->nK, nD = ptData->nD;
  double **aadZ = ptCluster->aadZ,**aadX = ptData->aadX;
  double *adLDet = ptCluster->adLDet, *adPi = ptCluster->adPi;
  double **aadCovar = NULL;
  double dN = (double) nN;
  int    status;
  //gsl_matrix* ptSigmaMatrix = gsl_matrix_alloc(nD,nD);

  aadCovar = (double **) malloc(nD*sizeof(double*));
  if(!aadCovar)
    goto memoryError;

  for(i = 0; i < nD; i++){
    aadCovar[i] = (double *) malloc(nD*sizeof(double));
    if(!aadCovar[i])
      goto memoryError;
  }
  /*perform M step*/
  for(k = 0; k < nK; k++){ /*loop components*/    
    double*     adMu          = ptCluster->aadMu[k];
    gsl_matrix  *ptSigmaMatrix = ptCluster->aptSigma[k];
    /*recompute mixture weights and means*/
    for(j = 0; j < nD; j++){
      adMu[j] = 0.0;
      for(l = 0; l < nD; l++){
	aadCovar[j][l] = 0.0;
      }
      /*prevents singularities*/
      aadCovar[j][j] = MIN_COVAR;
    }

    /* compute weight associated with component k*/
    adPi[k] = 0.0;
    for(i = 0; i < nN; i++){
      adPi[k] += aadZ[i][k];
      for(j = 0; j < nD; j++){
	adMu[j] += aadZ[i][k]*aadX[i][j];
      }
    }
    /*normalise means*/
    for(j = 0; j < nD; j++){
      adMu[j] /= adPi[k];
    }
     
    /*calculate covariance matrices*/
    for(i = 0; i < nN; i++){
      double adDiff[nD];
      
      for(j = 0; j < nD; j++){
	adDiff[j] = aadX[i][j] - adMu[j];
      }

      for(l = 0; l < nD; l++){
	for(m = 0; m <=l ; m++){
	  aadCovar[l][m] += aadZ[i][k]*adDiff[l]*adDiff[m];
	}
      } 
    }
    
    for(l = 0; l < nD; l++){
      for(m = l + 1; m < nD; m++){
	aadCovar[l][m] = aadCovar[m][l];
      }
    }

    for(l = 0; l < nD; l++){
      for(m = 0; m < nD; m++){
	aadCovar[l][m] /= adPi[k];
	gsl_matrix_set(ptSigmaMatrix, l, m, aadCovar[l][m]);
      }
    }

    if(nK == -1){
      printf("nK = %d k = %d\n",nK, k);
      for(l = 0; l < nD; l++){
	for(m = l; m < l+1; m++){
	  printf("%d %d %f\n",l,m,aadCovar[l][m]);
	}
	printf("\n");
      }
    }
    //printf("zik = %f\n",adPi[k]);
    adPi[k] /= dN; /*normalise weights*/
    
    adLDet[k] = decomposeMatrix(ptSigmaMatrix,nD);
  }

  /*free up memory*/
  for(i = 0; i < nD; i++){
    free(aadCovar[i]);
  }

  //gsl_matrix_free(ptSigmaMatrix);
  free(aadCovar);

  return;
 memoryError:
  fprintf(stderr, "Failed allocating memory in performMStep\n");
  fflush(stderr);
  exit(EXIT_FAILURE);  
}

void updateMeans(t_Cluster *ptCluster, t_Data *ptData){
  int i = 0, j = 0, k = 0, l = 0, m = 0;
  int nN = ptData->nN, nK = ptCluster->nK, nD = ptData->nD;
  int *anMaxZ = ptCluster->anMaxZ;
  int *anW    = ptCluster->anW;
  double **aadX = ptData->aadX, **aadMu = ptCluster->aadMu;
  double *adPi = ptCluster->adPi, dN = (double) nN;

  for(k = 0; k < nK; k++){
    
    for(j = 0; j < nD; j++){
      aadMu[k][j] = 0.0;
    }
  }

  for(i = 0; i < nN; i++){
    int nZ = anMaxZ[i];

    for(j = 0; j < nD; j++){
      aadMu[nZ][j] += aadX[i][j];
    }
  }

  for(k = 0; k < nK; k++){ /*loop components*/
    
    /*normalise means*/
    if(anW[k] > 0){
      for(j = 0; j < nD; j++){
	aadMu[k][j] /= (double) anW[k];
      }
    }
    else{
      for(j = 0; j < nD; j++){
	aadMu[k][j] = 0.0;
      }
    }
  }

  return;
}

void initRandom(gsl_rng *ptGSLRNG, t_Cluster *ptCluster, t_Data *ptData)
{
  /*very simple initialisation assign each data point to random cluster*/
  int i = 0, k = 0;

  for(i = 0; i < ptData->nN; i++){
    int nIK = -1;

    for(k = 0; k < ptCluster->nK; k++){
      ptCluster->aadZ[i][k] = 0.0;
    }

    nIK =  gsl_rng_uniform_int (ptGSLRNG, ptCluster->nK);

    ptCluster->aadZ[i][nIK] = 1.0;
  }
  
  performMStep(ptCluster, ptData);
  
  return;
}

double calcDist(double* adX, double *adMu, int nD)
{
  double dDist = 0.0;
  int i = 0;
  
  for(i = 0; i < nD; i++){
    double dV = adX[i] - adMu[i];
    dDist += dV*dV;
  }
  
  return sqrt(dDist);
}

void initKMeans(gsl_rng *ptGSLRNG, t_Cluster *ptCluster, t_Data *ptData)
{
  /*very simple initialisation assign each data point to random cluster*/
  int i = 0, k = 0, nN = ptData->nN, nK = ptCluster->nK, nD = ptData->nD;
  double **aadMu = ptCluster->aadMu, **aadX = ptData->aadX, *adPi = ptCluster->adPi; 
  int *anMaxZ = ptCluster->anMaxZ, *anW = ptCluster->anW, nChange = nN;
  int nIter = 0;
  for(i = 0; i < nN; i++){
    int nIK = gsl_rng_uniform_int (ptGSLRNG, nK);

    ptCluster->anMaxZ[i] = nIK;
    anW[nIK]++;
  }
  
  updateMeans(ptCluster, ptData);
  
  while(nChange > 0 && nIter < MAX_ITER){
    nChange = 0;
    /*reassign vectors*/
    for(i = 0; i < nN; i++){
      double dMinDist = DBL_MAX;//calcDist(adPi[i],aadX[i],aadMu[0],nD);
      int    nMinK = NOT_SET;

      for(k = 0; k < nK; k++){
	double dDist = calcDist(aadX[i],aadMu[k],nD);

	if(dDist < dMinDist){
	  nMinK = k;
	  dMinDist = dDist;
	}
      }

      if(nMinK != anMaxZ[i]){
	int nCurr = anMaxZ[i];
	nChange++;
	anW[nCurr]--;
	anW[nMinK]++;
	anMaxZ[i] = nMinK;

	/*check for empty clusters*/
	if(anW[nCurr] == 0){
	  int nRandI =  gsl_rng_uniform_int (ptGSLRNG, nN);
	  int nKI = 0;
	  /*select at random from non empty clusters*/

	  while(anW[anMaxZ[nRandI]] == 1){
	    nRandI =  gsl_rng_uniform_int (ptGSLRNG, nN);
	  }

	  nKI = anMaxZ[nRandI];
	  anW[nKI]--;
	  anW[nCurr] = 1;
	  anMaxZ[nRandI] = nCurr;
	}
      }
    }
    //printf("%d %d\n",nIter,nChange);
    nIter++;
    updateMeans(ptCluster, ptData);
  }

  for(i = 0; i < nN; i++){
    for(k = 0; k < nK; k++){
      ptCluster->aadZ[i][k] = 0.0;
    }
    ptCluster->aadZ[i][anMaxZ[i]] = 1.0;
  }

  performMStep(ptCluster, ptData);
  return;
}

double calcLNP(t_Cluster* ptCluster, double *adX, double* adZ){
  int i = 0, j = 0, k = 0, l = 0, m = 0;
  int nK = ptCluster->nK, nD = ptCluster->nD;
  gsl_vector *ptDiff = gsl_vector_alloc(nD);
  gsl_vector *ptRes = gsl_vector_alloc(nD);
  double adDist[nK], dD = (double) nD;
  double** aadMu = ptCluster->aadMu, *adPi = ptCluster->adPi;
  double dMinDist = DBL_MAX;
  double dP = 0.0;

  for(k = 0; k < nK; k++){
    /*set vector to data point*/
    for(l = 0; l < nD; l++){
      gsl_vector_set(ptDiff,l,adX[l] - aadMu[k][l]);
    }
    //These functions compute the matrix-vector product and sum y = \alpha A x + \beta y for the symmetric matrix A. Since the matrix A is symmetric only its upper half or lower half need to be stored. When Uplo is CblasUpper then the upper triangle and diagonal of A are used, and when Uplo is CblasLower then the lower triangle and diagonal of A are used.

    gsl_blas_dsymv (CblasLower, 1.0, ptCluster->aptSigma[k], ptDiff, 0.0, ptRes);

    //gsl_linalg_cholesky_svx (ptCluster->aptSigma[k], ptDiff);

    //gsl_linalg_cholesky_solve (ptCluster->aptSigma[k], ptDiff, ptRes);

    gsl_blas_ddot (ptDiff, ptRes, &adDist[k]);

    adDist[k] += ptCluster->adLDet[k];

    if(adDist[k] < dMinDist){
      dMinDist = adDist[k];
    }
  }

  for(k = 0; k < nK; k++){
    adZ[k] = adPi[k]*exp(-0.5*(adDist[k]-dMinDist));
    dP += adZ[k];
  }

  dP = log(dP) - 0.5*(dMinDist + dD*log(2.0*M_PI));

  gsl_vector_free(ptRes);
  gsl_vector_free(ptDiff);
  return dP;
}

void gmmTrainEM(t_Cluster *ptCluster, t_Data *ptData)
{
  int i = 0, j = 0, k = 0, l = 0, m = 0, nIter = 0;
  int nN = ptData->nN, nK = ptCluster->nK, nD = ptData->nD;
  /*change in log-likelihood*/
  double dDelta = DBL_MAX;
  double **aadZ = ptCluster->aadZ, **aadX = ptData->aadX;
  double *adLDet = ptCluster->adLDet, *adPi = ptCluster->adPi, **aadMu = ptCluster->aadMu;
  double dN = (double) nN, dD = (double) nD, dK = (double) nK;
  double dP = (double) 0.5*nK*(dD*(dD + 3.0)) + dK - 1.0;

  /*assumes initialised ptCluster object need to provide z values too*/
  ptCluster->dLL = 0.0;
  /*calculate data likelihood*/
  for(i = 0; i < nN; i++){
    ptCluster->dLL += calcLNP(ptCluster,aadX[i],aadZ[i]);
  }

  while(dDelta > MIN_CHANGE_LL && nIter < MAX_ITER){
    double dNewLL = 0.0;
    /*normalise Z stored from LNP step*/
    for(i = 0; i < nN; i++){ /*loop data points i*/
      double dTotalZ = 0.0;

      for(k = 0; k < nK; k++){
	dTotalZ += aadZ[i][k];
      }

      for(k = 0; k < nK; k++){
	aadZ[i][k] /= dTotalZ;
      }
    }
    /*update parameter estimates*/
    performMStep(ptCluster, ptData);

    /*calculate data likelihood*/
    for(i = 0; i < nN; i++){
      dNewLL += calcLNP(ptCluster,aadX[i],aadZ[i]);
    }
    
    dDelta = fabs(dNewLL - ptCluster->dLL);
    //printf("%f %f %f\n",ptCluster->dLL,dNewLL,dDelta);
    ptCluster->dLL = dNewLL;
  }

  /*assign to best clusters*/
  for(i = 0; i < nN; i++){
    double dMaxZ = aadZ[i][0];
    int    nMaxK = 0;
    for(k = 1; k < nK; k++){
      if(aadZ[i][k] > dMaxZ){
	nMaxK = k;
	dMaxZ = aadZ[i][k];
      }
    }
    ptCluster->anMaxZ[i] = nMaxK;
  }
  ptCluster->dBIC = -2.0*ptCluster->dLL + dP*log(dN);

  return;
 memoryError:
  fprintf(stderr, "Failed allocating memory in gmmTrain\n");
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void writeClusters(char *szOutFile, t_Cluster *ptCluster, t_Data *ptData)
{
  int nN = ptCluster->nN, i = 0;
  FILE* ofp = fopen(szOutFile,"w");

  if(ofp){
    for(i = 0; i < nN; i++){
      fprintf(ofp,"%s,%d\n",ptData->aszSampleNames[i],ptCluster->anMaxZ[i]);
    }
  }
  else{
    fprintf(stderr,"Failed to open %s for writing in writeClusters\n", szOutFile);
    fflush(stderr);
  }
}

void writeMeans(char *szOutFile, t_Cluster *ptCluster)
{
  int nK = ptCluster->nK, nD = ptCluster->nD,i = 0, j = 0;
  FILE* ofp = fopen(szOutFile,"w");

  if(ofp){
    for(i = 0; i < nK; i++){
      for(j = 0; j < nD - 1; j++){
	fprintf(ofp,"%f,",ptCluster->aadMu[i][j]);
      }
      fprintf(ofp,"%f\n",ptCluster->aadMu[i][nD - 1]);
    }
  }
  else{
    fprintf(stderr,"Failed to open %s for writing in writeMeanss\n", szOutFile);
    fflush(stderr);
  }
}


void* fitEM(void *pvCluster)
{
  t_Cluster          *ptCluster = (t_Cluster *) pvCluster;
  gsl_rng            *ptGSLRNG     = NULL;
  const gsl_rng_type *ptGSLRNGType = NULL;

  /*initialise GSL RNG*/
  ptGSLRNGType = gsl_rng_default;
  ptGSLRNG     = gsl_rng_alloc(ptGSLRNGType);

  gsl_rng_set (ptGSLRNG, ptCluster->lSeed);

  initKMeans(ptGSLRNG, ptCluster, ptCluster->ptData);

  gmmTrainEM(ptCluster, ptCluster->ptData);

  gsl_rng_free(ptGSLRNG);
}

void* runRThreads(void *pvpDCluster)
{
  t_Cluster   **pptDCluster = (t_Cluster **) pvpDCluster;
  t_Cluster   *ptDCluster = (t_Cluster *) *pptDCluster;
  double      dBestBIC = DBL_MAX;
  t_Cluster** aptCluster = NULL;
  pthread_t   atRestarts[N_RTHREADS]; /*run each restart on a separate thread*/
  int         iret[N_RTHREADS];
  int         r = 0, nBestR = -1;

  aptCluster = (t_Cluster **) malloc(N_RTHREADS*sizeof(t_Cluster*));
  if(!aptCluster)
    goto memoryError;

  for(r = 0; r < N_RTHREADS; r++){
    aptCluster[r] = (t_Cluster *) malloc(sizeof(t_Cluster));

    allocateCluster(aptCluster[r],ptDCluster->nN,ptDCluster->nK,ptDCluster->nD,ptDCluster->ptData,ptDCluster->lSeed + r*R_PRIME);
 
    iret[r] = pthread_create(&atRestarts[r], NULL, fitEM, (void*) aptCluster[r]);
  }

  for(r = 0; r < N_RTHREADS; r++){
    pthread_join(atRestarts[r], NULL);
  }

  /*free up memory associated with input cluster*/
  free(ptDCluster);

  for(r = 0; r < N_RTHREADS; r++){
    if(aptCluster[r]->dBIC < dBestBIC){
      nBestR = r;
      dBestBIC = aptCluster[r]->dBIC;
    }
  }
  
  *pptDCluster = aptCluster[nBestR];
  for(r = 0; r < N_RTHREADS; r++){
    if(r != nBestR){
      destroyCluster(aptCluster[r]);
      free(aptCluster[r]);
    }
  }
  free(aptCluster);
  
  return NULL;
 memoryError:
  fprintf(stderr, "Failed allocating memory in runRThreads\n");
  fflush(stderr);
  exit(EXIT_FAILURE);
}
