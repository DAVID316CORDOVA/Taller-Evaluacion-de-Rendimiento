/************************************************************************************************
								=	Pontificia Universidad Javeriana	=

	Author: John Corredor
	Versión Prototipo "Inicial" MPI Distribución de Matrices en Nodos de Cómputo	

	Descripción:
		Multiplicación de Matrices cuadradas, algoritmo Filas x Columnas
		Square Matrix Multiplication, rowsxcolumns version:
		=> MPI_Init:				- Inicializacíón del entorno MPI 
		=> MPI_Comm_size:			- Obtener cantidad de procesos o workers en ejecución
		=> MPI_Comm_rank:			- Obtener el ID del proceso o worker  
		=> MPI_Get_processor_name:	- Obtener nombre del Nodo de Cómputo 
		=> MPI_Finalize:			- Finalización del entorno MPI

	Ejecución:
		$mpirun -hostfile fileHOSTSMPI -np 3 ./mmMPIv1 8 1
************************************************************************************************/
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "moduloMPI.h" 

struct timeval inicio, fin;

void impMatrix(double *mat, int n){
	if(n<13){
		/*Impresión de Matriz */
		printf("\n====================================================================");
		for (int i = 0; i < n*n; i++, mat++) {
			if(i%n == 0 ) printf("\n"); 
			printf("%0.3f ", *mat);
		}
		printf("\n====================================================================\n");
	}
}

void matrixTRP(int N, double *mB, double *mT){
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			mT[i*N+j] = mB[j*N+i];
	impMatrix(mT, N);
}

void mxmOmpFxT(double *mA, double *mB, double *mC, int tw, int D, int nH){
	double *mT  = (double *)calloc(D*D, sizeof(double));
	matrixTRP(D, mB, mT);
	omp_set_num_threads(nH);
	#pragma omp parallel
	{
	#pragma omp for
	for(int i=0; i<tw; i++)
		for(int j=0; j<D; j++){
			double *pA, *pB, Suma = 0.0;
			pA = mA+i*D;
			pB = mT+j*D;
			for(int k=0; k<D; k++, pA++, pB++)
				Suma += *pA * *pB;
			mC[i*D+j] = Suma;
		}
	}
	free(mT);
}

void mxmOmpFxC(double *mA, double *mB, double *mC, int tw, int D, int nH){
	omp_set_num_threads(nH);
	#pragma omp parallel
	{
	#pragma omp for
	for(int i=0; i<tw; i++)
		for(int j=0; j<D; j++){
			double *pA, *pB, Suma = 0.0;
			pA = mA+i*D;
			pB = mB+j;
			for(int k=0; k<D; k++, pA++, pB+=D)
				Suma += *pA * *pB;
			mC[i*D+j] = Suma;
		}
	}
}

void mensajeVerifica(int N, int cantidadW){
	if(N<13){
		printf("\n");
		printf("********************************************************************\n");
		printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		printf("++++++ \t\tDimensión de Matrix NxN \t  = %dx%d \t++++\n", N, N);
		printf("++++++ \t\tCantidad de Workers (np - MASTER) = %d \t\t++++\n", cantidadW);
		printf("++++++ \t\tTajada de matriz A para Workers   = %dx%d \t++++\n", N/cantidadW,N);
		printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
		printf("********************************************************************\n");
		printf("\n");
	}
}


void iniMatrix(double *mA, double *mB, int D){
	srand(time(NULL));
	for(int i=0; i<D*D; i++, mA++, mB++){
		*mA = 0.08*i; //(double) rand()/RAND_MAX*(5.1-2.3);
		*mB = 0.02*i;   //(double) rand()/RAND_MAX*(6.1-2.8);
	}
}

void iniTime(){
		gettimeofday(&inicio, (void *)0);
}

void endTime(){
		gettimeofday(&fin, (void *)0);
		fin.tv_usec -= inicio.tv_usec;
		fin.tv_sec  -= inicio.tv_sec;
		double tiempo = (double) (fin.tv_sec*1000000 + fin.tv_usec);
		printf("%9.0f \n", tiempo);
}

void argumentos(int cantidad){
	if (cantidad != 3){
		printf("Ingreso de Argumentos: \n\n");
		printf("\t\t$mpirun -hostfile file -np p ./ejecutable DimMatriz NumHilos \n\n");
		printf("\nfile: Archivo de Master y Workers \n");
		printf("\np: procesos Master+Workers\n");
		exit(0);
	}
}

void verificarDiv(int qworkers, int Dim){
/*** Check if N is divisible by number of worker processes ***/
	if ((qworkers < 1) || (Dim%qworkers != 0)) {
		printf("Error: NxN (%d) debe ser divisible por cantidad de workers (%d)\n", Dim, qworkers);
		printf("Error: Número de procesos (%d) > 1 \n", qworkers);
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(0);
	}
}
