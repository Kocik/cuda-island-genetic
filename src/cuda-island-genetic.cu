#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_runtime.h>

const int ISLANDS = 64;
const int ISLANDS_PER_ROW = 8;
const int GENOME_LENGTH=5;
const int ISLAND_POPULATION=50;
const int SELECTION_COUNT=40;
const float MUTATION_CHANCE= 0.8;
const int ITEMS_MAX_WEIGHT = 5;
const int ITEMS_MAX_VALUE = 20;
const int ITEMS_MAX = 20;



__device__ float fitnessValue(float *baseSizes, float *baseValues, int*phenotype, float backpackMaxSize)
{
    float size = 0, value = 0;
    int count = 0;

    for( int i = 0; i < GENOME_LENGTH; i++ ) {
    	count = phenotype[i];
    	size += baseSizes[i] * count;
    	value += baseValues[i] * count;
    }

    if(size > backpackMaxSize) {
    	return 0;
    }
    return value;
}

__device__ void
sortByFitness(float*populationFitness, int* sortedAssoc, float* totalFitness)
{
    int i, j;
    *totalFitness = 1;
    float phenotypeFitness = 0;
    for ( i = 0; i < ISLAND_POPULATION; ++i ){
    	sortedAssoc[i] = i;
        phenotypeFitness = populationFitness[i];
        for (
        		j = i;
        		j > 0 && populationFitness[sortedAssoc[j - 1]] > phenotypeFitness;
        		j-- )
        {
        	sortedAssoc[j] = sortedAssoc[j - 1];
        }
        sortedAssoc[j] = i;
    }
}

__device__ void
normalizeFitness(float*populationFitness, int* sortedAssoc, float totalFitness)
{
    int i, j;
    float lastFitness = 0;
    for ( i = 0; i < ISLAND_POPULATION; ++i ){
    	j = sortedAssoc[i];
        lastFitness += populationFitness[j];
        populationFitness[j] = lastFitness/totalFitness;
    }
}

__device__ void
selectionTrunc(int* sortedAssoc, int* selectedAssoc)
{
	for(int i = 1; i <= SELECTION_COUNT; i++) {
		selectedAssoc[i-1] = sortedAssoc[ISLAND_POPULATION - i];
	}
}

__device__ float
computeFitnessValue(float *baseSizes, float *baseValues, int*populationRow, float*populationFitness, float backpackMaxSize)
{
    float max = 0;
    for(int i = 0; i < ISLAND_POPULATION; i++ ) {
    	populationFitness[i] = fitnessValue(
    			baseSizes,
    			baseValues,
    			&(populationRow[GENOME_LENGTH * i]),
    			backpackMaxSize);
    	if( populationFitness[i] > max) {
    		max = populationFitness[i];
    	}
    }
    return max;
}

__device__ void
crossover(
		int*populationRow,
		int*newPopulation,
		int*selectedPopulation,
		curandState_t *randomState)
{
	int i,j;
	int selectedPhenotype,
		selectedPhenotypeA,
		selectedPhenotypeB;
	int treshold = 0;
	for( i = 0; i < ISLAND_POPULATION; i++) {

		selectedPhenotypeA = selectedPopulation[ curand(randomState) % SELECTION_COUNT ];
		selectedPhenotypeB = selectedPopulation[ curand(randomState) % SELECTION_COUNT ];

		treshold = curand(randomState) % GENOME_LENGTH;

		for(j = 0; j < GENOME_LENGTH; j++) {
			if(j < treshold) {
				selectedPhenotype = selectedPhenotypeA;
			} else {
				selectedPhenotype = selectedPhenotypeB;
			}

			newPopulation[i * GENOME_LENGTH + j] =
					populationRow[selectedPhenotype * GENOME_LENGTH];
		}
	}
}

__device__ void
mutation(
		int*newPopulation,
		curandState_t *randomState)
{
	int i;

	for( i = 0; i < ISLAND_POPULATION; i++) {
		if(curand_uniform(randomState) < MUTATION_CHANCE) {
			newPopulation[ i* GENOME_LENGTH + (curand(randomState) % GENOME_LENGTH ) ]
			               = curand(randomState) % ITEMS_MAX;
		}
	}
}

__device__ void
killPreviousPopulation(
		int*populationRow,
		int*newPopulation)
{
	int i;

	for( i = 0; i < ISLAND_POPULATION * GENOME_LENGTH; i++) {
		populationRow[i] = newPopulation[i];
	}
}


__global__ void
geneticAlgorithmGeneration(curandState_t* states, float *baseSizes, float *baseValues, int*population, float* bestValues, float backpackMaxSize)
{
	//index of the island itself
    int island_y = blockDim.y * blockIdx.y + threadIdx.y;
    int island_x = blockDim.x * blockIdx.x + threadIdx.x;

    int* populationRow = &population[island_y * GENOME_LENGTH * ISLAND_POPULATION * ISLANDS_PER_ROW + island_x * GENOME_LENGTH * ISLAND_POPULATION ];


	curandState_t *randomState = &(states[blockDim.x]);

	float populationFitness[ISLAND_POPULATION];


	float best = computeFitnessValue(
			baseSizes,
			baseValues,
			populationRow,
			populationFitness,
			backpackMaxSize);

	bestValues[island_y * ISLANDS_PER_ROW + island_x] = best;

	int sortAssoc[ISLAND_POPULATION];
	float totalFitness;

	sortByFitness(populationFitness, sortAssoc, &totalFitness);
	//normalizeFitness(populationFitness, sortAssoc, totalFitness);

	int selectedAssoc[SELECTION_COUNT];
	selectionTrunc(sortAssoc, selectedAssoc);

	int newPopulation[ISLAND_POPULATION];
	crossover(populationRow, newPopulation, selectedAssoc, randomState);
	mutation(newPopulation, randomState);

	killPreviousPopulation(populationRow, newPopulation);

}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/**
 * Host main routine
 */
int
main(void)
{
    cudaError_t err = cudaSuccess;

    srand(time(NULL));
    int threadsPerBlock = ISLANDS;
    int blocksPerGrid = 1;

    int sizeFloat = sizeof(float);
    int sizeInt = sizeof(int);

    int baseLength =  GENOME_LENGTH;
    int sizeBase =  baseLength * sizeFloat;
    int populationLength = ISLANDS * GENOME_LENGTH * ISLAND_POPULATION;
    int sizePopulation = populationLength * sizeInt;
    int sizeBestValue = ISLANDS * sizeFloat;


    float *cu_baseSizes = NULL;
    err = cudaMalloc((void **)&cu_baseSizes, sizeBase);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector baseSize (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *cu_baseValues = NULL;
    err = cudaMalloc((void **)&cu_baseValues, sizeBase);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector baseValues (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *cu_population = NULL;
    err = cudaMalloc((void **)&cu_population, sizePopulation);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Population (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *cu_bestValue = NULL;
    err = cudaMalloc((void **)&cu_bestValue, sizeBestValue);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector bestValue (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    curandState_t *states = NULL;
    /* allocate space on the GPU for the random states */
    err = cudaMalloc((void**) &states, blocksPerGrid * sizeof(curandState_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector randomStates (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    float *baseSizes = (float*)malloc(sizeBase);
    float *baseValues = (float*)malloc(sizeBase);
    int *population = (int*)malloc(sizePopulation);
    float *bestValue = (float*)malloc(sizeBestValue);

    if (baseSizes == NULL || baseValues == NULL || population == NULL || bestValue == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i <baseLength; i++) {
    	baseSizes[i] = (float)rand()/(float)(RAND_MAX/ITEMS_MAX_WEIGHT);
    	baseValues[i] =  (float)rand()/(float)(RAND_MAX/ITEMS_MAX_VALUE);
    	printf("%f - %f |", baseSizes[i], baseValues[i] );
    }
	printf("\n");

    for(int i = 0; i <populationLength; i++) {
    	population[i] = rand() % ITEMS_MAX;
    	printf("%d - ", population[i]);
    }

    err = cudaMemcpy( cu_baseSizes, baseSizes, sizeBase, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector baseSizes from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy( cu_baseValues, baseValues, sizeBase, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector baseValues from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy( cu_population, population, sizePopulation, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector population from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("CUDA Init kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    /* invoke the GPU to initialize all of the random states */
    init<<<blocksPerGrid, 1>>>(time(0), states);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch geneticAlgorithmGeneration kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int backpackMaxSize = ITEMS_MAX*2;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    for( int i = 1; i <= 100 ; i++) {

        geneticAlgorithmGeneration<<<blocksPerGrid, threadsPerBlock>>>(
        		states,
        		cu_baseSizes,
        		cu_baseValues,
        		cu_population,
        		cu_bestValue,
        		backpackMaxSize);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch geneticAlgorithmGeneration 1 kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaDeviceSynchronize();


        if( i % 20 == 0 ) {
        	float max = 0;
			// Verify that the result vector is correct
        	printf("Copy output data from the CUDA device to the host memory\n");
        	err = cudaMemcpy(bestValue, cu_bestValue, sizeBestValue, cudaMemcpyDeviceToHost);
			for (int i = 0; i < ISLANDS; ++i)
			{
				printf("%f | ", bestValue[i]);
				if(bestValue[i] > max) {
					max = bestValue[i];
				}
			}
			printf("\nMax: %f\n", max);
			printf("\n");
        }
    }

    // Free device global memory
    err = cudaFree(cu_baseSizes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(cu_baseValues);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(cu_population);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(cu_bestValue);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(states);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(baseSizes);
    free(baseValues);
    free(population);
    free(bestValue);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

