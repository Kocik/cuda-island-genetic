################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuda-island-genetic.cu 

OBJS += \
./src/cuda-island-genetic.o 

CU_DEPS += \
./src/cuda-island-genetic.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/0_Simple" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/kocik/cuda-workspace/cuda-genetic-island" -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/0_Simple" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/kocik/cuda-workspace/cuda-genetic-island" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


