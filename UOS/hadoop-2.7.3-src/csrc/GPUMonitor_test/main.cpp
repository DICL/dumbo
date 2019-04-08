#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include "sys/time.h"
#include <nvml.h>

nvmlDevice_t device;
nvmlProcessInfo_t pinfos[10];

void init_nvml ()
{
    nvmlReturn_t result;
    unsigned int device_count, i;
	char sentence[200];
	std::string err = "";

    result = nvmlInit();
    if (NVML_SUCCESS != result) { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        sprintf(sentence, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
		err.append( (std::string)sentence );
    }
	char name[NVML_DEVICE_NAME_BUFFER_SIZE];
	result = nvmlDeviceGetHandleByIndex(0, &device);
	if (NVML_SUCCESS != result) { 
		printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
		sprintf(sentence,"Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
		err.append( (std::string)sentence );
		result = nvmlShutdown();
		return ;
	}
	result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
	if (NVML_SUCCESS != result) { 
		printf("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
		sprintf(sentence,"Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
		err.append( (std::string)sentence );
		result = nvmlShutdown();
		return ;
	}
	printf("Device : %s\n",name);
	sprintf(sentence,"Device : %s\n",name);
	err.append( (std::string)sentence );
}


/*
 * get_util()
 * GET STATE OF GPU-UTIL & GPU-MEMORY
 */

int util_gpu=0, util_mem=0;

unsigned int get_util()
{
	unsigned int return_value=0;
	unsigned int infoCount=-1;
    nvmlReturn_t result;
	result = nvmlDeviceGetComputeRunningProcesses(device , &infoCount, pinfos);
//	startTimer( &st_gur);
	nvmlUtilization_t utilization;
	nvmlDeviceGetUtilizationRates( device, &utilization);
	
//            infoCount memory    gpu
//	0000 0000 0000 0000 0000 0000 0000 0000	
	return_value  = (utilization.gpu        ) & 0x000000FF;
	return_value |= (utilization.memory << 8) & 0x0000FF00;
	return_value |= (infoCount          <<16) & 0x00FF0000;

	util_gpu = utilization.gpu;
	util_mem = utilization.memory;
	return return_value;
}

void print_util()
{
	printf("util: %d %d", util_gpu, util_mem);
}

/*
 * get_clock()
 * GET CLOCK OF GPU's SM & MEMORY
 */

unsigned int clock_gpu=0, clock_mem=0;

void get_clock()
{
//	nvmlReturn_t result;
	nvmlClockType_t type;

	type = NVML_CLOCK_SM;
	nvmlDeviceGetClockInfo(device, type, &clock_gpu);
	type = NVML_CLOCK_MEM;
	nvmlDeviceGetClockInfo(device, type, &clock_mem);
}

void print_clock()
{
	printf("clock: %d %d", clock_gpu, clock_mem);
}

/*
 * main
 */

void print_all()
{
	print_util();
	printf("\t");
	print_clock();
	printf("\n");
}

int main (int argc, char* argv[]) {
	init_nvml();
	
	bool idle = true;
	useconds_t internal = 60 * 1000; // us

	while(1)
	{
	get_util();
	if (util_gpu != 0 || util_mem != 0) {
		print_all();
		idle = false;
	} else {
		if (!idle) {
		print_all();
		printf("--------------------------------\n");
		}
		idle = true;
	}
	usleep(internal);
	}
	
	return 0;
}
