#include "LAAT.h"

size_t exporting_data(vector<float> &pheromone, string &ouptut_folder_address)
{
	FILE *output_file = NULL;
	output_file = fopen(ouptut_folder_address.c_str(), "w");
	if (output_file == NULL)
	{
		printf("\nWARNING: Error opening Output file %s\n", ouptut_folder_address.c_str());
		printf("The new output file is localized in the shell (terminal) address and called pheromone.csv\n");
		ouptut_folder_address = "pheromone.csv";
		output_file = fopen(ouptut_folder_address.c_str(), "w");
		if (output_file == NULL)
		{
			printf("\n\nERROR, it is not possible to create or read the pheromone.csv file\n\n");
			return _FAILURE_;
		}
	}

	printf("\nExporting pheromone at '%s'\n", ouptut_folder_address.c_str());

	for( size_t idx = 0;idx<pheromone.size();idx++)
	{
		fprintf(output_file, "%1.12f\n", pheromone[idx]);
	}
		
	fclose(output_file);

	return _SUCCESS_;
}