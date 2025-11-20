#include "LAAT.h"

size_t reading_data(int argc, char *argv[], vector<vector<float>> &data, 
	size_t &numberOfAnts,
  size_t &numberOfIterations,
  size_t &numberOfSteps,
	size_t &pso_number_particles,
	float &pso_min_radii,
	float &pso_max_radii,
	size_t &dynamic_radius_actived,
	size_t &th_neighb,
	float &kappa,
  size_t &numberofthreads,
	string &output_file_address)
{
	vector<pair<string,size_t>> paramters_names{{"numberOfAnts",0} , 
																							{"numberOfIterations",0} , 
																							{"numberOfSteps",0},
																							{"pso_number_particles",0},
																							{"pso_min_radii",1},
																							{"pso_max_radii",1},
																							{"dynamic_radius_actived",0},
																							{"th_neighb",0} , 
																							{"kappa",1} , 
																							{"numberofthreads",0}};

	vector<pair<size_t *, float *>> full_list_parameters(paramters_names.size()) ;
	full_list_parameters[0].first = &numberOfAnts;
	full_list_parameters[1].first = &numberOfIterations;
	full_list_parameters[2].first = &numberOfSteps;
	full_list_parameters[3].first = &pso_number_particles;
	full_list_parameters[4].second = &pso_min_radii;
	full_list_parameters[5].second = &pso_max_radii;
	full_list_parameters[6].first = &dynamic_radius_actived;
	full_list_parameters[7].first = &th_neighb;
	full_list_parameters[8].second = &kappa;
	full_list_parameters[9].first = &numberofthreads;

	//DEFUAL PARAMETERS VALUES
	th_neighb = 5;
	kappa = 0.8f;
	numberofthreads = 16;
	numberOfAnts = 5*5*5;
	numberOfIterations = 100 ;
	numberOfSteps = 2500;
	pso_number_particles = 100;
	pso_min_radii = 1.0;
	pso_max_radii = 1.0;
	dynamic_radius_actived = 0;

	if(argc != 2)
	{
		printf("\n\n ERROR, there are more than 2 input value, only use the ./***/LAAT.exe and your .ini file \n\n");
		return _FAILURE_;
	}

	char extension[5];
	strncpy(extension,(argv[1]+strlen(argv[1])-4),4);
	extension[4]='\0';
	if(strcmp(extension,".ini") != 0)
	{
		printf("\n\n ERROR, the file '%s' has an extension different from .ini\n\n",argv[1]);
		return _FAILURE_;
	}

	char input_file_name[100];
	sprintf(input_file_name,"%s",argv[1]);
	ifstream input_file(input_file_name);
	string line;
	//char ch;
	size_t counter = 0;
	size_t aux_counter_idx;
	string parameter_name;
	string parameter_value;
	bool check;
	bool check_pure_float = true;
	vector <size_t> flags_parameters(paramters_names.size(), 0);
	size_t counter_flags_paramters = 0;

	if (input_file.is_open()) 
	{
    // Read each line from the file and store it in the
    // 'line' variable.
		while (getline(input_file, line) && counter_flags_paramters < paramters_names.size()) 
		{
			parameter_name.clear();
			counter++;
			if(line[0] != '#' && line[0] != '\0')
			{
				aux_counter_idx = 0;
				
				while(line[aux_counter_idx] == ' ' || line[aux_counter_idx] == '\t')
				{
					aux_counter_idx++;
				}
			
				while(line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '=' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0' && line[aux_counter_idx] != '#')
				{
					parameter_name.append(1,line[aux_counter_idx]);
					aux_counter_idx++;
				}
				
				while(line[aux_counter_idx] != '=' && line[aux_counter_idx] != '\0')
				{
					aux_counter_idx++;
				}

				//Reading the value of the parameter
				if(line[aux_counter_idx] == '=')
				{
					aux_counter_idx++;
					check_pure_float = false;
					parameter_value.clear();

					while(line[aux_counter_idx] == ' ' || line[aux_counter_idx] == '\t' )
					{
						aux_counter_idx++;
					}

					while(line[aux_counter_idx] >= '0' && line[aux_counter_idx] <= '9')
					//while(line[aux_counter_idx] == '8')
					{
						//printf("upsi\n");
						parameter_value.append(1,line[aux_counter_idx]);
						aux_counter_idx++;
					}

					if(line[aux_counter_idx] != '.' && line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0' && line[aux_counter_idx] != '#')
					{
						parameter_value.clear();
					}
					else
					{
						if(line[aux_counter_idx] == '.')
						{
							parameter_value.append(1,line[aux_counter_idx]);
							aux_counter_idx++;
							check_pure_float = true;
						}
						while(line[aux_counter_idx] >= '0' && line[aux_counter_idx] <= '9')
						{
							parameter_value.append(1,line[aux_counter_idx]);
							aux_counter_idx++;
						}

						if(line[aux_counter_idx] != '.' && line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0' && line[aux_counter_idx] != '#')
						{
							parameter_value.clear();
						}
					}
				}
			}

			//Checking paramters
			aux_counter_idx = 0;
			check = false;

			while(!check && aux_counter_idx < paramters_names.size())
			{
				if(strcmp(parameter_name.c_str(),paramters_names[aux_counter_idx].first.c_str()) == 0)
				{

					if(parameter_value.empty())
					{
						printf("\nWARNING: the parameter '%s' has been declare in the input file but the value is not the right type\n",parameter_name.c_str());
					}
					else
					{
						if(flags_parameters[aux_counter_idx] == 0)
						{
							//size_t type
							if(paramters_names[aux_counter_idx].second == 0)
							{
								//The paramter used is not a size_t type
								if(check_pure_float == true)
								{
									printf("\n\n ERROR, the input parameter '%s' must be size_t type, and can not be equal to '%s'\n\n",paramters_names[aux_counter_idx].first.c_str() ,parameter_value.c_str());
									input_file.close();
									return _FAILURE_;
								}
								stringstream stream(parameter_value);
								stream >> *(full_list_parameters[aux_counter_idx].first);
							}
							//float type
							else if(paramters_names[aux_counter_idx].second == 1)
							{
								*(full_list_parameters[aux_counter_idx].second) = stof(parameter_value);
							}
							flags_parameters[aux_counter_idx] = 1;
							counter_flags_paramters++;
						}
						else
						{
							printf("\nWarning: The input parameter '%s', has beed declare more than one times in the input file. Only the first declaration is used\n",paramters_names[aux_counter_idx].first.c_str());
						}
					}
					check = true;
				}
				aux_counter_idx++;
			}
		}
    input_file.close();
  }
  else {
		printf("Error opening the input file %s\n",input_file_name);
		return _FAILURE_;
  }

	if (pso_max_radii == pso_min_radii && dynamic_radius_actived == 1)
	{
		dynamic_radius_actived = 0;
		printf("\nWarning, pso_max_radii = pso_min_radii, dynamic_radius_actived is defined as 0\n");
		flags_parameters[6] = 0;
	}

	if(dynamic_radius_actived == 1)
	{
		size_t min_th_dynamic_radius = ceilf((3.0f * 100.0f/(100.0f - percentage_removed)));
		if(flags_parameters[7] == 0)
		{
			th_neighb = min_th_dynamic_radius;
		}
		else if(min_th_dynamic_radius > th_neighb)
		{
			th_neighb = min_th_dynamic_radius;
			printf("\nWarning, th_neighb < than minimum value accepted %ld, considering the %% removed (%f %%) of neighbours in the radius probability computation\n",min_th_dynamic_radius,percentage_removed );
			flags_parameters[7] = 0;
		}
	}






	printf("\nInput file = '%s'\n", input_file_name);

	//Checking list of parameters readed

	printf("\n\nPARAMETERS USED: \n\n");

	vector<string> user_default {"DEFAULT","USER"};
	for(size_t aux_idx = 0; aux_idx < paramters_names.size(); aux_idx++)
	{
		if(paramters_names[aux_idx].second == 0)
		{
			printf("%s = %d, %s value\n", paramters_names[aux_idx].first.c_str(),(int) *(full_list_parameters[aux_idx].first), user_default[flags_parameters[aux_idx]].c_str() );
		}
		else
		{
			printf("%s = %f, %s value\n", paramters_names[aux_idx].first.c_str(), *(full_list_parameters[aux_idx].second), user_default[flags_parameters[aux_idx]].c_str() );
		}
	}


	////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// READING DATA FILE ADDRESS /////////////////////
	////////////////////////////////////////////////////////////////////////////////////

	check = false;
	aux_counter_idx = 0;
	size_t counter_line = 0;
	string data_file_address;
	data_file_address.clear();
	parameter_name.clear();

	input_file.open(input_file_name);

	//Reopen the input file to find the 'input_data_file'
  if (input_file.is_open()) 
	{
    // Read each line from the file and store it in the
    // 'line' variable.
    while (getline(input_file, line) && check == false) 
		{
			parameter_name.clear();
			if(line[0] != '#' && line[0] != '\0')
			{
				aux_counter_idx = 0;
				
				while(line[aux_counter_idx] == ' ' || line[aux_counter_idx] == '\t')
				{
					aux_counter_idx++;
				}
			
				while(line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '=' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0' && line[aux_counter_idx] != '#')
				{
					parameter_name.append(1,line[aux_counter_idx]);
					aux_counter_idx++;
				}

				if(strcmp(parameter_name.c_str(),"input_data_file") == 0)
				{

					while(line[aux_counter_idx] != '=' && line[aux_counter_idx] != '\0')
					{
						aux_counter_idx++;
					}

					data_file_address.clear();

					//Reading the address
					if(line[aux_counter_idx] == '=')
					{
						aux_counter_idx++;

						while(line[aux_counter_idx] == ' ' || line[aux_counter_idx] == '\t' )
						{
							aux_counter_idx++;
						}

						while(line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0' && line[aux_counter_idx] != '#')
						{
							data_file_address.append(1,line[aux_counter_idx]);
							aux_counter_idx++;
						}
						check = true;
					}

					if(data_file_address.empty())
					{
						printf("\nWARNING: at line  %d the parameter 'input_data_file' has been declare in the input file but the value is empty\n",(int) counter_line);
					}
				}
			}
			counter_line++;
		}
		input_file.close();
  }

	if(parameter_name.empty())
	{
		printf("\n\n ERROR, there are NO 'input_data_file' variable in your input_file_name \n\n");
		return _FAILURE_;
	}
	else if(data_file_address.empty())
	{
		printf("\n\n ERROR, there are NO appropriate value for your 'input_data_file' variable in your input_file_name \n\n");
		return _FAILURE_;
	}

	printf("\nInput data file = '%s'\n", data_file_address.c_str());

	////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// READING THE DATA FILE ////////////////////////
	////////////////////////////////////////////////////////////////////////////////////

	FILE *data_file = NULL;
	data_file = fopen(data_file_address.c_str(), "r");
	if (data_file == NULL)
	{
		printf("Error opening the data_file_address %s\n",data_file_address.c_str());
		return _FAILURE_;
	}

	std::ifstream myfile(data_file_address);

	// new lines will be skipped unless we stop it from happening:    
	myfile.unsetf(std::ios_base::skipws);

	// count the newlines with an algorithm specialized for counting:
	unsigned line_count = std::count(
				std::istream_iterator<char>(myfile),
				std::istream_iterator<char>(), 
				'\n');
	printf("\nData size = %d\n", (int) line_count );

	data.resize(line_count);
	for (size_t idx = 0; idx < line_count;idx++)
	{
		data[idx].resize(3);
	}

	for (size_t idx = 0; idx < line_count; idx++)
	{
		if (fscanf(data_file, "%f,%f,%f", &data[idx][0],&data[idx][1],&data[idx][2]) == 0)
		{
			printf("\n\nERROR: Fail to read the position %d of the data_file %s\n\n",(int) idx,data_file_address.c_str());
			return _FAILURE_;
		}
	}

	fclose(data_file);

	////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////// READING OUTPUT DATA FILE ADDRESS /////////////////////
	////////////////////////////////////////////////////////////////////////////////////

	check = false;
	aux_counter_idx = 0;
	parameter_name.clear();

	input_file.open(input_file_name);

	//Reopen the input file to find the 'input_data_file'
  if (input_file.is_open()) 
	{
    // Read each line from the file and store it in the
    // 'line' variable.
    while (getline(input_file, line) && check == false) 
		{
			parameter_name.clear();
			if(line[0] != '#' && line[0] != '\0')
			{
				aux_counter_idx = 0;
				
				while(line[aux_counter_idx] == ' ' || line[aux_counter_idx] == '\t')
				{
					aux_counter_idx++;
				}
			
				while(line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '=' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0' && line[aux_counter_idx] != '#')
				{
					parameter_name.append(1,line[aux_counter_idx]);
					aux_counter_idx++;
				}

				if(strcmp(parameter_name.c_str(),"ouptut_file_address") == 0)
				{

					while(line[aux_counter_idx] != '=' && line[aux_counter_idx] != '\0')
					{
						aux_counter_idx++;
					}

					output_file_address.clear();

					//Reading the address
					if(line[aux_counter_idx] == '=')
					{
						aux_counter_idx++;

						while(line[aux_counter_idx] == ' ' || line[aux_counter_idx] == '\t' )
						{
							aux_counter_idx++;
						}

						while(line[aux_counter_idx] != ' ' && line[aux_counter_idx] != '\t' && line[aux_counter_idx] != '\0'  && line[aux_counter_idx] != '#')
						{
							output_file_address.append(1,line[aux_counter_idx]);
							aux_counter_idx++;
						}
						check = true;
					}

					if(output_file_address.empty())
					{
						printf("\nWARNING: at line  %d the parameter 'output_file_address' has been declare in the input file but the value is empty\n",(int) counter_line);
					}
				}
			}
			counter_line++;
		}
		input_file.close();
  }

	if(parameter_name.empty())
	{
		printf("\n\n ERROR, there are NO 'output_file_address' variable in your input_file_name \n\n");
		return _FAILURE_;
	}
	else if(output_file_address.empty())
	{
		printf("\n\n ERROR, there are NO appropriate value for your 'output_file_address' variable in your input_file_name \n\n");
		return _FAILURE_;
	}


	printf("\nOutput file address = %s\n", output_file_address.c_str());

	return _SUCCESS_;
}