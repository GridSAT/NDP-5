// NDP-5_2_3.cpp
// 
// NDP - Parallel SAT-Solver with OpenMPI for unlimited scalability
// 
// Copyright (c) 2024 GridSAT Stiftung
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
// 
// GridSAT Stiftung - Georgstr. 11 - 30159 Hannover - Germany - info@gridsat.io
//
// https://gridsat.eth/limo - https://gridsat.eth.link - http://gridsat.io - ipns://gridsat.eth
//
//
//
// 
// +++ READ.me +++
//
// NDP - Parallel SAT-Solver with OpenMPI for unlimited scalability
//
// This NDP is an efficient parallel SAT-Solver. It parses DIMACS, factorizes, granulates input DIMACS
// into subproblems with BFS (Breadth-First Search), and performs parallel DFS (Depth-First Search).
// NDP 5.2.3 supports custom configuration through command-line options, leveraging OpenMPI for
// distributed multi-node computation. The NDP outputs dynamically generated file names based on the
// input parameters and a truncated problem ID hash with the current UTC time.
// Version 5.2.3 now with -a CLI option for all assignments and #SAT.
// 
// 
// REQUIREMENTS:
// 
// 	Input:			Generate DIMACS files at Paul Purdom and Amr Sabry's CNF Generator at:
// 					https://cgi.luddy.indiana.edu/~sabry/cnf.html
// 					For bit-wise input generation, use e.g.: https://bigprimes.org/RSA-challenge
// 	or
// 					generate DIMACS locally with: https://github.com/GridSAT/CNF_FACT-MULT
// 					or on IPFS ipfs://QmYuzG46RnjhVXQj7sxficdRX2tUbzcTkSjZAKENMF5jba
//
//					Input file extensions: any & none
//					E.g: .dimacs ⎪ .cnf ⎪ .txt ⎪ .doc ⎪ .xyz ⎪ [filname]
// 
//   GMP Library:	Ensure you have the GMP (GNU Multiple Precision Arithmetic Library) installed to handle
// 					arbitrary-precision arithmetic.
//					Install with:
//
//					sudo apt install libgmp-dev libgmpxx4ldbl
//
//   OpenMPI:		Required for distributed computation across nodes.
//					Install with:
//
//					sudo apt install openmpi-bin libopenmpi-dev
// 
//
// INSTALLATION:
//
//		1. SSH Setup (Password-less Access)
//
// 		Generate an SSH key on the head node
//		ssh-keygen -t rsa -b 4096
//
//		note: use suggested default location and do not enter password (hit 3x enter) 
//
//
//		Copy the public key to each node (all nodes must have the respective SSH keys of each other)
//		ssh-copy-id -p <port_number> user@hostname
//
//		Test SSH access (use configured ports)
//		ssh -p <port_number> user@hostname
//
//
//		Configure SSH for ease of access. Edit `~/.ssh/config` on each node:
//
//		# Default settings for all hosts
//		Host *
//		    ForwardAgent no
//		    ForwardX11 no
//		    ServerAliveInterval 60
//		    ServerAliveCountMax 3
//		    StrictHostKeyChecking no
//		    UserKnownHostsFile /dev/null
//		    LogLevel ERROR
//		    ControlMaster auto
//		    ControlPath ~/.ssh/sockets/%r@%h:%p
//		    ControlPersist 10m
//		
//		# Custom settings for specific hosts
//		Host node1
//			HostName <IP_ADDRESS_1>
//			Port <PORT_1>
//			User user_name
//
//		Host node2
//			HostName <IP_ADDRESS_2>
//			Port <PORT_2>
//			User user_name
//
//		Host nodeX
//			HostName <IP_ADDRESS_X>
//			Port <PORT_X>
//			User user_name
//
//		You can now SSH directly to `node1`, `node2`, `nodeX` without specifying ports or usernames.
//
//
//		2. MPI Hostfile (Define Hosts and Slots)
//
//		Create a hostfile to specify MPI slots per node (subtract at least 1 core for system on each node).
//		Save as `your_hostfile.txt`:
//
//		node1 slots=<number logic cores - system reserve> or any number >0
// 		node2 slots=<number logic cores - system reserve> or any number >0
//
//
//		3. Install Required Libraries
//
//		Update system packages and install the required libraries:
//
//		sudo apt update
//		sudo apt install build-essential
//		sudo apt install libgmp-dev libgmpxx4ldbl
//		sudo apt install openmpi-bin libopenmpi-dev
//
//		check with:
//
//		g++ --version
//		mpirun --version
//		which mpirun
//
//
//		4. Environment Setup
//
//		Ensure the environment variables for MPI are set up correctly:
//
//		export PATH=/usr/local/openmpi/bin:$PATH
//		export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
//		unset DISPLAY
//		source ~/.bashrc
//		mkdir -p ~/.ssh/sockets
//
//		check with:
//
//		echo $LD_LIBRARY_PATH
//
//
//		5. Permissions
//
//		Set permissions for SSH and the MPI hostfile:
//
//		chmod 600 ~/.ssh/config
//		chmod 644 /path/to/mpi_hostfile
//
//
// COMPILATION:
// 
// 	To compile the program on Linux (tested on Ubuntu 24.04.1 LTS), use the following command (compile on every node):
// 
// 	mpic++ -std=c++17 -Ofast -o NDP-5_2_3 NDP-5_2_3.cpp -lgmpxx -lgmp
// 
// 
// 	CLI USAGE:
// 
// 	Once compiled, the program can be run from the command line using the following format:
// 
//	mpirun --use-hwthread-cpus --map-by slot --hostfile <hostfile_path> --mca plm_rsh_args "-q -F <ssh_config_path>" ./NDP-5_2_3 <dimacs_file> [-d depth | -t max_tasks | -q max_queues] [-a find_all_assignments]
//
// 	mpirun:						Initializes MPI execution across multiple nodes.
// 	--use-hwthread-cpus:		Uses hardware threads for each logical CPU core, maximizing CPU utilization per node.
// 	--hostfile <hostfile_path>:	Specifies the file containing the list of nodes and the number of slots (CPUs) each node can contribute.
//	
// 	Command-Line Options:
// 
//     <dimacs_file>: The path to the input DIMACS file.
//     -d depth: Set a custom depth for BFS iterations. (Optional)
//     -t max_tasks: Set the maximum number of tasks for BFS. (Optional)
//     -q max_queues: Limit the maximum number of tasks in the BFS queue. (Optional)
//     -a find_all_assignments: Finds all assignments (if any) and outputs a summary file _sum.txt including #Solutions. (Optional)
// 
// 	Basic execution with nodes: mpirun --use-hwthread-cpus --map-by slot --hostfile your_hostfile.txt --mca plm_rsh_args "-q -F ./NDP-5_2_3 /home/your_username/.ssh/config" inputs/RSA/rsaFACT-128bit.dimacs
//           on single machine: mpirun --use-hwthread-cpus --map-by slot -np <#cores> ./NDP-5_2_3 /home/your_username/.ssh/config" inputs/RSA/rsaFACT-64bit.dimacs
//
// 	This will run the program using the default settings for BFS and DFS and output the results to the current working directory
// 	with the node setup as specified in `~/.ssh/config` and `your_hostfile.txt` - the node running the command will be head-node, any other connected node will be a worker.
//		
//	Defaults:	max_tasks = num_clauses - num_vars
//				output_directory = input_directory
//	 
// 	Setting a custom depth:					-d 5000
// 
// 	Limiting the number of tasks:			-t 1000
//
// 	Setting a custom Queue Size:			-q 256
// 	
// 	NOTE: results are saved into working directory ONLY !!!
//
// 
// MONITORING:
//
//		Monitor system and CPU usage on each node in real time:
//
//		mpstat -P ALL 1
//
//
// OUTPUT:
// 
//     The output file will be saved on the node which found the solution in the format:
//     NDP-5_2_3-<input_file_name>_<truncated_problem_id>_<cli-options>.txt
//     
//     Example: NDP-5_2_3_rsaFACT-128bit_8dfcb_auto.txt (no cli option for Depth/#Tasks/Queue Size)
//	   On node: node7 
//
//	   With CLI -a every assignment is saved separately on-the-fly with an additional summary file _sum.txt with #Solutions
// 
// 	NOTE:
// 			only accepts input generated by Paul Purdom and Amr Sabry's CNF Generator - for code comments and any assistance
// 			paste code into ChatGPT and/or contact GridSAT Stiftung at gridsat.io
//
//
#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <set>
#include <fcntl.h>
#include <atomic>
#include <string>
#include <regex>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
// Third-party library includes
#include <gmpxx.h>
#include <mpi.h>

int dev_null = open("/dev/null", O_WRONLY);
auto _ = dup2(dev_null, STDERR_FILENO);

#define TAG_SOLUTION_COUNT 1

std::string version = "\n NDP-version: 5.2.3";

std::string getWorkingDirectory() {
    char temp[PATH_MAX];
    if (getcwd(temp, sizeof(temp)) != nullptr) {
        return std::string(temp);
    } else {
        return std::string("");
    }
}

std::string formatDuration(double seconds) {
    int months = static_cast<int>(seconds / (60 * 60 * 24 * 30));
    seconds -= months * 60 * 60 * 24 * 30;
    int days = static_cast<int>(seconds / (60 * 60 * 24));
    seconds -= days * 60 * 60 * 24;
    int hours = static_cast<int>(seconds / (60 * 60));
    seconds -= hours * 60 * 60;
    int minutes = static_cast<int>(seconds / 60);
    seconds -= minutes * 60;

    std::stringstream ss;
    if (months > 0) ss << months << " months ";
    if (days > 0) ss << days << " days ";
    if (hours > 0) ss << hours << " hours ";
    if (minutes > 0) ss << minutes << " minutes ";
    ss << std::fixed << std::setprecision(2);
    ss << seconds << " seconds\n";
    return ss.str();
}

std::string getCurrentUTCTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm = *std::gmtime(&now_c);

    std::stringstream ss;
    ss << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S UTC");
    return ss.str();
}

std::string createProblemID(const std::string& input_number, int num_bits, int world_size, const std::string& utcTime) {
    std::stringstream ss;
    ss << input_number << "-" << num_bits << "-" << world_size << "-" << utcTime;

    std::string data = ss.str();
    std::size_t hash_value = std::hash<std::string>{}(data);

    std::stringstream hash_ss;
    hash_ss << std::hex << hash_value;
    return hash_ss.str().substr(0, 16);
}

using Matrix = std::vector<std::vector<int>>;
using PairMatrix = std::pair<Matrix, Matrix>;

PairMatrix ResolutionStep(Matrix A, int i) {
    Matrix LA;
    Matrix RA;

    for (auto &subarray : A) {
        std::vector<int> modified_subarray;
        for (int x : subarray) {
            modified_subarray.push_back(x + i);
        }

        if (std::find(modified_subarray.begin(), modified_subarray.end(), 2 * i) != modified_subarray.end()) {
            continue;
        }

        LA.push_back(modified_subarray);
    }

    for (auto &subarray : LA) {
        for (int &x : subarray) {
            if (x != 0) x -= i;
        }
    }

    for (auto &subarray : A) {
        std::vector<int> modified_subarray;
        for (int x : subarray) {
            modified_subarray.push_back(x - i);
        }

        if (std::find(modified_subarray.begin(), modified_subarray.end(), -2 * i) != modified_subarray.end()) {
            continue;
        }

        RA.push_back(modified_subarray);
    }

    for (auto &subarray : RA) {
        for (int &x : subarray) {
            if (x != 0) x += i;
        }
    }

    return std::make_pair(LA, RA);
}

int choice(const std::vector<std::vector<int>> &A) {
    for (const auto &subarray : A) {
        if (std::count(subarray.begin(), subarray.end(), 0) == 2) {
            for (int x : subarray) {
                if (x != 0) return std::abs(x);
            }
        }
    }

    for (const auto &subarray : A) {
        if (std::count(subarray.begin(), subarray.end(), 0) == 1 && subarray.size() == 3) {
            for (int x : subarray) {
                if (x != 0) return std::abs(x);
            }
        }
    }

    return A.size() > 0 && A[0].size() > 0 ? std::abs(A[0][0]) : 0;
}

bool containsZeroSubarray(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        if (row.size() == 1 && row[0] == 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<int>> Satisfy_iterative(std::vector<std::vector<int>> A, bool firstAssignment = false, bool find_all_assignments = false) {
    std::vector<std::pair<std::vector<std::vector<int>>, std::vector<int>>> stack = {{A, {}}};
    std::vector<std::vector<int>> results;
    std::set<std::vector<int>> unique_results;
    bool found_first_assignment = false;

    while (!stack.empty()) {
        auto [current_A, choices] = stack.back();
        stack.pop_back();
        
        if (containsZeroSubarray(current_A)) {
            continue;
        }

        int i = choice(current_A);
        if (i == 0) {
            if (unique_results.insert(choices).second) {
                results.push_back(choices);
                if (firstAssignment && !find_all_assignments) {
                    found_first_assignment = true;
                    break;
                }
            }
            continue;
        }

        auto [LA, RA] = ResolutionStep(current_A, i);

        if (LA.empty() || std::any_of(LA.begin(), LA.end(), [](const std::vector<int>& subarray) { return subarray == std::vector<int>{0, 0, 0}; })) {
            if (LA.empty()) {
                std::vector<int> new_choices = choices;
                new_choices.push_back(i);
                if (unique_results.insert(new_choices).second) {
                    results.push_back(new_choices);
                    if (firstAssignment && !find_all_assignments) {
                        found_first_assignment = true;
                        break;
                    }
                }
            }
        } else {
            std::vector<int> new_choices = choices;
            new_choices.push_back(i);
            stack.emplace_back(LA, new_choices);
        }

        if (RA.empty() || std::any_of(RA.begin(), RA.end(), [](const std::vector<int>& subarray) { return subarray == std::vector<int>{0, 0, 0}; })) {
            if (RA.empty()) {
                std::vector<int> new_choices = choices;
                new_choices.push_back(-i);
                if (unique_results.insert(new_choices).second) {
                    results.push_back(new_choices);
                    if (firstAssignment && !find_all_assignments) {
                        found_first_assignment = true;
                        break;
                    }
                }
            }
        } else {
            std::vector<int> new_choices = choices;
            new_choices.push_back(-i);
            stack.emplace_back(RA, new_choices);
        }
    }

    return results;
}

std::pair<std::queue<std::pair<Matrix, std::vector<int>>>, int> 
Satisfy_iterative_BFS(Matrix A, int max_iterations, int max_tasks, bool override_max_tasks, int &iterations, int max_queues, int world_rank, size_t &initial_queue_size) {
    std::queue<std::pair<Matrix, std::vector<int>>> queue;
    queue.push(std::make_pair(A, std::vector<int>{}));

    iterations = 0;
    int task_count = 1;

    int previous_task_count = task_count;
    int previous_iterations = iterations;

    while (!queue.empty()) {
        if (max_queues != -1 && queue.size() >= static_cast<size_t>(max_queues)) {
            break;
        }

        if (max_queues == -1 && !override_max_tasks && task_count >= max_tasks) {
            break;
        }

        auto [current_A, choices] = queue.front();
        queue.pop();

        int i = choice(current_A);

        if (i == 0) {
            continue;
        }

        auto [LA, RA] = ResolutionStep(current_A, i);

        if (!LA.empty() && !std::any_of(LA.begin(), LA.end(), [](const std::vector<int>& subarray) { return subarray == std::vector<int>{0, 0, 0}; })) {
            std::vector<int> new_choices = choices;
            new_choices.push_back(i);
            queue.push(std::make_pair(LA, new_choices));
            task_count++;
        }

        if (!RA.empty() && !std::any_of(RA.begin(), RA.end(), [](const std::vector<int>& subarray) { return subarray == std::vector<int>{0, 0, 0}; })) {
            std::vector<int> new_choices = choices;
            new_choices.push_back(-i);
            queue.push(std::make_pair(RA, new_choices));
            task_count++;
        }

        iterations++;

        if ((task_count != previous_task_count || iterations != previous_iterations) && world_rank == 0) {

            std::cout << "\r\033[K  Queue size: " << queue.size() << " - Depth: " << iterations << " - Tasks: " << task_count << std::flush;
            previous_task_count = task_count;
            previous_iterations = iterations;
        }

        if (max_queues == -1 && iterations >= max_iterations) {
            break;
        }
    }
    
	if (initial_queue_size == 0) {
	initial_queue_size = queue.size();
    }
    
    return {queue, task_count};
}

std::vector<std::vector<int>> parseDimacsString(const std::string& data) {
    std::istringstream file(data);
    std::vector<std::vector<int>> result;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c') {
            continue;
        }
        if (line[0] == 'p') {
            continue;
        }

        std::istringstream iss(line);
        std::vector<int> clause;
        int literal;
        int count = 0;

        while (iss >> literal) {
            if (literal == 0) {
                if (!clause.empty() && count == 1) {
                    clause = {0, 0, clause[0]};
                }
                if (!clause.empty()) {
                    result.push_back(clause);
                }
                clause.clear();
                count = 0;
            } else {
                clause.push_back(literal);
                count++;
            }
        }

        if (!clause.empty() && count == 1) {
            clause = {0, 0, clause[0]};
        }
        if (!clause.empty()) {
            result.push_back(clause);
        }
    }

    return result;
}

void ExtractInputsFromDimacs(const std::string& dimacsString, std::vector<int>& v1, std::vector<int>& v2) {
    std::regex regex_first_input(R"(Variables for first input \[msb,...,lsb\]: \[(.*?)\])");
    std::regex regex_second_input(R"(Variables for second input \[msb,...,lsb\]: \[(.*?)\])");

    std::smatch match;

    if (std::regex_search(dimacsString, match, regex_first_input)) {
        std::string numbers = match[1].str();
        std::istringstream iss(numbers);
        std::string number;
        while (std::getline(iss, number, ',')) {
            try {
                v1.push_back(std::stoi(number));
            } catch (const std::invalid_argument& e) {
                std::cout << "\nInvalid argument while converting to int: " << number << std::endl;
                throw;
            } catch (const std::out_of_range& e) {
                std::cout << "\nOut of range error while converting to int: " << number << std::endl;
                throw;
            }
        }
    } else {
        std::cout << "\nError: Could not find 'first input' section in the DIMACS string.\n" << std::endl;
    }

    if (std::regex_search(dimacsString, match, regex_second_input)) {
        std::string numbers = match[1].str();
        std::istringstream iss(numbers);
        std::string number;
        while (std::getline(iss, number, ',')) {
            try {
                v2.push_back(std::stoi(number));
            } catch (const std::invalid_argument& e) {
                std::cout << "Invalid argument while converting to int: " << number << std::endl;
                throw;
            } catch (const std::out_of_range& e) {
                std::cout << "Out of range error while converting to int: " << number << std::endl;
                throw;
            }
        }
    } else {
        std::cout << "\nError: Could not find 'second input' section in the DIMACS string.\n" << std::endl;
    }
}

mpz_class binaryStringToDecimal(const std::string& binaryString) {
    mpz_class result;
    result.set_str(binaryString, 2);
    return result;
}

mpz_class processVector(const std::vector<int>& v, std::vector<int> vec) {
    std::unordered_set<int> v_set(v.begin(), v.end());
    std::string binaryString;

    for (int k : vec) {
        if (v_set.find(k) != v_set.end()) {
            binaryString += '1';
        } else if (v_set.find(-k) != v_set.end()) {
            binaryString += '0';
        } else {
            binaryString += '0';
        }
    }

    return binaryStringToDecimal(binaryString);
}

std::pair<mpz_class, mpz_class> convert(const std::vector<std::vector<int>>& v, const std::vector<int>& v1, const std::vector<int>& v2) {
    if (v.empty()) {
        throw std::runtime_error("\nError: Input vector 'v' is empty.\n");
    }
    const std::vector<int>& firstElement = v[0];
    mpz_class d1 = processVector(firstElement, v1);
    mpz_class d2 = processVector(firstElement, v2);
    
    return {d1, d2};
}

// Define formula for max_tasks 
int calculate_max_tasks(int num_vars, int num_clauses) {

	int max_tasks = (num_clauses - num_vars);
    return max_tasks;
}

void printWorkerCount(const std::string& version, int world_rank, int world_size) {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::vector<char> all_names(world_size * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);

    int worker_count = (world_rank != 0) ? 1 : 0;
    int total_worker_count = 0;
    MPI_Reduce(&worker_count, &total_worker_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        total_worker_count += 1;

        std::map<std::string, int> node_core_count;
        for (int i = 0; i < world_size; ++i) {
            std::string node_name(&all_names[i * MPI_MAX_PROCESSOR_NAME]);
            node_core_count[node_name]++;
        }

		std::cout << "\n" << version << "\n" << std::endl;
		for (const auto& [node_name, core_count] : node_core_count) {
			std::cout << " " << node_name << " - " << core_count << " cores" << std::endl;
		}
		std::cout << "\n Total Cores: " << total_worker_count << std::endl;
    }
}

void printHeadNodeDetails(mpz_class input_number, int num_bits, int num_clauses,
                          int num_vars, int max_tasks, bool override_max_tasks, int depth, int max_queues) {
    std::cout << "\nInput Number: " << input_number << std::endl;
    std::cout << "        Bits: " << num_bits << std::endl;
    std::cout << "     Clauses: " << num_clauses << std::endl;
    std::cout << "        VARs: " << num_vars << std::endl;
    std::cout << std::endl;

    if (max_queues > 0) {
        std::cout << "  Queue size: " << max_queues << std::endl;
    } 
    else if (max_tasks > 0 && !override_max_tasks) {
        std::cout << "  BFS #Tasks: " << max_tasks << std::endl;
    } 
    else if (depth > 0 && override_max_tasks) {
        std::cout << "       Depth: " << depth << std::endl;
    }
    
    std::cout << std::endl;
}
std::string formatPercentage(double part, double total) {
    double percentage = (total > 0.0) ? (part / total) * 100.0 : 0.0;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << percentage << "%";
    return ss.str();
}

std::string formatFilename(const std::string& script_name, const std::string& filename, const std::string& problemID, const std::string& flag) {
    std::string sanitizedFilename = filename;
    size_t pos = sanitizedFilename.find(".dimacs");
    if (pos != std::string::npos) {
        sanitizedFilename = sanitizedFilename.substr(0, pos);
    }

    std::regex numberRegex(R"((\d{5})(\d+))");
    sanitizedFilename = std::regex_replace(sanitizedFilename, numberRegex, "$1e$2");

    std::string shortProblemID = problemID.substr(0, 5);

    std::stringstream ss;
    ss << script_name << "_" << sanitizedFilename << "_" << shortProblemID << "-" << flag << ".txt";

    return ss.str();
}

void exportResultsToFile(const std::string& filename, const std::string& content) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << content;
        outFile.close();
    } else {
        std::cout << "\nError: Could not write to file " << filename << std::endl;
        std::cout << "\n" << std::endl;
	std::terminate();
    }
}

std::string formatVector(const std::vector<int>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

void generate_output(bool solution_found, const std::vector<std::vector<int>>& valid_solutions, 
                     const std::chrono::duration<double>& bfs_duration, 
                     const std::chrono::duration<double>& dfs_duration, 
                     int num_bits, int num_vars, int num_clauses, mpz_class input_number, 
                     int task_count, const std::vector<int>& v1, const std::vector<int>& v2, 
                     const std::string& script_name, const std::string& filename, 
                     const std::string& cli_flag, const std::string& output_directory, 
                     int iterations, const size_t &initial_queue_size, int world_rank, int world_size) {

    std::chrono::duration<double> ndp_duration = bfs_duration + dfs_duration;
    std::ostringstream output_ss;

    output_ss << std::fixed << std::setprecision(2);

    if (solution_found) {
        for (const auto& solution : valid_solutions) {
            auto [d1, d2] = convert({solution}, v1, v2);
            
            output_ss << "\nInput Number: " << input_number << "\n"
                      << "      FACT 1: " << d1 << "\n"
                      << "      FACT 2: " << d2 << "\n";
            output_ss << (d1 * d2 == input_number ? "              verified.\n" : "              FALSE\n");
        }
    } else {
        output_ss << "\n\nInput Number: " << input_number << "\n"
                  << "              Prime!\n";
    }

    output_ss << "\n        Bits: " << num_bits;
    output_ss << "\n        VARs: " << num_vars;
    output_ss << "\n     Clauses: " << num_clauses;

    output_ss << "\n\n    BFS time: " << bfs_duration.count() << " seconds (" 
              << formatPercentage(bfs_duration.count(), ndp_duration.count()) << ")\n"
              << "              " << formatDuration(bfs_duration.count()) << "\n";

    output_ss << "    DFS time: " << dfs_duration.count() << " seconds (" 
              << formatPercentage(dfs_duration.count(), ndp_duration.count()) << ")\n"
              << "              " << formatDuration(dfs_duration.count()) << "\n";

    output_ss << "    NDP time: " << ndp_duration.count() << " seconds\n"
              << "              " << formatDuration(ndp_duration.count()) << "\n";

    output_ss << "\n Total Cores: " << world_size << "\n"
              << "  Queue Size: " << initial_queue_size << "\n"
              << "       Depth: " << iterations << "\n"
              << "       Tasks: " << task_count << "\n";

    output_ss << version << "\n"
              << "      DIMACS: " << filename << "\n";

    std::string utcTime = getCurrentUTCTime();
    output_ss << "   Zulu time: " << utcTime << "\n";
    std::string problemID = createProblemID(input_number.get_str(), num_bits, world_size, utcTime);
    output_ss << "  Problem ID: " << problemID << "\n";

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << output_ss.str();
    
	for (const auto& solution : valid_solutions) {
		output_ss << "\n Assignments: " << formatVector(solution) << "\n";
	}

    std::string input_filename_only = std::filesystem::path(filename).filename().string();
    std::string output_filename = formatFilename(script_name, input_filename_only, problemID, cli_flag);
    std::string full_output_path = output_directory + "/" + output_filename;
    exportResultsToFile(full_output_path, output_ss.str());

    std::cout << "Result saved: " << full_output_path << "\n"
              << "     On node: " << processor_name << "\n" << std::endl;
}

std::vector<int> flattenMatrix(const Matrix& matrix) {
    std::vector<int> flat_matrix;
    for (const auto& row : matrix) {
        flat_matrix.push_back(row.size());
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
    return flat_matrix;
}

Matrix unflattenMatrix(const std::vector<int>& flat_matrix) {
    Matrix matrix;
    for (size_t i = 0; i < flat_matrix.size();) {
        int row_size = flat_matrix[i++];
        std::vector<int> row(flat_matrix.begin() + i, flat_matrix.begin() + i + row_size);
        matrix.push_back(row);
        i += row_size;
    }
    return matrix;
}

std::vector<std::vector<int>> process_queue(
    std::queue<std::pair<std::vector<std::vector<int>>, std::vector<int>>> queue, 
    bool parallel, mpz_class input_number, int num_bits, int num_vars, int num_clauses, 
    std::vector<int>& v1, std::vector<int>& v2, 
    std::chrono::high_resolution_clock::time_point bfs_start, 
    std::chrono::high_resolution_clock::time_point dfs_start, int task_count, 
    const std::string& script_name, const std::string& filename, const std::string& cli_flag, 
    const std::string& output_directory, bool override_max_tasks, int iterations,
    size_t &initial_queue_size, int world_rank, int world_size, MPI_Comm mpi_comm, 
    bool find_all_assignments, const std::chrono::duration<double>& bfs_duration) {

    std::vector<std::vector<int>> valid_solutions;
    std::atomic<bool> solution_found(false);
    std::atomic<int> solution_count(0);

    int flat_matrix_size = 0; 
    int choices_size = 0;
    MPI_Status status;
    
    if (parallel) {
        if (world_rank == 0) {
        	std::cout << std::fixed << std::setprecision(2);
            std::cout << "\n\n    BFS time: " << bfs_duration.count() << " seconds  -  DFS parallel initiated.." 
                      << std::endl;
        }

        auto dfs_start_time = std::chrono::high_resolution_clock::now();

        std::thread time_printer([&]() {
            if (world_rank == 0) {
                while (!solution_found) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - dfs_start_time);
                    std::cout << "\033[2K\r    DFS time: " << elapsed.count() << " seconds" << std::flush;
                }
            }
        });

        if (world_rank == 0) {
            int active_workers_count = world_size - 1;

            while (active_workers_count > 0) {
                MPI_Recv(&flat_matrix_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_comm, &status);

                if (!queue.empty()) {
                    auto task = queue.front();
                    queue.pop();

                    std::vector<int> flat_matrix = flattenMatrix(task.first);
                    flat_matrix_size = flat_matrix.size();
                    choices_size = task.second.size();

                    MPI_Send(&flat_matrix_size, 1, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                    MPI_Send(flat_matrix.data(), flat_matrix_size, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                    MPI_Send(&choices_size, 1, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                    if (choices_size > 0) {
                        MPI_Send(task.second.data(), choices_size, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                    }
                } else {
                    --active_workers_count;
                }

                if (status.MPI_TAG == TAG_SOLUTION_COUNT) {
                    solution_count.fetch_add(1, std::memory_order_relaxed);
                }
            }

            auto dfs_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dfs_duration = dfs_end_time - dfs_start;
            std::chrono::duration<double> ndp_duration = bfs_duration + dfs_duration;

			if (find_all_assignments && solution_count.load() == 0) {
				std::cout << "\n\n\n               NDP terminated - no valid assignments found.\n";
				std::cout << "\n  #Solutions: " << solution_count.load() << "\n";

				generate_output(false, {}, bfs_duration,
								std::chrono::high_resolution_clock::now() - dfs_start,
								num_bits, num_vars, num_clauses, input_number, task_count,
								v1, v2, script_name, filename, cli_flag, output_directory,
								iterations, initial_queue_size, world_rank, world_size);
				std::terminate();
				
			} else if (find_all_assignments) {

				std::cout << "\n\n\n               NDP terminated - no more valid assignments found.\n";
				std::cout << "\n                   All assignments saved in respective files.\n";
				std::ostringstream output_ss;
				output_ss << std::fixed << std::setprecision(2);
				output_ss << "\n  #Solutions: " << solution_count.load() << "\n";
			
				output_ss << "\n    BFS time: " << bfs_duration.count() << " seconds (" 
						  << formatPercentage(bfs_duration.count(), ndp_duration.count()) << ")\n"
						  << "              " << formatDuration(bfs_duration.count()) << "\n";
				output_ss << "FIN DFS time: " << dfs_duration.count() << " seconds (" 
						  << formatPercentage(dfs_duration.count(), ndp_duration.count()) << ")\n"
						  << "              " << formatDuration(dfs_duration.count()) << "\n";
				output_ss << "FIN NDP time: " << ndp_duration.count() << " seconds\n"
						  << "              " << formatDuration(ndp_duration.count()) << "\n";
			
				std::string utcTime = getCurrentUTCTime();
				output_ss << "   Zulu time: " << utcTime << "\n\n\n";
				
				std::string problemID = createProblemID(input_number.get_str(), num_bits, world_size, utcTime);
				output_ss << "  Problem ID: " << problemID << "\n";

				std::cout << output_ss.str();
				
				char processor_name[MPI_MAX_PROCESSOR_NAME];
				int name_len;
				MPI_Get_processor_name(processor_name, &name_len);
			
				std::string input_filename_only = std::filesystem::path(filename).filename().string();
				std::string extended_cli_flag = cli_flag;
				if (find_all_assignments) {
					extended_cli_flag += "_sum";
				}
				std::string output_filename = formatFilename(script_name, input_filename_only, problemID, extended_cli_flag);
				std::string full_output_path = output_directory + "/" + output_filename;
				exportResultsToFile(full_output_path, output_ss.str());
				
				std::cout << "   All saved: " << full_output_path << "\n"
						  << "     On node: " << processor_name << "\n" << std::endl;
			
				std::terminate();
				
			} else if (solution_count.load() == 0) {
			
				generate_output(false, {}, bfs_duration,
								std::chrono::high_resolution_clock::now() - dfs_start,
								num_bits, num_vars, num_clauses, input_number, task_count,
								v1, v2, script_name, filename, cli_flag, output_directory,
								iterations, initial_queue_size, world_rank, world_size);
				std::terminate();
			}

        } else {
            while (true) {
                MPI_Send(&world_rank, 1, MPI_INT, 0, 0, mpi_comm);
                MPI_Recv(&flat_matrix_size, 1, MPI_INT, 0, MPI_ANY_TAG, mpi_comm, &status);
                
                if (status.MPI_TAG != 0) break;

                std::vector<int> flat_matrix(flat_matrix_size);
                MPI_Recv(flat_matrix.data(), flat_matrix_size, MPI_INT, 0, 0, mpi_comm, &status);

                MPI_Recv(&choices_size, 1, MPI_INT, 0, 0, mpi_comm, &status);
                std::vector<int> choices(choices_size);
                if (choices_size > 0) {
                    MPI_Recv(choices.data(), choices_size, MPI_INT, 0, 0, mpi_comm, &status);
                }

                Matrix matrix = unflattenMatrix(flat_matrix);
                auto new_choices = Satisfy_iterative(matrix, false, find_all_assignments);

                for (const auto& nc : new_choices) {
                    std::vector<int> final_choices_i = choices;
                    final_choices_i.insert(final_choices_i.end(), nc.begin(), nc.end());
                    
                    valid_solutions.push_back(final_choices_i);
                    solution_found.store(true);
                    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_SOLUTION_COUNT, mpi_comm);

                    generate_output(true, {final_choices_i}, bfs_duration,
                                    std::chrono::high_resolution_clock::now() - dfs_start,
                                    num_bits, num_vars, num_clauses, input_number, task_count, 
                                    v1, v2, script_name, filename, cli_flag, output_directory, 
                                    iterations, initial_queue_size, world_rank, world_size);

                    if (!find_all_assignments) {
                        std::terminate();
                    }
                }
            }
        }

        time_printer.join();
    }

    return valid_solutions;
}

std::string readFileToString(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "\nError: Could not open file " << filename << std::endl;
        return "";
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void parseCLIOptions(int argc, char* argv[], int& max_queues, int& max_tasks, int& depth, 
                     bool& override_max_tasks, bool& find_all_assignments, std::string& cli_flag) {
    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0]
                  << " <filename> [-d depth | -t max_tasks | -q max_queues] [-a find_all_assignments]\n"
                  << "Note: -t (max_tasks), -d (depth), and -q (max_queues) cannot be used together.\n"
                  << std::endl;
        std::terminate();
    }
    bool has_depth = false, has_tasks = false, has_queues = false;
    cli_flag.clear();
    find_all_assignments = false; // Default to skipping assignments

    std::string filename = argv[1];
    if (filename[0] == '-') {
        std::cout << "\n\n               [ERROR] Missing input filename.\n";
        std::cout << "               Please provide a valid filename as the first argument.\n\n\n";
        std::terminate();
    }
    for (int i = 2; i < argc; ++i) {
        std::string option = argv[i];

        if (option == "-q" && i + 1 < argc) {
            if (has_depth || has_tasks) {
                std::cout << "\n\n               Error: Invalid combination: -q (max_queues) cannot be used with -t (max_tasks) or -d (depth).\n\n\n";
                std::terminate();
            }
            max_queues = std::stoi(argv[++i]);
            cli_flag += "q" + std::to_string(max_queues);
            has_queues = true;

        } else if (option == "-t" && i + 1 < argc) {
            if (has_depth || has_queues) {
                std::cout << "\n\n               Error: Invalid combination: -t (max_tasks) cannot be used with -d (depth) or -q (max_queues).\n\n\n";
                std::terminate();
            }
            max_tasks = std::stoi(argv[++i]);
            depth = max_tasks;
            cli_flag += "t" + std::to_string(max_tasks);
            has_tasks = true;

        } else if (option == "-d" && i + 1 < argc) {
            if (has_tasks || has_queues) {
                std::cout << "\n\n               Error: Invalid combination: -d (depth) cannot be used with -t (max_tasks) or -q (max_queues).\n\n\n";
                std::terminate();
            }
            depth = std::stoi(argv[++i]);
            override_max_tasks = true;
            cli_flag += "d" + std::to_string(depth);
            has_depth = true;

        } else if (option == "-a") {
            find_all_assignments = true;
            if (cli_flag.find("-a") == std::string::npos) {
                cli_flag += "a-";
            }

        } else {
            std::cout << "\n\n               [ERROR] Unknown or invalid argument: " << option << std::endl;
            std::cout << "\n               Usage:\n"
                      << "               <filename> [-d depth | -t max_tasks | -q max_queues] [-a find_all_assignments]\n"
                      << "               Note: -t (max_tasks), -d (depth), and -q (max_queues) cannot be used together.\n\n\n"
                      << std::endl;
            std::terminate();
        }
    }
    if (find_all_assignments && !has_depth && !has_tasks && !has_queues) {
        cli_flag = "auto-a";
    }
    if (!find_all_assignments && !has_depth && !has_tasks && !has_queues) {
        cli_flag = "auto";
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int HEAD_NODE = 0;

    bool find_all_assignments = false;
    int max_queues = -1;
    int total_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int depth = 0;
    int max_tasks = 0;
    int num_bits = 0;
    int num_vars = 0;
    int num_clauses = 0;
    int iterations = 0;
    size_t initial_queue_size = 0;
    bool override_max_tasks = false;
	std::string cli_flag;
    
    if (world_rank == 0) {
        parseCLIOptions(argc, argv, max_queues, max_tasks, depth, override_max_tasks, find_all_assignments, cli_flag);
    }
    std::cout << std::fixed << std::setprecision(2);

    MPI_Bcast(&max_queues, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&depth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&override_max_tasks, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&find_all_assignments, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    int cli_flag_length = cli_flag.size();
    MPI_Bcast(&cli_flag_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cli_flag.resize(cli_flag_length);
    MPI_Bcast(cli_flag.data(), cli_flag_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::string filename = argv[1];
    std::string fileContent = readFileToString(filename);
    if (fileContent.empty()) {
        std::cout << "\n\n               Error: reading file or file is empty.\n\n\n" << std::endl;
        return 1;
    }
    std::smatch match;
    std::regex regex_product(R"(Circuit for product = ([0-9]+) \[)");
    std::regex regex_problem(R"(p cnf ([0-9]+) ([0-9]+))");

    if (std::regex_search(fileContent, match, regex_problem)) {
        num_vars = std::stoi(match[1].str());
        num_clauses = std::stoi(match[2].str());

        std::regex regex_bits(R"(Variables for second input \[msb,...,lsb\]: \[.*?,\s*(\d+)\])");
        if (std::regex_search(fileContent, match, regex_bits)) {
            num_bits = std::stoi(match[1].str());
        }
    } else {
        std::cout << "\nError: Could not extract number of variables and clauses from DIMACS header.\n" << std::endl;
        return 1;
    }
    if (max_tasks == 0 && !override_max_tasks) {
        max_tasks = calculate_max_tasks(num_vars, num_clauses);
        depth = max_tasks;
    }
    mpz_class input_number;
    if (std::regex_search(fileContent, match, regex_product)) {
        input_number.set_str(match[1].str(), 10);
    } else {
        std::cout << "\nError: Could not extract input number from DIMACS header.\n" << std::endl;
        return 1;
    }
    printWorkerCount(version, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == HEAD_NODE) {
        printHeadNodeDetails(input_number, num_bits, num_clauses, num_vars,
                             max_tasks, override_max_tasks, depth, max_queues);
    }
    std::vector<std::vector<int>> clauses = parseDimacsString(fileContent);
    if (clauses.empty()) {
        throw std::runtime_error("\nError parsing DIMACS string.\n");
    }
    std::vector<int> v1;
    std::vector<int> v2;

    ExtractInputsFromDimacs(fileContent, v1, v2);

    auto bfs_start = std::chrono::high_resolution_clock::now();
    auto [results, task_count] = Satisfy_iterative_BFS(clauses, depth, max_tasks, override_max_tasks, iterations, max_queues, world_rank, initial_queue_size);
    auto bfs_end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
	std::chrono::duration<double> bfs_duration = bfs_end - bfs_start; 
    if (world_rank == 0) {
		auto bfs_end = std::chrono::high_resolution_clock::now();
		bfs_duration = std::chrono::duration<double>(bfs_end - bfs_start);
	}
	MPI_Bcast(&bfs_duration, 1, MPI_DOUBLE, HEAD_NODE, MPI_COMM_WORLD);
    
    const size_t &locked_initial_queue_size = initial_queue_size;
    auto dfs_start = std::chrono::high_resolution_clock::now();
    process_queue(results, true, input_number, num_bits, num_vars, num_clauses, v1, v2, bfs_start, dfs_start, task_count,
              std::filesystem::path(argv[0]).stem().string(), filename, cli_flag, getWorkingDirectory(), override_max_tasks, iterations, 
              initial_queue_size, world_rank, world_size, MPI_COMM_WORLD, find_all_assignments, bfs_duration);

	auto dfs_end = std::chrono::high_resolution_clock::now();
	double local_dfs_time = std::chrono::duration<double>(dfs_end - dfs_start).count();
	double global_dfs_time = 0.0;
	
	MPI_Reduce(&local_dfs_time, &global_dfs_time, 1, MPI_DOUBLE, MPI_MAX, HEAD_NODE, MPI_COMM_WORLD);
		
    Matrix A; // BFS input data

	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    close(dev_null);
    return 0;
}