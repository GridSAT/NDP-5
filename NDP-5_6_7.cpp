// NDP-5_6_7.cpp
// 
// NDP - Parallel SAT-Solver with OpenMPI for unlimited scalability
// 
// Copyright (c) 2025 GridSAT Stiftung
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
// into independent subproblems with BFS (Breadth-First Search), and performs parallel DFS (Depth-First Search).
// NDP 5.6.7 supports custom configuration through command-line options, leveraging OpenMPI for
// distributed multi-node computation ready for spot instances. The NDP outputs dynamically generated
// file names based on the input parameters and a truncated problem ID hash with the current UTC time.
// Version 5.6.7 runs in spot environments with -s and -r CLI options to save and resume from BFS results.
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
//   json3:			Required for saving BFS.
//					Install with:
//
//					sudo apt install nlohmann-json3-dev
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
//		sudo apt install nlohmann-json3-dev
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
// 	mpic++ -std=c++17 -Ofast -march=native -mtune=native -fomit-frame-pointer -funroll-loops -fprefetch-loop-arrays -flto=auto -ffast-math -static-libgcc -static-libstdc++ -o NDP-5_6_7 NDP-5_6_7.cpp -lgmpxx -lgmp -pthread
// 
// 
// 	CLI USAGE:
// 
// 	Once compiled, the program can be run from the command line using the following format:
// 
//	mpirun --use-hwthread-cpus --map-by slot --hostfile <hostfile_path> --mca plm_rsh_args "-q -F <ssh_config_path>" ./NDP-5_6_7 <dimacs_file> [-d depth | -q max_queues] [-s spot_instance_ready]
//
// 	mpirun:						Initializes MPI execution across multiple nodes.
// 	--use-hwthread-cpus:		Uses hardware threads for each logical CPU core, maximizing CPU utilization per node.
// 	--hostfile <hostfile_path>:	Specifies the file containing the list of nodes and the number of slots (CPUs) each node can contribute.
//	
// 	Command-Line Options:
// 
//     <dimacs_file>: The path to the input DIMACS file.
//     -d depth: Set a custom depth for BFS iterations. (Optional)
//     -q max_queues: Limit the maximum number of BFS queues. (Optional)
//	   -s spot_instance_ready: save BFS on headnode for later use via cli -r [filename.json] (Optional)
//	   -r resume_from_bfs: resume BFS on headnode via cli -r [filename.json] or -r and wait for prompt to chose from list (Optional)
// 
// 	Basic execution with nodes: mpirun --use-hwthread-cpus --map-by slot --hostfile your_hostfile.txt --mca plm_rsh_args "-q -F ./NDP-5_6_7 /home/your_username/.ssh/config" inputs/RSA/rsaFACT-128bit.dimacs
//           on single machine: mpirun --use-hwthread-cpus --map-by slot -np $(nproc) ./NDP-5_6_7 inputs/RSA/rsaFACT-64bit.dimacs
//
// 	This will run the program using the default settings for BFS and DFS and output the results to the current working directory
// 	with the node setup as specified in `~/.ssh/config` and `your_hostfile.txt` - the node running the command will be head-node, any other connected node will be a worker.
//		
//	Defaults:	BFS depth (max_iterations) = num_clauses - num_vars + ((world_size /2) * num_bits)
//				output_directory = input_directory
//				spot-ready: add or remove CPUs/Instances/Clusters on-the-fly
//	 
// 	Setting a custom depth:					-d 5000
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
//     NDP-5_6_7-<input_file_name>_<truncated_problem_id>_<cli-options>.txt
//     
//     Example: NDP-5_6_7_rsaFACT-128bit_8dfcb_auto.txt (no CLI option for Depth/Queue Size)
//	   On node: node7 
//
//	   With CLI -s BFS is saved on headnode as JSON: bfs_NDP-5_6_7_<input_file_name>_<truncated_problem_id>-<cli-options>.json
//
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
#include <unordered_map>
#include <string>
#include <regex>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <mutex>
#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
#else
    #include <unistd.h>
    #include <sys/sysinfo.h>
#endif
// Third-party library includes
#include <gmpxx.h>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "ClauseSetPool.hpp" // make sure to have this file in the working directory

#ifdef ENABLE_PROFILING
#define PROFILE_SCOPE(name) ScopedTimer timer##__LINE__(name)
#else
#define PROFILE_SCOPE(name)
#endif

#ifdef __GNUC__
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif

// ==========================
// PROFILING INFRASTRUCTURE
// ==========================

std::mutex profiler_mutex;
std::unordered_map<std::string, std::pair<double, int>> profiler_data;

using big_int = mpz_class;

class ScopedTimer {
public:
    ScopedTimer(const std::string &name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()) { }
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start_).count();
        std::lock_guard<std::mutex> lock(profiler_mutex);
        auto &entry = profiler_data[name_];
        entry.first += elapsed;
        entry.second += 1;
    }
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ==========================
// END PROFILING INFRASTRUCTURE
// ==========================

using json = nlohmann::json;
using Matrix = std::vector<std::vector<int>>;

int dev_null = open("/dev/null", O_WRONLY);
auto _ = dup2(dev_null, STDERR_FILENO);

#define TAG_SOLUTION_FOUND 1
#define TAG_REGISTER_WORKER 2
#define TAG_QUEUE_SIZE 3
#define TAG_TASK_DONE 4

std::string version = "\n NDP-version: 5.6.7";

void dumpProfilingResults() {
    std::lock_guard<std::mutex> lock(profiler_mutex);
    std::cout << "\n\n=== Profiling Results ===\n";
    for (const auto &entry : profiler_data) {
        const std::string &func = entry.first;
        double total_time = entry.second.first;
        int calls = entry.second.second;
        std::cout << "Function [" << func << "]: Total time = " << total_time 
                  << " s, Calls = " << calls 
                  << ", Avg = " << (calls ? total_time/calls : 0) << " s\n";
    }
    std::cout << "=========================\n";
}

std::string getWorkingDirectory() {
    PROFILE_SCOPE("getWorkingDirectory");

    char temp[PATH_MAX];
#ifdef _WIN32
    if (_getcwd(temp, sizeof(temp)) != nullptr) {
#else
    if (getcwd(temp, sizeof(temp)) != nullptr) {
#endif
        return std::string(temp);
    } else {
        return std::string("");
    }
}

int get_processor_count() {
    PROFILE_SCOPE("get_processor_count");

#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    ::GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

int hashTask(const std::pair<ClauseSet, std::vector<int>>& task) {
    int hash = 0;
    
    // Hash ClauseSet
    for (const auto& clause : task.first) {
        for (int lit : clause.l) {
            hash ^= lit + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
    }

    // Hash vector of choices
    for (int val : task.second) {
        hash ^= val + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    return hash;
}

std::mutex state_mutex;
std::atomic<bool> solution_found{false};
const std::string bfs_filename = "bfs_results.json";
bool spot_instance_ready = false; // Flag to enable spot-instance readiness

std::string promptForJSONFile(int world_rank) {
    PROFILE_SCOPE("promptForJSONFile");
    if (world_rank != 0) {
        return "";
    }

    std::vector<std::string> bfsFiles;
    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        if (entry.is_regular_file() && entry.path().string().find("bfs_") != std::string::npos) {
            bfsFiles.push_back(entry.path().filename().string());
        }
    }

    if (bfsFiles.empty()) {
        throw std::runtime_error("\n\n Error: No BFS results files found in the current directory.\n");
    }

    std::cout << "\n\n Available BFS results files:\n";
    std::cout << " ============================\n\n";
    for (size_t i = 0; i < bfsFiles.size(); ++i) {
        std::cout << " " << i + 1 << ": " << bfsFiles[i] << "\n";  // Added blank space before each index
    }

    int choice = 0;
    while (true) {
        std::cout << "\n Enter the number of the file to resume from: ";
        if (!(std::cin >> choice) || choice < 1 || choice > static_cast<int>(bfsFiles.size())) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "\n Invalid input. Please enter a number between 1 and " << bfsFiles.size() << ".\n";
        } else {
            break;
        }
    }

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string selectedFile = bfsFiles[choice - 1];
    std::cout << "\n Selected BFS file: " << selectedFile << "\n";
    return selectedFile;
}

std::string formatDuration(double seconds) {
    PROFILE_SCOPE("formatDuration");
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
	PROFILE_SCOPE("getCurrentUTCTime");
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm = *std::gmtime(&now_c);

    std::stringstream ss;
    ss << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S UTC");
    return ss.str();
}

std::string createProblemID(const std::string& input_number, int num_bits, int world_size, const std::string& utcTime) {
    PROFILE_SCOPE("createProblemID");
    std::stringstream ss;
    ss << input_number << "-" << num_bits << "-" << world_size << "-" << utcTime;

    std::string data = ss.str();
    std::size_t hash_value = std::hash<std::string>{}(data);

    std::stringstream hash_ss;
    hash_ss << std::hex << hash_value;
    return hash_ss.str().substr(0, 16);
}

std::string formatFilename(const std::string& script_name, const std::string& filename, const std::string& problemID, const std::string& cli_flag) {
    PROFILE_SCOPE("formatFilename");
    std::string sanitizedFilename = filename;
    size_t pos = sanitizedFilename.find_last_of('.');
    if (pos != std::string::npos) {
        sanitizedFilename = sanitizedFilename.substr(0, pos);
    }

    std::regex numberRegex(R"((\d{5})(\d+))");
    sanitizedFilename = std::regex_replace(sanitizedFilename, numberRegex, "$1e$2");

    std::string shortProblemID = problemID.substr(0, 5);

    std::stringstream ss;
    ss << script_name << "_" << sanitizedFilename << "_" << shortProblemID << "-" << cli_flag << ".txt";

    return ss.str();
}

json serializeBFSResults(const std::queue<std::pair<ClauseSet, std::vector<int>>>& queue, double bfs_duration = 0) {
    PROFILE_SCOPE("serializeBFSResults");
    json j;
    std::queue<std::pair<ClauseSet, std::vector<int>>> temp_queue = queue;

    while (!temp_queue.empty()) {
        auto [clauseSet, choices] = temp_queue.front();
        temp_queue.pop();
        json entry;
        
        // Serialize ClauseSet (vector of Clause3)
        json serialized_clauses;
        for (const Clause3 &cl : clauseSet) {
            serialized_clauses.push_back({cl.l[0], cl.l[1], cl.l[2]});
        }

        entry["clauses"] = serialized_clauses;
        entry["choices"] = choices;
        j["bfs_results"].push_back(entry);
    }

    j["bfs_time"] = bfs_duration;
    return j;
}

std::pair<std::queue<std::pair<ClauseSet, std::vector<int>>>, double> deserializeBFSResults(const json& j) {
    PROFILE_SCOPE("deserializeBFSResults");
    std::queue<std::pair<ClauseSet, std::vector<int>>> queue;

    for (const auto& entry : j["bfs_results"]) {
        ClauseSet clauseSet;
        
        // Deserialize ClauseSet (vector of Clause3)
        for (const auto& cl : entry["clauses"]) {
            Clause3 clause = {cl[0], cl[1], cl[2]};
            clauseSet.push_back(clause);
        }

        std::vector<int> choices = entry["choices"].get<std::vector<int>>();
        queue.emplace(clauseSet, choices);
    }

    double bfs_duration = j.value("bfs_time", 0.0);
    return {queue, bfs_duration};
}

std::string createBFSFilename(const std::string& script_name,
                              const std::string& filename,
                              const std::string& problemID,
                              const std::string& flag) {
    PROFILE_SCOPE("createBFSFilename");
    std::string baseFilename = std::filesystem::path(filename).filename().string();

    size_t pos = baseFilename.find_last_of('.');
    if (pos != std::string::npos) {
        baseFilename = baseFilename.substr(0, pos);
    }

    auto sanitize = [](std::string input) {
        for (char& c : input) {
            if (!isalnum(c) && c != '_' && c != '-') {
                c = '_';
            }
        }
        return input;
    };

    baseFilename = sanitize(baseFilename);
    std::string shortProblemID = sanitize(problemID.substr(0, 5));
    std::string sanitizedFlag = sanitize(flag);

    std::stringstream ss;
    ss << "bfs_" << script_name << "_" << baseFilename << "_" << shortProblemID << "-" << sanitizedFlag << ".json";

    return ss.str();
}

void saveBFSResults(const std::queue<std::pair<ClauseSet, std::vector<int>>>& queue,
                    const std::string& script_name,
                    const std::string& filename,
                    const std::string& problemID,
                    const std::string& flag,
                    const std::string& output_directory,
                    double bfs_duration = 0) {
    PROFILE_SCOPE("saveBFSResults");
    try {
        std::lock_guard<std::mutex> lock(state_mutex);

        json j = serializeBFSResults(queue, bfs_duration);
		
        std::string bfsFilename = createBFSFilename(script_name, filename, problemID, flag);

        if (!std::filesystem::exists(output_directory)) {
            throw std::runtime_error("\n  Output directory does not exist: " + output_directory);
        }
        if (!std::filesystem::is_directory(output_directory)) {
            throw std::runtime_error("\n  Output path is not a directory: " + output_directory);
        }

        std::ofstream outFile(output_directory + "/" + bfsFilename);
        if (!outFile.is_open()) {
            throw std::runtime_error("\n  Could not open file for writing: " + bfsFilename);
        }
        outFile << j.dump(4);
        outFile.close();

        std::cout << "\n\n  BFS results saved to " << bfsFilename << "\n";
    } catch (const std::exception& e) {
        std::cerr << "\n  Error in saving BFS results: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

std::pair<std::queue<std::pair<ClauseSet, std::vector<int>>>, double> readBFSResults(const std::string& filename) {
    PROFILE_SCOPE("readBFSResults");
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        throw std::runtime_error("\n    Error: Could not open " + filename + " for reading.\n");
    }

    json j;
    inFile >> j;
    inFile.close();

    return deserializeBFSResults(j);
}

// === Fixed-Size Clause Representation ===

using ClauseSet = std::vector<Clause3>;

// Parse DIMACS string: convert 1-literal clauses to {0,0,x} and 3-literal clauses to {x,y,z}.
ClauseSet parseDimacsString(const std::string &data) {
    PROFILE_SCOPE("parseDimacsString");
    std::istringstream file(data);
    ClauseSet result;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c' || line[0] == 'p')
            continue;
        std::istringstream iss(line);
        std::vector<int> lits;
        int literal;
        while (iss >> literal && literal != 0)
            lits.push_back(literal);
        if (lits.size() == 1) {
            Clause3 cl = {0, 0, lits[0]};
            result.push_back(cl);
        } else if (lits.size() == 3) {
            Clause3 cl = {lits[0], lits[1], lits[2]};
            result.push_back(cl);
        }
        // Ignore clauses of unexpected size.
    }
    return result;
}

struct ClauseSetBranch {
    ClauseSet cs;
    bool conflict;
};

struct DFSState {
    ClauseSet* state;
    std::vector<int> choices;
    bool conflict;
};
inline std::pair<ClauseSetBranch, ClauseSetBranch> ResolutionStepWithConflict(const ClauseSet &A, int i) __attribute__((always_inline));
inline std::pair<ClauseSetBranch, ClauseSetBranch> ResolutionStepWithConflict(const ClauseSet &A, int i) {
    // Prepare output branch containers.
    ClauseSetBranch branchLA, branchRA;
    branchLA.conflict = false;
    branchRA.conflict = false;
    branchLA.cs.reserve(A.size());
    branchRA.cs.reserve(A.size());
    
    // Process each clause once.
    for (const Clause3 &cl : A) {
        bool skipLA = false;
        bool skipRA = false;
        Clause3 newLA, newRA;
        for (int j = 0; j < 3; ++j) {
            int lit = cl.l[j];
            
            if (lit == i) {
                newRA.l[j] = 0;
                skipLA = true;
            } else if (lit == -i) {
                newLA.l[j] = 0;
                skipRA = true;
            } else {
                newRA.l[j] = lit;
                newLA.l[j] = lit;
            }
        }
        // For LA branch: if not satisfied, check for conflict and add clause.
        if (!skipLA) {
            // If every literal is zero, then the clause is contradictory.
            if (newLA.l[0] == 0 && newLA.l[1] == 0 && newLA.l[2] == 0)
                branchLA.conflict = true;
            branchLA.cs.push_back(newLA);
        }
        // Repeat for RA branch.
        if (!skipRA) {
            if (newRA.l[0] == 0 && newRA.l[1] == 0 && newRA.l[2] == 0)
                branchRA.conflict = true;
            branchRA.cs.push_back(newRA);
        }
    }
    return { std::move(branchLA), std::move(branchRA) };
}

// ResolutionStep: given a ClauseSet and integer i, produce two ClauseSets.
inline std::pair<ClauseSet, ClauseSet> ResolutionStep(const ClauseSet &A, int i) __attribute__((always_inline));
inline std::pair<ClauseSet, ClauseSet> ResolutionStep(const ClauseSet &A, int i) {
    PROFILE_SCOPE("ResolutionStep_Fused");  // Entire function profiling

    // Prepare output clause sets. Reserve memory to avoid repeated allocations.
    ClauseSet LA, RA;
    LA.reserve(A.size());
    RA.reserve(A.size());

    // Outer loop: iterate each clause in A
    {
        PROFILE_SCOPE("ResolutionStep_Fused_ClauseLoop");  // Profile the clause loop as a whole
        for (const Clause3 &cl : A) {
            bool skipLA = false;
            bool skipRA = false;
            Clause3 newLA, newRA;
            // Process each of the three literals directly.
            for (int j = 0; j < 3; ++j) {
                int lit = cl.l[j];
                // LA branch (setting variable i = true):
                // - If the literal is i, then the clause is satisfied so we skip it.
                // - If the literal is -i, then it becomes false (i.e. 0).
                // - Otherwise, leave the literal unchanged.
                if (lit == i)
                    skipLA = true;
                else if (lit == -i)
                    newLA.l[j] = 0;
                else
                    newLA.l[j] = lit;

                // RA branch (setting variable i = false):
                // - If the literal is -i, then the clause is satisfied so we skip it.
                // - If the literal is i, then it becomes false (i.e. 0).
                // - Otherwise, leave the literal unchanged.
                if (lit == -i)
                    skipRA = true;
                else if (lit == i)
                    newRA.l[j] = 0;
                else
                    newRA.l[j] = lit;
            }
            // Only add the new clause to the corresponding branch if it wasn’t skipped.
            if (!skipLA)
                LA.push_back(newLA);
            if (!skipRA)
                RA.push_back(newRA);
        }
    }
    // Return the result using move semantics.
    return { std::move(LA), std::move(RA) };
}

// choice: choose a literal from ClauseSet.
inline int choice(const ClauseSet &A) __attribute__((always_inline));
inline int choice(const ClauseSet &A) {
    PROFILE_SCOPE("choice");
    for (const auto &cl : A) {
        int zeroCount = 0, nonzero = 0;
        for (int j = 0; j < 3; ++j) {
            if (cl.l[j] == 0)
                ++zeroCount;
            else
                nonzero = cl.l[j];
        }
        if (zeroCount == 2)
            return std::abs(nonzero);
    }
    for (const auto &cl : A) {
        int zeroCount = 0, nonzero = 0;
        for (int j = 0; j < 3; ++j) {
            if (cl.l[j] == 0)
                ++zeroCount;
            else
                nonzero = cl.l[j];
        }
        if (zeroCount == 1)
            return std::abs(nonzero);
    }
    if (!A.empty())
        return std::abs(A[0].l[0]);
    return 0;
}

// Satisfy_iterative: DFS search on ClauseSet.
std::vector<std::vector<int>> Satisfy_iterative(ClauseSet A, bool firstAssignment = false) {
    PROFILE_SCOPE("Satisfy_iterative_with_pool");
    ClauseSetPool csPool;  // Use pool as before.
    
    // Instead of (ClauseSet*, vector<int>) pairs, we now use DFSState.
    std::vector<DFSState> stack;
    
    // Obtain initial state from pool. Since the input DIMACS should be conflict–free,
    // we mark it as not conflicted.
    ClauseSet* initialState = csPool.obtain(A.size());
    *initialState = std::move(A);
    stack.push_back({initialState, {}, false});
    
    std::vector<std::vector<int>> results;
    std::set<std::vector<int>> unique_results;
    bool found_first_assignment = false;
    
    while (!stack.empty()) {
        PROFILE_SCOPE("Satisfy_iterative_loop_with_pool");
        
        // Pop a DFS state.
        DFSState current = std::move(stack.back());
        stack.pop_back();
        
        // Instead of scanning with containsZeroSubarray, check our conflict flag.
        if (current.conflict) {
            csPool.release(current.state);
            continue;
        }
        ClauseSet* current_A = current.state;
        std::vector<int> choices = std::move(current.choices);
        
        int i = choice(*current_A);
        if (i == 0) {  // Terminal state: no unassigned variables.
            if (unique_results.insert(choices).second) {
                results.push_back(choices);
                if (firstAssignment) {
                    csPool.release(current_A);
                    break;
                }
            }
            csPool.release(current_A);
            continue;
        }
        
        // Instead of calling ResolutionStep then scanning for conflicts,
        // use ResolutionStepWithConflict to get both the new clause sets and their conflict flags.
        auto branches = ResolutionStepWithConflict(*current_A, i);
        csPool.release(current_A);  // Release current state as before.
        
        // Process LA branch.
        {
            // branchLA conflict flag is set by our new function.
            if (!branches.first.cs.empty() && !branches.first.conflict) {
                ClauseSet* newLA = csPool.obtain(branches.first.cs.size());
                *newLA = std::move(branches.first.cs);
                std::vector<int> new_choices = choices;
                new_choices.push_back(i);
                stack.push_back({newLA, new_choices, branches.first.conflict});
            } else if (branches.first.cs.empty()) {
                std::vector<int> new_choices = choices;
                new_choices.push_back(i);
                if (unique_results.insert(new_choices).second) {
                    results.push_back(new_choices);
                    if (firstAssignment) {
                        break;  // Found a solution, exit loop.
                    }
                }
            }
        }
        
        // Process RA branch.
        {
            if (!branches.second.cs.empty() && !branches.second.conflict) {
                ClauseSet* newRA = csPool.obtain(branches.second.cs.size());
                *newRA = std::move(branches.second.cs);
                std::vector<int> new_choices = choices;
                new_choices.push_back(-i);
                stack.push_back({newRA, new_choices, branches.second.conflict});
            } else if (branches.second.cs.empty()) {
                std::vector<int> new_choices = choices;
                new_choices.push_back(-i);
                if (unique_results.insert(new_choices).second) {
                    results.push_back(new_choices);
                    if (firstAssignment) {
                        break;
                    }
                }
            }
        }
        if (found_first_assignment)
            break;
    }
    return results;
}

// Satisfy_iterative_BFS: BFS search on ClauseSet.
std::pair<std::queue<std::pair<ClauseSet, std::vector<int>>>, int> 
Satisfy_iterative_BFS(ClauseSet A, int max_iterations, bool override_max_iterations, int &iterations, int max_queues, int world_rank, size_t &initial_queue_size) {
    PROFILE_SCOPE("Satisfy_iterative_BFS");
    std::queue<std::pair<ClauseSet, std::vector<int>>> queue;
    queue.push({A, {}});

    iterations = 0;
    int task_count = 1;
    
    int previous_task_count = task_count;
    int previous_iterations = iterations;

    while (!queue.empty()) {
        if (max_queues > 0 && queue.size() >= static_cast<size_t>(max_queues))
            break;
        if (max_iterations == -1 && !override_max_iterations && iterations >= max_iterations)
            break;
        auto [current_A, choices] = queue.front();
        queue.pop();
        int i = choice(current_A);
        if (i == 0)
            continue;
        auto [LA, RA] = ResolutionStep(current_A, i);
        {
            bool conflict = false;
            for (const auto &cl : LA)
                if (cl.l[0] == 0 && cl.l[1] == 0 && cl.l[2] == 0) { conflict = true; break; }
            if (!LA.empty() && !conflict) {
                std::vector<int> new_choices = choices;
                new_choices.push_back(i);
                queue.push({LA, new_choices});
                task_count++;
            }
        }
        {
            bool conflict = false;
            for (const auto &cl : RA)
                if (cl.l[0] == 0 && cl.l[1] == 0 && cl.l[2] == 0) { conflict = true; break; }
            if (!RA.empty() && !conflict) {
                std::vector<int> new_choices = choices;
                new_choices.push_back(-i);
                queue.push({RA, new_choices});
                task_count++;
            }
        }
        iterations++;

        if ((task_count != previous_task_count || iterations != previous_iterations) && world_rank == 0) {

            std::cout << "\r\033[K  Queue size: " << queue.size() << " - Depth: " << iterations << " - Tasks: " << task_count << std::flush;
            previous_task_count = task_count;
            previous_iterations = iterations;
        }

        if (max_queues <= 0 && iterations >= max_iterations)
            break;
    }
    
	if (initial_queue_size == 0) {
	initial_queue_size = queue.size();
    }
    
    return {queue, task_count};
}

void ExtractInputsFromDimacs(const std::string& dimacsString, std::vector<int>& v1, std::vector<int>& v2) {
    PROFILE_SCOPE("ExtractInputsFromDimacs");
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

std::string mpz_to_string(const mpz_class& num) {
    PROFILE_SCOPE("mpz_to_string");
    return num.get_str();
}

big_int binaryStringToDecimal(const std::string& binaryString) {
    PROFILE_SCOPE("binaryStringToDecimal");
    big_int result = 0;
    for (char c : binaryString) {
        result *= 2;
        if (c == '1')
            result += 1;
    }
    return result;
}

big_int processVector(const std::vector<int>& v, std::vector<int> vec) {
    PROFILE_SCOPE("processVector");
    std::string binaryString;
    std::unordered_set<int> v_set(v.begin(), v.end());
    for (int k : vec)
        binaryString += (v_set.find(k) != v_set.end()) ? '1' : '0';
    return binaryStringToDecimal(binaryString);
}

std::pair<big_int, big_int> convert(const std::vector<std::vector<int>>& v, 
                                    const std::vector<int>& v1, 
                                    const std::vector<int>& v2) {
    PROFILE_SCOPE("convert");
    if (v.empty())
        throw std::runtime_error("\nError: Input vector 'v' is empty.\n");
    const std::vector<int>& firstElement = v[0];
    big_int d1 = processVector(firstElement, v1);
    big_int d2 = processVector(firstElement, v2);
    return {d1, d2};
}

// +++ DEFINE MAX_ITERATIONS DEFAULTS +++
// +++ dynamic approach: num_clauses - ((world_size /2) * num_bits)
int calculate_max_iterations(int world_size, int num_clauses, int num_vars, int num_bits) {
    PROFILE_SCOPE("calculate_max_iterations");
    
    int max_iterations = num_clauses - num_vars + ((world_size /2) * num_bits);

    return max_iterations;
}

int printWorkerCount(const std::string& version, int world_rank, int world_size) {
    PROFILE_SCOPE("printWorkerCount");
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
    
	return total_worker_count;
}

void printHeadNodeDetails(mpz_class input_number, int num_bits, int num_clauses,
                          int num_vars, bool override_max_max_iterations, int depth, int max_queues) {
    PROFILE_SCOPE("printHeadNodeDetails");
    std::cout << "\nInput Number: " << input_number << std::endl;
    std::cout << "        Bits: " << num_bits << std::endl;
    std::cout << "     Clauses: " << num_clauses << std::endl;
    std::cout << "        VARs: " << num_vars << std::endl;
    std::cout << std::endl;

    if (max_queues > 0) {
        std::cout << "  Queue size: " << max_queues << std::endl;
    } 
    else if (depth > 0) {
        std::cout << "       Depth: " << depth << std::endl;
    }
    
    std::cout << std::endl;
}

std::string formatPercentage(double part, double total) {
	PROFILE_SCOPE("formatPercentage");
    double percentage = (total > 0.0) ? (part / total) * 100.0 : 0.0;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << percentage << "%";
    return ss.str();
}

void exportResultsToFile(const std::string& filename, const std::string& content) {
    PROFILE_SCOPE("exportResultsToFile");
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << content;
        outFile.close();
    } else {
        std::cout << "\n  Error: Could not write to file " << filename << std::endl;
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
                     const std::vector<int>& v1, const std::vector<int>& v2, 
                     const std::string& script_name, const std::string& filename, 
                     const std::string& cli_flag, const std::string& output_directory, 
                     int iterations, const size_t &initial_queue_size, int world_rank, int world_size,
                     int final_queue_size) {

    PROFILE_SCOPE("generate_output");
    std::chrono::duration<double> ndp_duration = bfs_duration + dfs_duration;
    std::ostringstream output_ss;

    output_ss << std::fixed << std::setprecision(2);
    int processed_queues = initial_queue_size - final_queue_size;

    if (solution_found) {
        for (const auto& solution : valid_solutions) {
            auto [d1, d2] = convert({solution}, v1, v2);

        output_ss << "\n\n                                  Process " << world_rank << " found a solution!\n"
                  << "                    Remaining Queue Size: " << final_queue_size << "\n"
                  << "\nInput Number: " << input_number << "\n"
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
              << "              " << formatDuration(bfs_duration.count()) << "";

    output_ss << "    DFS time: " << dfs_duration.count() << " seconds (" 
              << formatPercentage(dfs_duration.count(), ndp_duration.count()) << ")\n"
              << "              " << formatDuration(dfs_duration.count()) << "";

    output_ss << "    NDP time: " << ndp_duration.count() << " seconds\n"
              << "              " << formatDuration(ndp_duration.count()) << "";

    output_ss << "\n Total Cores: " << world_size << "\n"
              << "  Queue Size: " << initial_queue_size << "\n"
              << "Proc. Queues: " << processed_queues << "\n"
              << "       Depth: " << iterations << "\n";

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

std::vector<int> flattenClauseSet(const ClauseSet& clauses) {
    std::vector<int> flat_data;
    for (const auto& clause : clauses) {
        for (int lit : clause.l) {
            flat_data.push_back(lit);
        }
    }
    return flat_data;
}

ClauseSet unflattenClauseSet(const std::vector<int>& flat_data) {
    ClauseSet clauses;
    for (size_t i = 0; i < flat_data.size(); i += 3) {
        Clause3 clause;
        clause.l[0] = flat_data[i];
        clause.l[1] = flat_data[i + 1];
        clause.l[2] = flat_data[i + 2];
        clauses.push_back(clause);
    }
    return clauses;
}

std::vector<std::vector<int>> process_queue(
    std::queue<std::pair<ClauseSet, std::vector<int>>> queue, 
    bool parallel, big_int input_number, int num_bits, int num_vars, int num_clauses, 
    std::vector<int>& v1, std::vector<int>& v2,
    std::chrono::high_resolution_clock::time_point bfs_start,
    std::chrono::high_resolution_clock::time_point dfs_start, int task_count,
    const std::string& script_name, const std::string& filename, const std::string& cli_flag,
    const std::string& output_directory, int iterations, size_t& initial_queue_size, 
    int world_rank, int world_size, MPI_Comm mpi_comm,
    const std::chrono::duration<double>& bfs_duration) {

    PROFILE_SCOPE("process_queue");
    
    std::vector<std::vector<int>> valid_solutions;
    std::unordered_map<int, std::pair<ClauseSet, std::vector<int>>> task_assignments;
    std::unordered_map<int, bool> task_states;
    std::unordered_map<int, bool> worker_active;
    std::unordered_map<int, std::chrono::steady_clock::time_point> worker_last_seen;

	std::atomic<bool> solution_found(false);
    std::atomic<int> active_workers_count(0);
    std::atomic<int> pending_tasks(0);
    std::atomic<int> total_tasks_assigned(0);

	std::mutex queue_mtx;
    int flat_clause_size = 0, choices_size = 0;
    MPI_Status status;
    int flag = 0;
	ClauseSetPool csPool;

    bool queue_processed = false;

    if (parallel) {
        if (world_rank == 0) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "\n\n    BFS time: " << bfs_duration.count() << " seconds  -  DFS parallel initiated.." 
                      << std::endl;
        }

        auto dfs_start_time = std::chrono::high_resolution_clock::now();
		std::atomic<int> final_queue_size(queue.size());
		
        std::thread time_printer([&]() {
        	PROFILE_SCOPE("time_printer");
            if (world_rank == 0) {
                while (!solution_found.load()) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - dfs_start_time);
                    {
                    	std::lock_guard<std::mutex> lock(queue_mtx);
                    	final_queue_size.store(queue.size(), std::memory_order_relaxed);
						std::cout << "\033[2K\r    DFS time: " << elapsed.count() << " seconds"
								  << "  -  Remaining Queue Size: " << queue.size()
								  << "  -  Active Workers: " << pending_tasks.load()
								  << std::flush;
                	}
                }
            }
        });
        	
        if (world_rank == 0) {
			while ((!queue_processed || pending_tasks.load() > 0) && !solution_found.load()) {
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_comm, &flag, &status);

                if (flag) {
                    MPI_Recv(&flat_clause_size, 1, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, mpi_comm, &status);

                     if (status.MPI_TAG == TAG_SOLUTION_FOUND) {
                        solution_found.store(true);
                        
                        int current_queue_size = queue.size(); 
                        MPI_Send(&current_queue_size, 1, MPI_INT, 
                        status.MPI_SOURCE, TAG_QUEUE_SIZE, mpi_comm);

                    } 
                    else if (status.MPI_TAG == TAG_TASK_DONE) {
                    	int worker_id = status.MPI_SOURCE;
                        pending_tasks.fetch_sub(1, std::memory_order_relaxed);
                    } 
                    else if (flat_clause_size == -1) {
                        if (worker_active[status.MPI_SOURCE]) {
                            worker_active[status.MPI_SOURCE] = false;
                            --active_workers_count;
                        }
                    }
                    else if (status.MPI_TAG == TAG_REGISTER_WORKER) {
                        if (!worker_active[status.MPI_SOURCE]) {
                            worker_active[status.MPI_SOURCE] = true;
                            ++active_workers_count;
                            worker_last_seen[status.MPI_SOURCE] = std::chrono::steady_clock::now();
                        }
                    }
                    else if (!queue.empty()) {
                        auto task = queue.front();
                        queue.pop();

                        pending_tasks.fetch_add(1, std::memory_order_relaxed);
                        total_tasks_assigned.fetch_add(1, std::memory_order_relaxed);
						
                        int task_id = hashTask(task);
                        if (task_states[task_id]) {
                            continue;
                        }

                        task_assignments[status.MPI_SOURCE] = task;
                        task_states[task_id] = true;

                        std::vector<int> flat_clauses = flattenClauseSet(task.first);
                        flat_clause_size = flat_clauses.size();
                        choices_size = task.second.size();

                        MPI_Send(&flat_clause_size, 1, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                        MPI_Send(flat_clauses.data(), flat_clause_size, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                        MPI_Send(&choices_size, 1, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                        if (choices_size > 0) {
                            MPI_Send(task.second.data(), choices_size, MPI_INT, status.MPI_SOURCE, 0, mpi_comm);
                        }
                    } else if (!queue_processed && pending_tasks.load() > 0) {
                        queue_processed = true;
                        std::cout << "\n\n              Queue fully processed. Waiting for workers to complete tasks...\n\n";
                    }
                }

                // Check for inactive workers
                auto now = std::chrono::steady_clock::now();
				for (auto it = worker_last_seen.begin(); it != worker_last_seen.end();) {
					if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count() > 3) {
						int worker_id = it->first;
						if (worker_active[worker_id]) {
							worker_active[worker_id] = false;
							--active_workers_count;
							pending_tasks.fetch_sub(1, std::memory_order_relaxed);
						}
						it = worker_last_seen.erase(it);
					} else {
						++it;
					}
				}

                if (queue_processed && pending_tasks.load() > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));                    
                }
            }

            for (int worker = 1; worker < world_size; ++worker) {
                int termination_signal = -1;
                MPI_Send(&termination_signal, 1, MPI_INT, worker, 0, mpi_comm);
            }

			// Wait for remaining workers to exit (up to 5 seconds)
			auto cleanup_start = std::chrono::steady_clock::now();
			while (pending_tasks.load() > 0 && 
				   std::chrono::steady_clock::now() - cleanup_start < std::chrono::seconds(5)) {
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_comm, &flag, &status);
				if (flag) {
					// Handle any final messages (e.g., worker exits)
					MPI_Recv(&flat_clause_size, 1, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, mpi_comm, &status);
					if (flat_clause_size == -1) {
						if (worker_active[status.MPI_SOURCE]) {
							worker_active[status.MPI_SOURCE] = false;
							--active_workers_count;
						}
					}
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

            if (!solution_found && pending_tasks.load() == 0) {
                generate_output(false, {}, bfs_duration,
                                std::chrono::high_resolution_clock::now() - dfs_start,
                                num_bits, num_vars, num_clauses, input_number, v1, v2,
                                script_name, filename, cli_flag, output_directory,
                                iterations, initial_queue_size, world_rank, world_size, 0);

                std::terminate();
            }
        } else {
        	while (true) {
                MPI_Send(&world_rank, 1, MPI_INT, 0, 0, mpi_comm);
                MPI_Recv(&flat_clause_size, 1, MPI_INT, 0, MPI_ANY_TAG, mpi_comm, &status);
                
                if (flat_clause_size == -1) break;
                
                std::vector<int> flat_clauses(flat_clause_size);
                MPI_Recv(flat_clauses.data(), flat_clause_size, MPI_INT, 0, 0, mpi_comm, &status);

                MPI_Recv(&choices_size, 1, MPI_INT, 0, 0, mpi_comm, &status);
                std::vector<int> choices(choices_size);
                if (choices_size > 0) {
                    MPI_Recv(choices.data(), choices_size, MPI_INT, 0, 0, mpi_comm, &status);
                }

                ClauseSet* clause_set = csPool.obtain();
                *clause_set = unflattenClauseSet(flat_clauses);
                auto new_choices = Satisfy_iterative(*clause_set, true);
                csPool.release(clause_set);
                
                MPI_Send(nullptr, 0, MPI_INT, 0, TAG_TASK_DONE, mpi_comm);

                for (const auto& nc : new_choices) {
                    std::vector<int> final_choices_i = choices;
                    final_choices_i.insert(final_choices_i.end(), nc.begin(), nc.end());
					pending_tasks.fetch_add(1, std::memory_order_relaxed);
					total_tasks_assigned++;
					valid_solutions.push_back(final_choices_i);
					solution_found.store(true);
					MPI_Send(nullptr, 0, MPI_INT, 0, TAG_SOLUTION_FOUND, mpi_comm);
					
					int current_queue_size;
					MPI_Recv(&current_queue_size, 1, MPI_INT, 0, TAG_QUEUE_SIZE, mpi_comm, &status);

                    generate_output(true, {final_choices_i}, bfs_duration,
                                    std::chrono::high_resolution_clock::now() - dfs_start,
                                    num_bits, num_vars, num_clauses, input_number, v1, v2,
                                    script_name, filename, cli_flag, output_directory, 
                                    iterations, initial_queue_size, world_rank, world_size,
                                    current_queue_size);
                    std::terminate();
                }
            }
        }
        time_printer.join();
    }
    return valid_solutions;
}

std::string readFileToString(const std::string& filename) {
    PROFILE_SCOPE("readFileToString");
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "\nError: Could not open file " << filename << std::endl;
        return "";
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void parseCLIOptions(int argc, char* argv[], bool& spot_instance_ready, bool& resume_from_bfs, std::string& bfs_filename_to_resume, int& max_queues,
					 int& depth, bool& override_max_iterations, std::string& cli_flag, int world_rank, int worker_count) {
    PROFILE_SCOPE("parseCLIOptions");
    if (argc < 2) {
        std::cout << "\n               Usage: " << argv[0]
                  << "               <filename> [-s spot_instance_ready] [-r resume_from_bfs] [-d depth | -q max_queues]\n"
                  << "\n"
                  << "               Note: -d (depth) and -q (max_queues) cannot be used together.\n\n\n"
                  << std::endl;
        std::terminate();
    }
    bool has_depth = false, has_queues = false;
    cli_flag.clear();
    
    std::string filename = argv[1];
    if (filename[0] == '-') {
        std::cout << "\n\n               [ERROR] Missing input filename.\n";
        std::cout << "               Please provide a valid filename as the first argument.\n\n\n";
        std::terminate();
    }
    
    for (int i = 2; i < argc; ++i) {
        std::string option = argv[i];

        if (option == "-s") {
            spot_instance_ready = true;
            std::cout << "\n\n              saving BFS after completion!\n\n";
        } else if (option == "-r") {
            resume_from_bfs = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bfs_filename_to_resume = argv[++i];
            } else {
            	if (world_rank == 0) {
                	bfs_filename_to_resume = promptForJSONFile(world_rank);
                }
            }
            
        } else if (option == "-q" && i + 1 < argc) {
            if (has_depth) {
                std::cout << "\n\n               Error: Invalid combination: -q (max_queues) cannot be used with -d (depth).\n\n\n";
                std::terminate();
            }
            max_queues = std::stoi(argv[++i]);
            override_max_iterations = true;
            cli_flag += "q" + std::to_string(max_queues);
            has_queues = true;

        } else if (option == "-d" && i + 1 < argc) {
            if (has_queues) {
                std::cout << "\n\n               Error: Invalid combination: -d (depth) cannot be used with -q (max_queues).\n\n\n";
                std::terminate();
            }
            depth = std::stoi(argv[++i]);
            cli_flag += "d" + std::to_string(depth);
            has_depth = true;

        } else {
            std::cout << "\n\n               [ERROR] Unknown or invalid argument: " << option << std::endl;
            std::cout << "\n               Usage:\n"
                      << "               <filename> [-s spot_instance_ready] [-r resume_from_bfs] [-d depth | -q max_queues]\n"
                      << "\n"
                      << "               Note: -d (depth) and -q (max_queues) cannot be used together.\n\n\n"
                      << std::endl;
            std::terminate();
        }
    }
    if (!has_depth && !has_queues) {
        cli_flag = "auto";
    }
}

int main(int argc, char* argv[]) {
    PROFILE_SCOPE("main");
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int HEAD_NODE = 0;

    int max_queues = -1;
    int total_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int depth = 0;
    int num_bits = 0;
    int num_vars = 0;
    int num_clauses = 0;
    int iterations = 0;
    int task_count = 0;
    int worker_count = 0;
    size_t initial_queue_size = 0;
    bool override_max_iterations = false;
    bool spot_instance_ready = false;
    bool resume_from_bfs = false;

    std::string bfs_filename_to_resume;
    std::string utcTime = getCurrentUTCTime();
	std::string cli_flag;
	std::string script_name = std::filesystem::path(argv[0]).stem().string();
	std::string output_directory = getWorkingDirectory();
    
    if (world_rank == 0) {
        parseCLIOptions(argc, argv, spot_instance_ready, resume_from_bfs, bfs_filename_to_resume, max_queues, depth, override_max_iterations, cli_flag, world_rank, worker_count);
    }
    std::cout << std::fixed << std::setprecision(2);

    MPI_Bcast(&max_queues, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&depth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&override_max_iterations, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

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
    if (depth == 0 && !override_max_iterations) {
        depth = calculate_max_iterations(world_size, num_clauses, num_vars, num_bits);
    }
    mpz_class input_number;
    if (std::regex_search(fileContent, match, regex_product)) {
        input_number.set_str(match[1].str(), 10);
    } else {
        std::cout << "\nError: Could not extract input number from DIMACS header.\n" << std::endl;
        return 1;
    }
    
    std::string problemID = createProblemID(input_number.get_str(), num_bits, world_size, utcTime);
	
    printWorkerCount(version, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printHeadNodeDetails(input_number, num_bits, num_clauses, num_vars,
                             override_max_iterations, depth, max_queues);
    }
    ClauseSet clauses = parseDimacsString(fileContent);
    if (clauses.empty()) {
        throw std::runtime_error("\n  Error parsing DIMACS string.\n");
    }
    std::vector<int> v1;
    std::vector<int> v2;

    ExtractInputsFromDimacs(fileContent, v1, v2);

	std::queue<std::pair<ClauseSet, std::vector<int>>> bfs_queue;

    auto bfs_start = std::chrono::high_resolution_clock::now();
	double bfs_duration = 0.0;
	
	if (resume_from_bfs) {
		if (world_rank == 0) {
			auto [loaded_queue, loaded_duration] = readBFSResults(bfs_filename_to_resume);
			bfs_queue = loaded_queue;
			bfs_duration = loaded_duration;
			std::cout << "\n\n     Loading: " << bfs_filename_to_resume << "\n"
					  << "  Queue size: " << bfs_queue.size() << "\n"
					  << " ET BFS time: " << bfs_duration << " seconds\n";
			}
		MPI_Barrier(MPI_COMM_WORLD);
	} else {
		auto [results, task_count_] = Satisfy_iterative_BFS(clauses, depth, override_max_iterations, iterations, max_queues, world_rank, initial_queue_size);
		auto bfs_end = std::chrono::high_resolution_clock::now();
		bfs_duration = std::chrono::duration<double>(bfs_end - bfs_start).count();
		MPI_Barrier(MPI_COMM_WORLD);		
		bfs_queue = results;
		task_count = task_count_;
	
		if (spot_instance_ready && world_rank == 0) {
			auto bfs_end = std::chrono::high_resolution_clock::now();
			bfs_duration = std::chrono::duration<double>(bfs_end - bfs_start).count();
			saveBFSResults(bfs_queue, script_name, filename, problemID, cli_flag, output_directory, bfs_duration);
		}
	}

	MPI_Bcast(&bfs_duration, 1, MPI_DOUBLE, HEAD_NODE, MPI_COMM_WORLD);

    const size_t &locked_initial_queue_size = initial_queue_size;

	std::chrono::duration<double> bfs_duration_chrono(bfs_duration);

    auto dfs_start = std::chrono::high_resolution_clock::now();
	
	process_queue(bfs_queue, true, input_number, num_bits, num_vars, num_clauses,
				  v1, v2, bfs_start, dfs_start, task_count, script_name, filename,
				  cli_flag, output_directory, iterations,
				  initial_queue_size, world_rank, world_size, MPI_COMM_WORLD,
				  bfs_duration_chrono);
              
	auto dfs_end = std::chrono::high_resolution_clock::now();
	double local_dfs_time = std::chrono::duration<double>(dfs_end - dfs_start).count();
	double global_dfs_time = 0.0;
	
	MPI_Reduce(&local_dfs_time, &global_dfs_time, 1, MPI_DOUBLE, MPI_MAX, HEAD_NODE, MPI_COMM_WORLD);
		
	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    close(dev_null);
    return 0;
}