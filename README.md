# Non-deterministic Processor (NDP) - Parallel SAT-Solver with OpenMPI for Unlimited Scalability

This NDP is an efficient parallel SAT-Solver. It parses DIMACS, factorizes, granulates input DIMACS
into independent subproblems with BFS (Breadth-First Search), and performs parallel DFS (Depth-First Search).
`NDP 5.6.7` supports custom configuration through command-line options, leveraging OpenMPI for
distributed multi-node computation ready for spot instances. The NDP outputs dynamically generated
file names based on the input parameters and a truncated problem ID hash with the current UTC time.
Version `5.6.7` runs in spot environments with `-s` and `-r` CLI options to save and resume from BFS results.  

`NDP-5_6_7.cpp` [IPFS CID](https://ipfs.tech): `QmR1cxLgYHxNuihYb11B9NkTTBBVm9RNKgBXf7EjSqwhof`  
`ClauseSetPool.hpp` [IPFS CID](https://ipfs.tech): `QmcM47BsTSgRGvxS7V3AWV1tnCiUxUp3yfWTgdMmFAMRQH`  

---

## Requirements

### Input
- Generate DIMACS files at Paul Purdom and Amr Sabry's CNF Generator:  
  [Paul Purdom's CNF Generator](https://cgi.luddy.indiana.edu/~sabry/cnf.html)
- For bit-wise input generation, use:  
  [RSA Challenge](https://bigprimes.org/RSA-challenge)
- Input file extensions: any & none
  E.g: `.dimacs ⎪ .cnf ⎪ .txt ⎪ .doc ⎪ .xyz ⎪ [filname]`
- Generate DIMACS locally:  
  [CNF_FACT-MULT](https://github.com/GridSAT/CNF_FACT-MULT)  
  or on IPFS: `ipfs://QmYuzG46RnjhVXQj7sxficdRX2tUbzcTkSjZAKENMF5jba`
  

### GMP Library
Ensure the GMP (GNU Multiple Precision Arithmetic Library) is installed to handle big numbers and verify the results.

**Install with:**
```bash
sudo apt install libgmp-dev libgmpxx4ldbl
```
### OpenMPI:
Required for distributed computation across nodes. Install with:
```bash
sudo apt install openmpi-bin libopenmpi-dev
```
### json3:
Required for saving BFS. Install with:
```bash
sudo apt install nlohmann-json3-dev
```

## Installation

### 1. SSH Setup (Password-less Access)

#### Generate an SSH key on the head node
```bash
ssh-keygen -t rsa -b 4096
```
NOTE: use suggested default location and do not enter password (hit 3x enter)

Copy the public key to each node (all nodes must have the respective SSH keys of each other)
```bash
ssh-copy-id -p <port_number> user@hostname
```
Test SSH access (use configured ports)
```bash
ssh -p <port_number> user@hostname
```
## SSH Configuration and MPI Setup

### Default Settings for All Hosts
Add the following configuration to the `~/.ssh/config` file on each node:

```plaintext
# Default settings for all hosts
Host *
    ForwardAgent no
    ForwardX11 no
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h:%p
    ControlPersist 10m
	
# Custom settings for specific hosts
Host node1
	HostName <IP_ADDRESS_1>
	Port <PORT_1>
	User user_name

Host node2
	HostName <IP_ADDRESS_2>
	Port <PORT_2>
	User user_name

Host nodeX
	HostName <IP_ADDRESS_X>
	Port <PORT_X>
	User user_name
```
You can now SSH directly to `node1`, `node2`, `nodeX` without specifying ports or usernames.


###	2. MPI Hostfile (Define Hosts and Slots)

Create a hostfile to specify MPI slots per node (subtract at least 1 core for system on each node).
Save as `your_hostfile.txt`:

```plaintext
node1 slots=<number logic cores - system reserve> or any number >0
node2 slots=<number logic cores - system reserve> or any number >0
```
Example:
```plaintext
node1 slots=24
node2 slots=24
```

###	3. Install Required Libraries

Update system packages and install the required libraries:
```bash
sudo apt update
sudo apt install build-essential
sudo apt install libgmp-dev libgmpxx4ldbl
sudo apt install openmpi-bin libopenmpi-dev
```
check with:
```bash
g++ --version
mpirun --version
which mpirun
```

###	4. Environment Setup

Ensure the environment variables for MPI are set up correctly:
```bash
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
unset DISPLAY
source ~/.bashrc
mkdir -p ~/.ssh/sockets
```
check with:
```bash
echo $LD_LIBRARY_PATH
```

###	5. Permissions

Set permissions for SSH and the MPI hostfile:
```bash
chmod 600 ~/.ssh/config
chmod 644 /path/to/mpi_hostfile
```

## Compilation

To compile the program on Linux (tested on `Ubuntu 24.04.1 LTS`), use the following command (compile on every node):
```bash
mpic++ -std=c++17 -Ofast -march=native -mtune=native -fomit-frame-pointer -funroll-loops -fprefetch-loop-arrays -flto=auto -ffast-math -static-libgcc -static-libstdc++ -o NDP-5_6_7 NDP-5_6_7.cpp -lgmpxx -lgmp -pthread
```
NOTE: make sure to have this file in the working directory `ClauseSetPool.hpp`

## CLI usage

Once compiled, the program can be run from the command line using the following format:
```bash
mpirun --use-hwthread-cpus --map-by slot --hostfile <hostfile_path> --mca plm_rsh_args "-q -F <ssh_config_path>" ./NDP-5_6_7 <dimacs_file> [-d depth | -q max_queues] [-s spot_instance_ready]
```
`mpirun` Initializes MPI execution across multiple nodes.  
`--use-hwthread-cpus`		Uses hardware threads for each logical CPU core, maximizing CPU utilization per node.  
`--hostfile <hostfile_path>`	Specifies the file containing the list of nodes and the number of slots (CPUs) each node can contribute.  

###	Command-Line Options:

`<dimacs_file>`: The path to the input DIMACS file.  
`-d` depth: Set a custom depth for BFS iterations. (Optional)  
`-q` max_queues: Limit the maximum number of BFS queues. (Optional)  
`-s` spot_instance_ready: save BFS on headnode for later use via cli -r [filename.json] (Optional)  
`-r` resume_from_bfs: resume BFS on headnode via cli `-r` `[filename.json]` or `-r` and wait for prompt to chose from list (Optional)
  

Basic execution with nodes (example):  
`mpirun --use-hwthread-cpus --map-by slot --hostfile your_hostfile.txt --mca plm_rsh_args "-q -F ./NDP-5_6_7 /home/your_username/.ssh/config" inputs/RSA/rsaFACT-128bit.dimacs`  

On single machine (example):  
`mpirun --use-hwthread-cpus --map-by slot -np $(nproc) ./NDP-5_6_7 inputs/RSA/rsaFACT-64bit.dimacs`  

This will run the program using the default settings for BFS and DFS and output the results to the current working directory
with the node setup as specified in `~/.ssh/config` and `your_hostfile.txt` - the node running the command will be head-node, any other connected node will be a worker.
	
Defaults:  
BFS depth (max_iterations) = `num_clauses - num_vars + ((world_size /2) * num_bits)`  
`output_directory = input_directory` where the very node which finds an assignments saves locally  
spot-ready: add or remove CPUs/Instances/Clusters on-the-fly  
 
Setting a custom depth:					`-d 5000`

Setting a custom Queue Size:			`-q 256`

NOTE: results are saved into working directory ONLY !!!


## Monitoring

Monitor system and CPU usage on each node in real time:
```bash
mpstat -P ALL 1
```

## Output

The output file will be saved on the node which found the solution in the format:  
`NDP-5_6_7-<input_file_name>_<truncated_problem_id>_<cli-options>.txt`  
    
Example:  
`NDP-5_6_7_rsaFACT-128bit_8dfcb_auto.txt` (no cli option for Depth/Queue Size)  
`On node: node7`

With CLI `-s` BFS is saved on headnode as JSON: `bfs_NDP-5_6_7_<input_file_name>_<truncated_problem_id>-<cli-options>.json`

## NOTE:
only accepts input generated by [Paul Purdom's CNF Generator](https://cgi.luddy.indiana.edu/~sabry/cnf.html) - for code comments and any assistance
paste code into [ChatGPT](https://chatgpt.com/) and/or [contact GridSAT Stiftung](https://keybase.io/gridsat) - [gridsat.io](https://gridsat.eth.limo)
