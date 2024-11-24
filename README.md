# Non-deterministic Processor (NDP) - Parallel SAT-Solver with OpenMPI for Unlimited Scalability

NDP is an efficient parallel SAT-Solver. It parses DIMACS, factorizes, granulates input DIMACS into subproblems with BFS (Breadth-First Search), and performs parallel DFS (Depth-First Search).  
`NDP 5.1.3` supports custom configuration through command-line options, leveraging OpenMPI for distributed multi-node computation. The NDP outputs dynamically generated file names based on the input parameters and a truncated problem ID hash with the current UTC time.

---

## Requirements

### Input
- Generate DIMACS files at Paul Purdom and Amr Sabry's CNF Generator:  
  [Paul Purdom's CNF Generator](https://cgi.luddy.indiana.edu/~sabry/cnf.html)
- For bit-wise input generation, use:  
  [RSA Challenge](https://bigprimes.org/RSA-challenge)
- Generate DIMACS locally:  
  [CNF_FACT-MULT](https://github.com/GridSAT/CNF_FACT-MULT)  
- Or on IPFS: `ipfs://QmYuzG46RnjhVXQj7sxficdRX2tUbzcTkSjZAKENMF5jba`

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

## Installation

### 1. SSH Setup (Password-less Access)

#### Generate an SSH key on the head node
```bash
ssh-keygen -t rsa -b 4096
```
note: use suggested default location and do not enter password (hit 3x enter)

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

To compile the program on Linux (tested on Ubuntu 24.04.1 LTS), use the following command (compile on every node):
```bash
mpic++ -std=c++17 -Ofast -o NDP-5_1_3 NDP-5_1_3.cpp -lgmpxx -lgmp
```

## CLI usage

Once compiled, the program can be run from the command line using the following format:
```bash
mpirun --use-hwthread-cpus --hostfile <hostfile_path> -quite --mca plm_rsh_args "-q -F <ssh_config_path>" ./NDP-5_1_3 <dimacs_file> [-d depth | -t max_tasks | -q max_queue_size] [-o output_directory]
```
`mpirun` Initializes MPI execution across multiple nodes.  
`--use-hwthread-cpus`		Uses hardware threads for each logical CPU core, maximizing CPU utilization per node.  
`--hostfile <hostfile_path>`	Specifies the file containing the list of nodes and the number of slots (CPUs) each node can contribute.  

###	Command-Line Options:

`<dimacs_file>`: The path to the input DIMACS file.  
`-d` depth: Set a custom depth for BFS iterations. (Optional)  
`-t` max_tasks: Set the maximum number of tasks for BFS. (Optional)  
`-q` max_queues: Limit the maximum number of tasks in the BFS queue. (Optional)  
`-o` output_directory: Specify a custom output directory for the result files. (Optional)  

Basic execution with nodes (example):  
`mpirun --use-hwthread-cpus --hostfile your_hostfile.txt --mca plm_rsh_args "-q -F /home/your_username/.ssh/config" inputs/RSA/rsaFACT-128bit.dimacs`  

On single machine (example):  
`mpirun --use-hwthread-cpus --map-by slot -np <#cores> ./NDP-5_1_3 inputs/RSA/rsaFACT-32bit.dimacs`  

This will run the program using the default settings for BFS and DFS and output the results to the current working directory
with the node setup as specified in `~/.ssh/config` and `your_hostfile.txt` - the node running the command will be head-node, any other connected node will be a worker.
	
Defaults:  
`max_tasks = num_clauses - num_vars`  
`output_directory = input_directory`  
 
Setting a custom depth:					`-d 5000`

Limiting the number of tasks:			`-t 1000`

Setting a custom Queue Size:			`-q 256`

Saving results to a specific directory:	`-o /path/to/output` (NOTE: must exist on every node!!!)


## Monitoring

Monitor system and CPU usage on each node in real time:
```bash
mpstat -P ALL 1
```

## Output

The output file will be saved on the node which found the solution in the format:  
`NDP-5_1_3-<input_file_name>_<truncated_problem_id>_<cli-options>.txt`  
    
Example:  
`NDP-5_1_3_rsaFACT-128bit_8dfcb_auto.txt` (no cli option for Depth/#Tasks/Queue Size)  
`On node: node7`  

## NOTE:
only accepts input generated by [Paul Purdom's CNF Generator](https://cgi.luddy.indiana.edu/~sabry/cnf.html) - for code comments and any assistance
paste code into [ChatGPT](https://chatgpt.com/) and/or [contact GridSAT Stiftung](https://keybase.io/gridsat)
