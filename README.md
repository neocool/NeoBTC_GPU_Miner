GPU Miner
Overview
This is a GPU miner that utilizes Numba and CUDA to mine blocks on the Bitcoin network. It communicates with a Bitcoin node to receive mining templates and then uses the mining template to mine the block.

Requirements
CUDA capable GPU
Numba
CUDA Toolkit
Installation
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/<username>/GPU-Miner.git
Install the required packages:
Copy code
pip install numba
Install the CUDA Toolkit from the NVIDIA website.
Configuration
Open the config.cfg file and set the following parameters:
btcNode_ip: IP address of the Bitcoin node.
btcNode_port: Port number of the Bitcoin node.
btcNode_user: User name for the Bitcoin node.
btcNode_pass: Password for the Bitcoin node.
btc_public_address: Public address for the miner to receive rewards.
miner_id: Unique identifier for the miner.
debug_level: Debug level for the miner. Set to 0 for minimal output.
Running the Miner
Open a terminal and navigate to the directory containing the miner.
Run the miner using the following command:
css
Copy code
python main.py
Contributing
Fork the repository.
Create a new branch for your changes.
Commit your changes and push to the new branch.
Create a pull request to the original repository.
License
This project is licensed under the MIT License - see the LICENSE file for details.
