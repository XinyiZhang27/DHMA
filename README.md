# DHMA

A Dynamics-Aware Hierarchical Multi-Agent (DHMA) deep reinforcement learning approach for UAV-assisted multi-edge collaborative computation offloading problem in vehicular networks.

Four scenarios are simulated with Simulation of Urban MObility (SUMO) (https://sumo.dlr.de/docs/index.html), including two synthetic networks $\mathrm{Grid}_{3\times 3}$ and $\mathrm{Net}_4$, and two real-world networks Pasubio and A.Costa.  
The simulated datasets are available at https://drive.google.com/drive/folders/1oXeOSBLP-BjTtyfzmWor61gPqSVROk26?usp=sharing.
You should download the datasets and build the directory yourself.


# File Structure
* /algorithms: Implementation of UAV and RSU agents.
* /envs: UAV-assisted multi-edge collaborative computation offloading environment.
* /myutils: Implementation of UAV and RSU buffers.
* /runner/separated: Codes for DHMA training and test.
* /st_prediction: Trajectory prediction-based dynamics perception module.
* /sumo: Four simulated datasets.
* main.py: The main function.
* config.py: The configuration file.

# Model Training
1. Download the simulated datasets and put folder "sumo" under folder "DHMA".
2. Under directory /st_prediction:  
   Run data_processing.py to generate the processed data in folder "trajectory_data".  
   Pre-train vehicular trajectory prediction models with main_informer.py. The pre-trained prediction models are saved in folder "checkpoints".
4. In file main.py:  
   Set "--stage" to "train" and run main.py to train MCMA.  
   Note that "--simulation_scenario", "--time_range", and "--num_edge" need to be modified according to different scenarios.

# Model Training
In file main.py:  
Set "--stage" to "train" and run main.py to test MCMA.  
Note that "--model_dir1" and "--model_dir2" should be filled with the paths of the trained UAV and RSU models, respectively.  
Set the test episodes with "start_epi" and "end_epi".
