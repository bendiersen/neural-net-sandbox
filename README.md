# neural-net-sandbox
A place to play around with the training of neural nets for basic machine control in simple simulation environments.


# Environment Setup:
run bash run_docker.sh in terminal to create and launch the docker container

for development in vs code open up the devcontainer.json within vscode to properly find all the imports

# Running programs
Once docker container is launched enter into the simple example folder:
cd simple_2d_deliver_pickup/

run the following in terminal to see simple model:
python3 run_simple_policy.py

To train a simple reinforcement learning algorithm run
python3 train_nn.py

To utilize the trained algorithm run the following:
python3 run_trained_model.py

## Simple 2D Deliver Pickup
The goal of this is a machine to pickup an item at the yellow box and then deposit it at the end blue target
Example 1:
![Live Plot](docs/simple_policy1.gif)
Example 2:
![Live Plot](docs/simple_policy2.gif)
