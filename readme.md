#**Batch DQN with High Confidence off-policy evaluation (HCOPE)**

The repository contains implementation of vanilla DQN modified to work in batch setting.
The policies generated by DQN are evaluated with HCOPE for safety critical applications.
Read about [Seldonian Algorithms](https://aisafety.cs.umass.edu/paper.html) 
and [AI Safety research at UMass](https://aisafety.cs.umass.edu/index.html).

## **Folder Guide**
```
|-----batch-dqn  
	|----agent                                     #Codes for running Model			                      
		|----agent.py                          #Agents implementation
		|----dqn.py                            #DQN
	|----data  
	    |----sample_data.zip                       #Sample Data
	    |----data_dqn.csv                          #Fed to DQN
	    |----data_processed.csv                    #Used for HCOPE
	|----e2e.py                                    #Generate and filters policies
	|----params                                    #Network checkpoints
	|----policy                                    #Policies generated by DQN  
	|----policies                                  #HCOPE filtered policies
	|----train.py                                  #Standard DQN trainer  
	|----Utils  
	    |----data_loader.py                        #Processes data for Network
	    |----hcope.py                              #Filter policies with HCOPE
```

## **Quick Start**
1. Unzip the sample data ```unzip sample_data.zip```

2. Install the dependencies using - pip3 install -r requirements.txt

3. Run: python3 e2e.py

```./policy``` contains candidate solutions generated generated by DQN and ```./policies``` evaluates the policies.
If a policy is cannot be said to improve performance with confidence threshold, 
```nsf``` is appended (a short for "no solution found").

## **Usage** ##

Training Parameter for DQN: ```train.py```

```     
    # Training Hyperparameters
    self.num_episodes = int(1e4)        #Training Episodes per policy
    self.num_test_eps = int(1e4)        #Test Episodes per policy
    self.learning_rate = 6e-4           #Learning Rate
    self.gamma = 0.95                   #Discount Factor    
    self.tau = 5e-3                     #Weight Update factor
    self.fc1_dims = 128                 #Network layer 1 dims
    self.fc2_dims = 128                 #Network layer 2 dims

    # Seed
    self.seed = 42                      #Seed boilerplate
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)

    # Environment variables
    self.num_states = 18                #Number of States
    self.num_actions = 4                #Number of Actions
```

HCOPE hyperparams: ```e2e.py```

```
    c = 2                              #Desired value 
    delta = 0.01                       #Lower value is higher confidence
```

## **Citations**
```
@article{thomas2019preventing,
   title={Preventing undesirable behavior of intelligent machines},
   author={Thomas, Philip S. and Castro da Silva, Bruno and Barto, Andrew G. and Giguere, Stephen and Brun, Yuriy and Brunskill, Emma},
   journal={Science},
   year={2019},
   volume={366},
   number={6468},
   pages={999--1004},
   publisher={American Association for the Advancement of Science}
} 
```
