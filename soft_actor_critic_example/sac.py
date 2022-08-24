import torch
from model import QNetwork
from torch.optim import Adam
from model import GaussianPolicy, DeterministicPolicy

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        #Set up hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval

         
        self.policy_type = args.policy
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        #Send to GPU if available
        self.device = torch.device("cuda" if args.cuda else "cpu")

        #add the 
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)

        #optimizer setup for the critic
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        if self.policy_type == "Gaussian":
            # Target Entropy = -dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    
