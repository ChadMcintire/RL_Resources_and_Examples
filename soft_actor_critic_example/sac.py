import torch
from model import QNetwork
from torch.optim import Adam
from model import GaussianPolicy, DeterministicPolicy
import torch.nn.functional as F
from utils import soft_update, hard_update
from torchinfo import summary

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

        #set up the critic, which will be a Q function approximator 
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
    
        #print a summary of the tensor model
        #this summary is based on the forward, so the left is the shape of the state and the right, the action space
        #for example see https://github.com/sksq96/pytorch-summary#multiple-inputs
        print("\n\n\n\nSummary of the Sac critic")
        summary(self.critic, [(args.batch_size, num_inputs)  ,(args.batch_size, action_space.shape[0])])

        #optimizer setup for the critic
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        
        #set up the critic_target, which will be a Q function approximator 
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        #print a summary of the tensor model
        print("\n\n\n\nSummary of the Sac target")
        summary(self.critic_target, [(args.batch_size, num_inputs)  ,(args.batch_size, action_space.shape[0])])

        #this is interesting, initializing a new target, generally I see a deep copy of the critic
        #not necessary to have here because it worked without it but might speed up convergence
        hard_update(self.critic_target, self.critic)

        #???why does the Gaussian policy employ automatic tuning and the deterministic not
        if self.policy_type == "Gaussian":
            # Target Entropy = -dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

            #print a summary of the tensor model
            print("\n\n\n\nSummary of the Gaussian Policy")
            summary(self.policy, [(args.batch_size, num_inputs)])

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

            #print a summary of the tensor model
            print("\n\n\n\nSummary of the Deterministic Policy")
            summary(self.policy, [(args.batch_size, num_inputs)])

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    """
    *******************************************************************
    *
    *The action selected is the current policy action or the mean after
    *the policy is trained
    *
    *******************************************************************
    """
    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluation is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    """
    *******************************************************************
    *
    *
    *
    *
    *******************************************************************
    """
    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        #Send devices to GPU if available
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        #As we are producing values to be used for our forward pass, a no grad is appropriate
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)

            #compute equation 3 from Soft Actor critic with Applications
            #??? unsure why we use the min of the 2 networks, need to read up why we would have a positive bias
            #Soft Actor critic with Application page 8, section 6 states we use the min of the 2 networks to 
            #speed up training and mitigate positive bias in policy improvement step
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
  
            #don't update value if we hit a terminal state
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        #forward pass on the critic network 
        qf1, qf2 = self.critic(state_batch, action_batch) #Two Q-functions to mitigate positive bias in the policy improvement step


        #this is on Soft Actor critic with Application page 6, equation 7
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = Expectation(st, at) ~ D[0.5(Q1(st,at) - r(st,at) - gamma(expectation_st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = Expectation(st, at) ~ D[0.5(Q1(st,at) - r(st,at) - gamma(expectation_st+1~p[V(st+1)]))^2]

        #compute both losses at the same time for efficiency
        #https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/3
        qf_loss = qf1_loss + qf2_loss

        #reset gradient for new batch for the critic
        self.critic_optim.zero_grad()

        #compute the gradients of the loss with respect to the modeters
        qf_loss.backward()

        #this is where the critic is actually updated
        self.critic_optim.step()

        
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)

        #Soft Actor critic with Application page 8, section 6 states we use the min of the 2 networks to 
        #speed up training and mitigate positive bias in policy improvement step
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        #this is on Soft Actor critic with Application page 6, equation 9
        #The .mean() is used for the expectation in the formula
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        #reset the policy gradient
        self.policy_optim.zero_grad()

        #compute policy gradient
        policy_loss.backward()

        #update policy parameters
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            #this is on Soft Actor critic with Application page 7, equation 17
            #same equation, just distribute the log_alpha and -1
            #the .mean() call comes from the #expectation in the formula
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        
            self.alpha = self.log_alpha.exp()
            alpha.tlogs = self.alpha.clone() #For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) #For TensorboardX logs 

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    """
    *******************************************************************
    *
    *Save the Policy, Critic, Critic target, and the Critic and Policies
    * optimizers
    *
    *******************************************************************
    """
    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Save models to {}".format(ckpt_path))
        torch.save({"policy_state_dict": self.policy.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "critic_target_state_dict": self.critic_target.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                    "policy_optimizer_state_dict": self.policy_optim.state_dict()}, ckpt_path)


    """
    *******************************************************************
    *
    *Load the Policy, Critic, Critic target, and the Critic and Policies
    * optimizers
    *
    *******************************************************************
    """
    #Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()

            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

