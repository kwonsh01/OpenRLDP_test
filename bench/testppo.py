import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

import opendp
import copy

#Hyperparameters
learning_rate = 0.001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_pi = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256**2, 1)
    def pi(self, x, softmax_dim=0):
        # x.shape = ( N x 9 )
	#print("x1 is : ", x)
        x = F.normalize(x, dim=0)
        #print("x is : ", x)
        x = F.relu(self.fc1(x))
	#print("x is : ", x)
        x = F.relu(self.fc2(x))
        # x.shape = ( N x 256 )
        x = self.fc_pi(x)
        # x.shape = ( N x 1 )
        prob = F.softmax(x, dim=softmax_dim)
        #prob = F.log_softmax(x, dim=softmax_dim)
        # prob.shape = ( N x 1 )
        return prob

    def v(self, x):
        # x.shape = ( N x 9)    B X (N x 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x.shape = ( N x 256 )
        if len(x.shape) > 2:
            x1 = torch.transpose(x, 1, 2)
        else:
            x1 = torch.transpose(x, 0, 1)
        x = torch.matmul(x1, x)
        A = x.shape
        # x.shape = ( 256 x 256 )
        if len(x.shape) > 2:
            x = torch.reshape(x, (A[0], -1))
        else:
            x = x.flatten()
        # x.shape = ( 256**2 x 1)
        v = self.fc_v(x)
        # v.shape = (1, )
        return v

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        #print(s_lst)
        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]: #revrese in
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage # r*A
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def read_state(Cell):
    state = []

    state = []
    for j in range(Cell.size()):
        disp_temp = Cell[j].disp
        height_temp = Cell[j].height
        id_temp = Cell[j].id
        isTried_temp = Cell[j].moveTry
        overlap_temp = Cell[j].overlapNum
        width_temp = Cell[j].width

        #print(Cell[j].id)
        state.append([disp_temp, height_temp, id_temp, isTried_temp, overlap_temp, width_temp])
    
    #state = np.array(state)
    return state

def main():
    print("===========================================================================")     
    print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
    print("   Developers : SangGi Do, Mingyu Woo                                      ")
    print("   RL : SH kwon, SH Kim, CH Lee                                            ")
    print("===========================================================================")

    # print("1: gcd_nangate45")
    # print("2: des_perf_a_md1")
    # file = input()
    file = '1'
    if(file == '1'):
        argv = "opendp -lef gcd_nangate45/Nangate45_tech.lef -lef gcd_nangate45/Nangate45.lef -def gcd_nangate45/gcd_nangate45_global_place.def -cpu 4 -output_def gcd_nangate45_output.def"
    elif(file == '2'):
        argv = "opendp -lef des_perf_a_md1/tech.lef -lef des_perf_a_md1/cells_modified.lef -def des_perf_a_md1/placed.def -cpu 4 -placement_constraints des_perf_a_md1/placement.constraints -output_def des_perf_a_md1_output.def"

    #post placement
    ckt = opendp.circuit()
    ckt_original = opendp.circuit()
    ckt_original.read_files(argv)
    
    ckt.copy_data(ckt_original)
    
    #get cells_list
    Cell = ckt.get_Cell()
    
    model = PPO()
    score = 0.0
    print_interval = 20
    
    for n_episode in range(100):
        print("[TRAIN] Start New Episode!")
        print("[TRAIN] EPISODE #",n_episode)
        
        #load initial circuit and state
        #s = copy.deepcopy(state)
        ckt.copy_data(ckt_original)
        ckt.pre_placement()
        
        #load Cellist and state
        
        state = read_state(Cell)
        s = state

#https://wikidocs.net/123324        

        done = False

        while not done:
            #step 
            for t in range(T_horizon):
                #action load
                indices = []
                s_List = state
                k = 0
                #print(s)
                print("Cell size: ")
                print(Cell.size())
                for index in range(Cell.size()):
                    if Cell[index].moveTry == True:
                        indices.append(index)
                        del s_List[index-k]
                        k += 1
                s_List = torch.tensor(s_List, dtype=torch.float)
                prob = model.pi(s_List)
                print("s_List size: ")
                print(s_List.size())
                probf = prob.flatten()
                #print(probf)
                probf = probf.tolist()
                for i in indices:
                    probf.insert(i, 0)
                probf = torch.tensor(probf, dtype=torch.float)
                a = Categorical(probf)
                a = a.sample()
                a= a.item()
                print("action: ")
                print(a)
                
                #placement and reward/done load
                ckt.place_oneCell(a)
                r = -1.0 * ckt.reward_calc()
                print("reward: ")
                print(r)
                done = ckt.isDone_calc()
                print("done: ")
                print(done)

                #print("state: ")
                #print(s)
                #cellist reload and state update
                s_prime = read_state(Cell)
                #print("next state: ")
                #print(s_prime)
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                #done = True
                #quit()
                score += r
                if done:
                    break

            model.train_net()
        #episode end
        if n_episode%print_interval==0 and n_episode!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_episode, score/print_interval))
            score = 0.0

    print("[TRAIN] End Training!")
    ckt.calc_density_factor(4)
    ckt.write_def(ckt.out_def_name)
    ckt.evaluation
    ckt.check_legality
    print("- - - - - < Program END > - - - - - ")
            


if __name__ == '__main__':
    main()
