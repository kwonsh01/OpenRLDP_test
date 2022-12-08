import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

import opendp
import time
import copy
import matplotlib.pyplot as plt

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
        self.fc1 = nn.Linear(5, 256)#feature 개수: 6
        self.fc2 = nn.Linear(256, 256)
        self.fc_pi = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256**2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim=0):
        # x.shape = ( N x 6 )
        x = F.normalize(x, dim=0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x.shape = ( N x 256 )
        x = self.fc_pi(x)
        # x.shape = ( N x 1 )
        prob = F.softmax(x, dim=softmax_dim)
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
            
            #reshape action
            A = a.shape
            A = A[0]
            a = a.reshape(A,1,1)
            
            pi_a = pi.gather(1,a).view(A,1) 
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage # r*A
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def read_state(Cell):
    state = []
    for j in range(Cell.size()):
        disp_temp = Cell[j].disp
        # height_temp = Cell[j].height
        id_temp = Cell[j].id
        isTried_temp = Cell[j].moveTry
        overlap_temp = Cell[j].overlapNum
        width_temp = Cell[j].width

        #print(Cell[j].id)
        # state.append([isTried_temp, disp_temp, height_temp, id_temp, overlap_temp, width_temp])
        state.append([isTried_temp, disp_temp, id_temp, overlap_temp, width_temp])
    
    return state

def main():
    print("===========================================================================")     
    print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
    print("   Developers : SangGi Do, Mingyu Woo                                      ")
    print("   RL : SH kwon, SH Kim, CH Lee                                            ")
    print("===========================================================================")

    file = 'nangate45'
    if(file == 'nangate45'):
        argv = "opendp -lef gcd_nangate45/Nangate45_tech.lef -lef gcd_nangate45/Nangate45.lef -def gcd_nangate45/gcd_nangate45_global_place.def -cpu 4 -output_def gcd_nangate45_output.def"
    elif(file == 'des_perf_a_md1'):
        argv = "opendp -lef des_perf_a_md1/tech.lef -lef des_perf_a_md1/cells_modified.lef -def des_perf_a_md1/placed.def -cpu 4 -placement_constraints des_perf_a_md1/placement.constraints -output_def des_perf_a_md1_output.def"
    elif(file == 'des_perf_1'):
        argv = "opendp -lef des_perf_1/tech.lef -lef des_perf_1/cells_modified.lef -def des_perf_1/placed.def -cpu 4 -placement_constraints des_perf_1/placement.constraints -output_def des_perf_1_output.def"

    #post placement
    ckt = opendp.circuit()
    ckt_original = opendp.circuit()
    ckt_original.read_files(argv)
    #Cell = ckt_original.get_Cell()
    
    ckt.copy_data(ckt_original)

    #get cells_list
    Cell = ckt.get_Cell()
    
    model = PPO()
    score = 0.0
    print_interval = 1
    
    reward_arr = []
    
    episode = int(input("episode? "))
    start = time.time()
    for n_episode in range(episode):
        print("[TRAIN] Start New Episode!")
        print("[TRAIN] EPISODE #",n_episode)
        stepN = 0
        #load initial circuit and state
        ckt.copy_data(ckt_original)
        ckt.pre_placement()
        
        #load Cellist and state
        s = read_state(Cell) 
        #print(s)
        done = False

        while not done:
            #step 
            for t in range(T_horizon):
                """""""""
                triedNum = 0
                print("move tried")
                for i in range(Cell.size()):
                    if(Cell[i].moveTry == 1):
                        triedNum += 1
                print(triedNum)
                """
                print("step number:", stepN)
                #action load
                indices = []
                s_List = copy.deepcopy(s)
                k=0
                #print(s)
                print("Cell size: ", len(s))
                for index in range(len(s)):
                    #3: moveTry index
                    if s[index][0] == True:
                        indices.append(index)
                        del s_List[index-k]
                        k += 1
                s_List = torch.tensor(s_List, dtype=torch.float)
                prob = model.pi(s_List)
                probf = prob.flatten()
                probf = probf.tolist()

                for i in indices:
                    probf.insert(i, 0)

                probf = torch.tensor(probf, dtype=torch.float)
                a = Categorical(probf)
                a = a.sample()
                a = a.item()
                
                print("action: ", a)
                
                #placement and reward/done loadj
                ckt.place_oneCell(a)

                r = -1.0 * ckt.reward_calc()
                print("reward: ", r)
                
                stepN += 1
                if (stepN == Cell.size()):
                    done = True
                print("done: ", done)

                #cellist reload and state update
                s_prime = read_state(Cell)
                model.put_data((s, a, r/10.0, s_prime, probf[a].item(), done))
                #print(probf)
                s = s_prime
                #done = True
                #quit()
                score += r
                if done:
                    break

            model.train_net()
        #episode end
        reward_arr.append(-1.0 * (1/r))
        if n_episode%print_interval==0 and n_episode!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_episode, score/print_interval))
            score = 0.0

    print("[TRAIN] End Training!")
    ckt.calc_density_factor(4)
    ckt.write_def("def/"+str(time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time())))+".def")
    ckt.evaluation()
    ckt.check_legality()
    
    end = time.time()
    
    print("Execute time: ", end-start)
    
    domain = np.arange(1, episode + 1, 1)
    plt.plot(domain, reward_arr)
    plt.show()
    print(reward_arr)
    
    print("- - - - - < Program END > - - - - - ")

if __name__ == '__main__':
    main()