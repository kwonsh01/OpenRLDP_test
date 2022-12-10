import ast

f0 = open("reward.txt", 'r')
f1 = open("hpwl.txt", 'r')
f2 = open("delta_hpwl.txt", 'r')

reward = f0.readline()
hpwl = f1.readline()
hpwl_delta = f2.readline()

reward = ast.literal_eval(reward)
hpwl = ast.literal_eval(hpwl)
hpwl_delta = ast.literal_eval(hpwl_delta)

reward_mean = sum(reward) / len(reward)
hpwl_mean = sum(hpwl) / len(hpwl)
hpwl_delta_mean = sum(hpwl_delta) / len(hpwl_delta)

print("reward_mean: ", reward_mean)
print("hpwl_mean: ", hpwl_mean)
print("hpwl_delta_mean: ", hpwl_delta_mean)