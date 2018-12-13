from environment import Env
from QLearningAgent import QLearningAgent
from replay_buffer import ReplayBuffer
from visualization import *
import numpy as np
import time
import copy
import rospy
import json
from utils import *
from tqdm import tqdm
class LearningController:
	def __init__(self):
		self.env=None
		self.agent=None

	def one_episode(self,replay,t_max=5000,replay_batch_size=32):
		# print("episode start!")
		watcher=-1
		total_reward=0
		total_err=0
		json.encoder.FLOAT_REPR= lambda o: format(o,'.4f')
		obs = self.env.reset()
		obs=str(np.reshape(obs,[-1]).tolist())
		n_observation=self.env.n_observation
		n_action=self.env.n_action
		self.env.ROSNode.log_axis_clear()
		self.env.ROSNode.log_xy_clear()
		
		for tstep in tqdm(range(t_max)):
			watcher=tstep

			x,y=self.env.target_pos[0],self.env.target_pos[1]

			uav_pos=self.env.ROSNode.get_uav_pos()[0:3]

			self.env.ROSNode.log_xy_add("command_pos",(x,y))
			self.env.ROSNode.log_xy_add("real_pos",(uav_pos[0],uav_pos[1]))
					
			self.env.ROSNode.log_axis_add("error_x",(tstep,x-uav_pos[0]))
			self.env.ROSNode.log_axis_add("error_y",(tstep,y-uav_pos[1]))
			self.env.ROSNode.log_axis_add("error_abs",(tstep,((x-uav_pos[0])**2+(y-uav_pos[1])**2)**0.5))
			
			action = self.agent.get_action(obs)
			# print(action)
			action_list=self.action_lib[action]

		
			next_obs,reward,done=self.env.step(action_list)
			next_obs=str(np.reshape(next_obs,[-1]).tolist())
			total_reward+=reward
			total_err+=(((x-uav_pos[0])**2+(y-uav_pos[1])**2)**0.5)

			self.env.ROSNode.log_axis_add("reward",(tstep,reward))
			self.env.ROSNode.log_axis_add("action_x",(tstep,action_list[0]))
			self.env.ROSNode.log_axis_add("action_y",(tstep,action_list[1]))
			self.env.ROSNode.log_axis_add("action_abs",(tstep,np.linalg.norm(action_list)))

			# print(next_obs)
			self.agent.update(obs,action,reward,next_obs)
			# print("----------------------------")
			# print(obs)
			# print(next_obs)
			# print(reward)
			# print(action)
	
###########################replay buffer############################
			replay.add(obs, action, reward, next_obs, False)
			if tstep>2*replay_batch_size:
				s_batch, a_batch, r_batch, next_s_batch, done_batch = replay.sample(replay_batch_size)

				for i in range(replay_batch_size):
					self.agent.update(s_batch[i], a_batch[i], r_batch[i], next_s_batch[i])
###########################replay buffer############################
			obs=next_obs

		return total_reward/watcher,total_err/watcher		

	def start(self):
		self.env=Env(height=3,discrete=True)
		self.action_lib=[[-1,-1],[-1,1],[-1,0],[0,-1],[0,1],[0,0],[1,-1],[1,1],[1,0]]
		self.agent=QLearningAgent(alpha=0.5, epsilon=0.2, discount=0.9, 
			get_legal_actions = lambda s: range(len(self.action_lib)))

		self.agent.load_param("q_table3.pickle")
		# print("================Qtable:================")
		# for m in self.agent._qvalues:
		# 	print(m,self.agent._qvalues[m])

		rewards = []
		replay = ReplayBuffer(10000)
		for i in range(500):
			mr,me=self.one_episode(replay,t_max=5000)
			rewards.append(mr)   
			
			#OPTIONAL YOUR CODE: adjust epsilon
			# self.agent.epsilon *= 0.99
			print("+++++++++++++++++++++++++++++++++++++++++++++++")
			print("episode = ",i, 'mean reward =', np.mean(rewards[-10:]))
			print("+++++++++++++++++++++++++++++++++++++++++++++++")
			# if i %2 ==0:
			title="mr:"+str(mr)+"me:"+str(me)
			self.env.ROSNode.log_showfigure(title) 
			self.agent.save_param("q_table3.pickle")
			self.agent.save_param("q_table_bak.pickle")

	def test(self):
		x=[]
		y=[]
		for i in range(10000):
			a,b=self.curve(i/10.0)
			x.append(a)
			y.append(b)
		filename="img/"+time.strftime("%d-%H:%M:%S",time.localtime())+".png"
		title=str("mean reward: ")+str(2.0/3)+str("   mean error: ")+str(2.0/4)
		plot_curve(x,y,x,y,title,filename)

if __name__ == '__main__':
	ctrl = LearningController()
	ctrl.start()
	#ctrl.test()