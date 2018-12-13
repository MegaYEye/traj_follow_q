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
class LearningController:
	def __init__(self):
		self.env=None
		self.agent=None


	def curve(self,tstep):
		if self.env is not None:
			rate=self.env.ROSNode.rate
		else:
			rate=10
		t=1.0*tstep/rate
		r=4
		if t<20:
			r=r*(t/20.0)
		x=r*np.cos(0.5*t)+0.5*np.cos(2*t);
		y=r*np.sin(0.5*t)+0.5*np.cos(0.5*t);

		return x,y

	def form_statebuf(self,state_buf,s):
		s2=s[0:2]
		if len(state_buf)>0:
			state_buf.pop(0)

		while len(state_buf)<2:
			state_buf.append(s2)

	def one_episode(self,t_max=1000,replay=None,replay_batch_size=32):
		# print("episode start!")
		total_reward = 0.0
		total_err=0.0

		s = self.env.reset()
		s=s[0:3]
		state_buf=[]
		watcher=-1
		t=0.0
		last_buf=None
		r_last=None
		echo_dict={}
		wait_step=0
		json.encoder.FLOAT_REPR= lambda o: format(o,'.4f')
		self.env.set_target_pos((0,0,self.env.height))
		traj_x=[]
		traj_y=[]

		traj_x2=[]
		traj_y2=[]


		for tstep in range(t_max):
			watcher=tstep


			uav_pos=self.env.ROSNode.get_uav_pos()[0:3]
			traj_x.append(uav_pos[0])
			traj_y.append(uav_pos[1])

			s, done=self.env.get_state()
			#we only consider x,y,z
			x1,y1,z1,_,_,_,_=self.env.ROSNode.get_uav_pos()

			echo_dict["z"]=z1
			
			if done:
				break

			r_cur=self.env.get_gaussian_score((x1,y1,z1))
			
			x,y=self.curve(tstep-wait_step)
			traj_x2.append(x)
			traj_y2.append(y)
			echo_dict["tar_pos"]=(x,y)
			self.env.set_target_pos((x,y,self.env.height))

			self.form_statebuf(state_buf,s)
			np_cur_state=np.array([state_buf]).reshape(1,-1)

			cur_state=str(np_cur_state.tolist())
			


			if last_buf!=None:
				np_last_state=np.array([last_buf]).reshape(1,-1)
				last_state=str(np_last_state.tolist())
				# r=r_cur-r_last
				r=r_cur
				total_reward+=r
				err=get_dis(x1,y1,0,x,y,0)
				echo_dict["cur_err"]=err
				total_err+=err
				echo_dict["a"]=a
				echo_dict["r"]=r
				echo_dict["s"]=cur_state
				echo_dict["q_before"]=self.agent.get_str_qvalue(last_state)

				

				
				# echo_dict["old_q"]=self.agent.get_all_qvalue(np_last_state).tolist()
				self.agent.update(last_state,a,r,cur_state)

				echo_dict["q_after"]=self.agent.get_str_qvalue(last_state)
				# echo_dict["new_q"]=self.agent.get_all_qvalue(np_last_state).tolist()
###########################replay buffer############################
				if replay is not None:

					replay.add(last_state, a, r, cur_state, False)

					s_batch, a_batch, r_batch, next_s_batch, done_batch = replay.sample(replay_batch_size)

					for i in range(replay_batch_size):
						self.agent.update(s_batch[i], a_batch[i], r_batch[i], next_s_batch[i])
				

				

###########################replay buffer############################
			
	


			a=self.agent.get_action(cur_state)
			a=0
			r=self.env.step(a)
			last_buf=copy.deepcopy(state_buf)
			r_last=r_cur

			js=json.dumps(echo_dict,sort_keys=True,indent=4,separators=(',',':'))
			rospy.loginfo_throttle(1,js)
			
		

		print("episode end at step: ",watcher)
		filename="img/"+time.strftime("%d-%H:%M:%S",time.localtime())+".png"
		title=str("mean reward: ")+str(total_reward/watcher)+str("   mean error: ")+str(total_err/watcher)
		plot_curve(traj_x,traj_y,traj_x2,traj_y2,title,filename)
	        
		return total_reward/watcher		

	def start(self):
		self.env=Env(height=3)

		self.agent=QLearningAgent(alpha=0.5, epsilon=0.2, discount=0.9, 
			get_legal_actions = lambda s: range(len(self.env.actions)))

		self.agent.load_param("q_table3.pickle")
		# print("================Qtable:================")
		# for m in self.agent._qvalues:
		# 	print(m,self.agent._qvalues[m])

		rewards = []
		replay = ReplayBuffer(1000)
		for i in range(500):
			
			rewards.append(self.one_episode(2000,replay=replay))   
			
			#OPTIONAL YOUR CODE: adjust epsilon
			# self.agent.epsilon *= 0.99
			print("+++++++++++++++++++++++++++++++++++++++++++++++")
			print("episode = ",i, 'mean reward =', np.mean(rewards[-10:]))
			print("+++++++++++++++++++++++++++++++++++++++++++++++")
			# if i %2 ==0:

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