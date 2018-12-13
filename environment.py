import rospy
from random import randint
from utils import *
import numpy as np
from ROSNode import ROSNode



class Env:
	def __init__(self,height):
		self.ROSNode=None
		self.target_pos=None
		self.height=height

		self.limitbox=[-20,20,-20,20,height-2,height+2]

		
		self.local_position = None
		self.actions=[(0,0),
		(0,-1),(0,1),(-1,0),(1,0)]

		self.alpha=0.5
		self.state_buf=[]
		
	def wait(self):
		self.ROSNode.hover(1)

	#reset all environment
	def reset(self):
		self.ROSNode=ROSNode()
		self.ROSNode.start()
		self.wait()
		self.set_target_pos((0,0,self.height))
		s,_=self.get_state()
		return s

	def set_target_pos(self,target_pos):
		self.target_pos=target_pos

	def reachable(self,pos):
		assert self.target_pos!=None
		x1,y1,z1=self.target_pos[0],self.target_pos[1],self.target_pos[2]
		x2,y2,z2=pos[0],pos[1],pos[2]
		# we do not consider z axis
		return get_dis(x1,y1,1,x2,y2,1)<2

	def get_state(self):
		done=False
		x,y,z,qx,qy,qz,qw=self.ROSNode.get_uav_pos()
		tx,ty,tz=self.target_pos[0],self.target_pos[1],self.target_pos[2]
		# print(x,y,z,self.limitbox)
		if x<self.limitbox[0] or x>self.limitbox[1] or y<self.limitbox[2] or y>self.limitbox[3] or z<self.limitbox[4] or z>self.limitbox[5]:
			done=True
		if get_dis(x,y,1,tx,ty,1)>5:
			done=True

		sx=round((x-tx))+3
		sy=round((y-ty))+3
		sz=round((y-tz))+3

		sx=np.clip(sx,0,6)
		sy=np.clip(sy,0,6)
		sz=np.clip(sz,0,6)

		
		return (sx,sy,sz,qx,qy,qz,qw),done


	#test+visual	
	def get_gaussian_score(self,uav_pos):
		score=0
		sigma_p2=3
		# attenuation=0.8
		x1,y1,z1=self.target_pos[0],self.target_pos[1],self.target_pos[2]
		x2,y2,z2=uav_pos[0],uav_pos[1],uav_pos[2]
	
		dis=get_dis(x1,y1,0,x2,y2,0)
		score+=5*np.exp(-( dis**2 / ( sigma_p2 ) ) )

		return score-0.5


	# def calculate_reward(self,pos_old,pos_new):
	# 	old_gaussian_score=self.get_gaussian_score(pos_old)
	# 	new_gaussian_score=self.get_gaussian_score(pos_new)
	# 	return new_gaussian_score-old_gaussian_score

	def calculate_reward(self,pos_old,pos_new):
		old_gaussian_score=self.get_gaussian_score(pos_old)
		new_gaussian_score=self.get_gaussian_score(pos_new)
		return new_gaussian_score
	def step(self,act):



		# print(act)
		tx=self.actions[act][0]*self.alpha+self.target_pos[0]
		ty=self.actions[act][1]*self.alpha+self.target_pos[1]
		tz=self.target_pos[2]
		self.ROSNode.moveto(tx,ty,tz)
		self.ROSNode.ros_interact_step()
		s,done=self.get_state()


		# print("reward:",r)


	def test(self):
		self.set_target_pos((1,1,1))
		print("==============gaussian score==============")
		print(self.get_gaussian_score((1,1,1)))
		print(self.get_gaussian_score((1,0.5,1)))
		print(self.get_gaussian_score((0.5,0.5,1)))
		print("==============calculate reward==============")
		print(self.calculate_reward((1,1,1),(0,0,1)))
		print(self.calculate_reward((1,1,1),(1,1,1)))
		print(self.calculate_reward((1,2,1),(1,1,1)))
		print("==============reachable==============")
		print(self.reachable((1,1,1)))
		print(self.reachable((1,1.4,1)))
		print(self.reachable((1,2,1)))


	
if __name__ == '__main__':
    e=Env(10)
    e.test()

