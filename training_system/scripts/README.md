# Laelaps ENV V0: FG

Action space X: [-0.1,0.1]
	     Y: [-0.49:-0.59]

Results analysis: 

* Conerged after some time to produce very small steps so it does not fall, therefore the x action space needs to expand but without causing the robot to explode (reach singularity)

# Laelaps ENV V1: FG

Action space X: [-0.2,0.2]
	     Y: [-0.49:-0.595]

If the inverse kinmatics calculation of the toe commands (x,y) gives Nan angles, then don't step in the simulation and give reward =-1 and continue by giving the observations of the current momment (a bit different than the state before the actions that gives NaN motor values)

Results analysis:



# Laelaps ENV Ellipse V0: EG

Action space X: [-0.1,0.1]
	     Y: [-0.58,-0.50]

Add ellipse trajectory 
Results analysis: 

