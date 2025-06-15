# Wildfire-Prediction
This repository lays the groundwork for my wildfire prediction project by generating a synthetic dataset tailored for training UAV-based deep reinforcement learning (DRL) agents. It simulates environmental variables relevant to fire risk assessment.

In this scenario, a 2D rectangular area is assumed to have multiple sensor nodes scattered in it. each sensor can measure the temperature, humidity, and wind speed. The area is gridded into equal-sized cells and the fire is assumed to ignite sometime in near future and an unknown cell.
The objective is to find the fire cell or at least its one-neighboring cells as soon as possible.

Multiple UAVs are assumed to work cooperatively as DRL agents to find the location of the fire. The UAV flies over a cell and reads the sensor data from that cell. Based on the acquired data, it calculates a Fire Risk Score (FRS). FRS will be used to calculate the reward for visiting that cell.

I created a function to simulate sensor data throughout the area. To do this, I assmued that fire will start in a specific cell in the center of the area. The fire time and location is used as the reference point for other cells. 
