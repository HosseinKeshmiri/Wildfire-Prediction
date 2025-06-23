# Wildfire-Prediction
This repository lays the conceptual groundwork for my wildfire prediction project by generating a **synthetic dataset** tailored for training **UAV-based Deep Reinforcement Learning (DRL) agents**. It simulates key environmental variables relevant to fire risk assessment, including temperature, humidity, and wind speed.

---

### Scenario Overview

A 2D rectangular area is considered, populated with multiple sensor nodes scattered across the region. Each sensor is capable of measuring:

- **Temperature**
- **Humidity**
- **Wind speed**

The area is divided into equal-sized grid cells. A fire is assumed to ignite in the near future, starting in one of the cells. The primary objective is to detect the ignition cell—or at least one of its neighboring cells—as quickly as possible.

---

### UAV-Based DRL Agent

Multiple UAVs operate cooperatively as **DRL agents**, flying over the grid to gather sensor readings. When a UAV visits a cell, it retrieves that cell's environmental data and computes a **Fire Risk Score (FRS)** based on those values. For now, only one UAV with no energy restrictions is assumed to be doing the search mission but it can be extended to a multi-UAV framework.

This score is then used to assign a **reward** for visiting that cell, guiding the DRL agent’s future exploration and fire localization strategy.

---

### Synthetic Data Generation

To simulate realistic sensor readings, I developed a function that generates spatial data patterns across the grid. A fire is assumed to ignite in a known central cell at a certain time, which acts as a reference point for generating environmental dynamics in surrounding cells. The data generated from 78 hours prior the ignition. The closer a cell is to the fire’s origin and ignition time, the more affected its temperature, humidity, and wind speed values will be.

This controlled setup enables the development and testing of DRL agents in a safe, simulated environment before applying them to real-world wildfire detection tasks.

