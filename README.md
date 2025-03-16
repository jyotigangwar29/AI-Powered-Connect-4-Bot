# AI-Powered Connect 4 Bot  

An end-to-end AI-powered **Connect 4 bot** utilizing **CNNs and Transformers** for strategic gameplay. This project integrates **machine learning, cloud deployment, and an interactive web application** to enable real-time AI-driven gameplay.  

## Overview & Features  

This project focuses on training **CNN and Transformer models** on **board position data** and deploying them as a cloud-based AI opponent.  

### Key Features  
- **CNN & Transformer-based AI models** trained to predict optimal moves  
- **Custom dataset generation** using **Monte Carlo Tree Search (MCTS)**  
- **Data augmentation** via board mirroring and randomized move initialization  
- **AWS-hosted inference API** using **Docker** for scalable real-time predictions  
- **Interactive web app** where users can challenge the AI  
- **Real-time AI decision-making** for an engaging gameplay experience  

## Data Generation  

To generate high-quality training data, **Monte Carlo Tree Search (MCTS)** was used, a search algorithm that simulates games by making intelligent move decisions based on probabilities and outcomes. **MCTS bots of varying skill levels played against each other**, ensuring a diverse dataset (~2 million examples) of board states and optimal moves.  

### Enhancements to Dataset Quality  
- Varied starting positions to prevent overfitting  
- Duplicate board states with inconsistent moves were removed for training efficiency  
- Data augmentation through **board mirroring** and **randomized move initialization**, increasing dataset diversity  

### Dataset Encoding  
The board was represented as a **6×7×2 tensor**, with two channels distinguishing player and opponent moves. This format improved feature extraction for the neural networks.  

## Model Development  

### CNN Model  
- Custom convolutional kernels to detect horizontal, vertical, and diagonal patterns  
- Layers include convolution, batch normalization, max pooling, and fully connected layers  
- Achieved up to 77% training accuracy  

### Transformer Model  
- Utilized overlapping patch extraction, positional embeddings, and multi-head self-attention  
- Captured global board relationships, though it was outperformed by the CNN in this specific task   

## Deployment  

### Dockerization & Cloud Hosting  
To make the AI bot accessible:  
- Trained models were containerized using Docker for portable and scalable deployment  
- AWS Lightsail was used for hosting, ensuring low-latency inference  

### Challenges & Fixes  
- Encountered batch shape errors when loading the CNN in Docker  
- Fixed this by aligning package versions using an updated `requirements.txt`  

## Web Application  

The AI bot is available through an interactive **Anvil-based web app**, allowing real-time gameplay.  

### Key Features  
- Grid-based interaction: Click to drop pieces, and the AI instantly responds  
- Choose AI opponent: Play against CNN or Transformer models  
- Restart button: Start new games at any time  
- Game tutorial: Helps users understand how to play  
 

## How to Play  

1. Visit the Web App: [Play Here](https://lnkd.in/eRnXZruh)  
2. Login: Username: `dan`, Password: `Optimization1234`  
3. Choose AI model: CNN or Transformer  
4. Click to drop pieces and play against the AI  

