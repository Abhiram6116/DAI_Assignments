import numpy as np
import matplotlib.pyplot as plt

def SimulateGaltonBoard(h, N):
    # h is the depth and N is the number of balls simulated
    
    # Each ball starts at 0.
    pockets = np.zeros(N, dtype=int) 
    
    # We need to simulate h random numbers each being 1 or -1
    steps = np.random.randint(0, 2, (N, h)) 
    # in step we get 0's and 1's.
    # To get 1's and -1's we use 2*steps-1
    steps = 2*steps - 1
    
    # Now we need to sum steps along axis = 1 to get net movement.
    steps = steps.sum(axis = 1)
        
    # Now add them to initial positions.
    pockets = pockets + steps 
    
    # Now we have positions of each ball which is a sequence of only odds or evens depending on h.
    # We need to use floor(pos/2) to map them to h+1 pockets
    pockets = (np.floor(pockets/2)).astype(int)
    # We now have the data of pockets of all N.
    
    # The possible pocket indexes are:
    pocket_indexes = np.floor(np.array(range(-h, h+1, 2))/2).astype(int)
    # Now we need to count number of balls in each index.
    balls_per_pocket = np.zeros(h+1)
    for pocket in pockets:
        balls_per_pocket[(pocket+((h+1)//2))]+=1
        
    # Now normalize using N.
    balls_per_pocket = balls_per_pocket/N 
    return pocket_indexes, balls_per_pocket

num_balls = 100000
depths = [10, 50, 100]

# Generate plot for each depth
i = 1
for depth in depths:
    pocket_indexes, balls_per_pocket = SimulateGaltonBoard(depth, num_balls)
    plt.figure(figsize=(10, 6))
    plt.bar(pocket_indexes, balls_per_pocket, width = 0.5, color='blue', edgecolor='black')
    plt.xlabel('Pocket')
    plt.ylabel('Normalized count')
    plt.title(f'Galton Board Simulation with Depth {depth}')
    plt.savefig(f'images/2d{i}.png')
    i+=1
    plt.show()
