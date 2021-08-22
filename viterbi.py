"""
viterbi.py
    Description:
        Runs the Viterbi algorithm to predict the most likely tag sequence of a test sentence
        given the hidden markov model (HMM), tags, and observations.
    
    Instructions:
        Run with:
            python viterbi.py <input-test-file>
            
        <input-test-file> must be a file containing the observation sequence as a single line.
            
    Imports:
        sys     - Used to get the arguments from command line
        numpy   - Used for max and argmax operations as well as array formatting
        pickle  - Used for loading pickle files
"""
import sys
import numpy as np
import pickle

if __name__ == "__main__":
    
    # Print an error for the incorrect number of arguments
    if len(sys.argv) != 2:
        sys.exit("ERROR: Incorrect number of arguments.\n\n\tPlease execute with: \n\tpython viterbi.py <input-test-file>")
    
    # Get arguments:
    #   test_file = location of input test file
    test_file = sys.argv[1]
    
    ###########################################################################
    # Get HMM transition probabilities from pickle file
    with open('A.pkl', 'rb') as f:
        A = pickle.load(f)
    
    ###########################################################################
    # Get HMM observation likelihood from pickle file
    with open('B.pkl', 'rb') as f:
        B = pickle.load(f)
    
    # Add dimension to matrix to simplify calculations
    B = np.expand_dims(B, -1)
    
    ###########################################################################
    # Load list of tags and observations from pickle files
    with open('tags.pkl', 'rb') as f:
        tags = pickle.load(f)
    with open('observations.pkl', 'rb') as f:
        observations = pickle.load(f)
    
    ###########################################################################
    # Get observation sequence from file
    testFile = open(test_file)
    lines = testFile.readlines()
    testFile.close()
    
    # Get sentence from test file -> should only be one line
    sentence = lines[0]
    
    # Split line by whitespace
    words = sentence.split()
    
    # Get observation sequence as indices
    o = [observations.index(obs) for obs in words]
    
    # Set N and T
    N = len(tags)
    T = len(o)
        
    ###########################################################################
    # Initialization step for viterbi
    v = np.zeros((N, T))
    v[:, 0] = A[0] * B[:, o[0], 0]
    
    # Initialization step for back trace
    bt = np.zeros((N, T), dtype=int)
    
    ###########################################################################
    # Loop through observations and perform recursive steps
    for t in range(1, T):
        
        # Calculate v * A * B
        vAB = v[:, t - 1] * A[1:].T * B[:, o[t]]
        
        # Get max value for each column in matrix and note index of maximum
        v[:, t] = np.max(vAB, axis=1)
        bt[:, t] = np.argmax(vAB, axis=1)
    
    ###########################################################################
    # Determine best score and last tag corresponding to best score
    best_score = np.max(v[:, T - 1])
    tag = np.argmax(v[:, T - 1])
    
    # Initialize output message
    out = words[-1] + "_" + tags[tag]
    
    # Loop through observation sequence and follow back trace
    for t in reversed(range(1, T)):
        tag = bt[tag, t]
        out = words[t - 1] + "_" + tags[tag] + " " + out
        
    # Output results
    print("Probability = " + str(best_score))
    print("\nMost likely tag sequence:")
    print(out)