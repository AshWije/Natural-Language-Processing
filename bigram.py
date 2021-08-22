"""
bigram.py
    Description:
        This program builds a word-based bigram model from a given training corpus and shows all
        the unigram/bigram counts and probabilities of the model. This built model can then be used
        for computing the bigram based probability of a given test sentence.
    
    Instructions:
        For building a word-based bigram model from the given training corpus and displaying the
        unigram/bigram counts and probabilities, run:
            python bigram.py <smoothing-type>
        
        For computing the bigram based probability of a test sentence, run:
            python bigram.py <smoothing-type> <input-test-file>
            
        <smoothing-type> must be one of the following values:
            (1) none
            (2) add-one
            (3) good-turing
            (4) add-one-fast
            
            Note:   add-one-fast performs add-one smoothing without computing ALL possible bigrams at
                    once. Instead, the bigram probability is calculated dynamically while testing.
                    (I recommend using this as add-one runs too slowly otherwise)
        
        <input-test-file> must be a file containing the test input sentence as a single line.
        
    Imports:
        sys - Used to get the arguments from command line
"""
import sys

if __name__ == "__main__":
    
    # Print an error for the incorrect number of arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("ERROR: Incorrect number of arguments.\n\n\tPlease execute with either: \n\t(1) python bigram.py <smoothing-type>\n\t(2) python bigram.py <smoothing-type> <input-test-file>")
    
    # Get arguments:
    #   smoothing_type = {'none', 'add-one', 'good-turing', 'add-one-fast'}
    #   test_file = location of input test file (optional)
    smoothing_type = sys.argv[1]
    test_file = None
    if len(sys.argv) == 3:
        test_file = sys.argv[2]
    
    # Verify smoothing type is valid
    if smoothing_type != "none" and smoothing_type != "add-one" and smoothing_type != "good-turing" and smoothing_type != "add-one-fast":
        sys.exit("ERROR: Incorrect smoothing type: " + smoothing_type +
                 "\n\n\tSmoothing type must be either:\n\t(1) none\n\t(2) add-one\n\t(3) good-turing\n\t(4) add-one-fast")
    
    # Get training set file lines
    trainingSet = open("TrainingSet.txt")
    lines = trainingSet.readlines()
    trainingSet.close()
    
    # Initialize dicts for unigram and bigrams
    unigramCount = dict()
    bigramCount = dict()
    unigramProb = dict()
    bigramProb = dict()
    
    # Initialize keys for start of sentence and end of sentence
    START_OF_SENTENCE = 0
    END_OF_SENTENCE = 1
    
    # Loop through lines in training set file
    for sentence in lines:
        
        # Start each sentence
        prevToken = None
        
        # Split each line by whitespace
        tokens = sentence.lower().split()
        
        # Loop through word_pos patterns (with start and end of sentence added)
        for t in [START_OF_SENTENCE] + tokens + [END_OF_SENTENCE]:
            
            # Get the word portion from the word_pos pattern
            token = t
            if token != START_OF_SENTENCE and token != END_OF_SENTENCE:
                token = t.split("_")[0]
            
            # Increment unigram counts by 1 for each token
            if token in unigramCount: unigramCount[token] += 1
            else: unigramCount.update({token: 1})
            
            # Increment bigram counts by 1 for each token|previous token
            if prevToken != None:
                if (token, prevToken) in bigramCount: bigramCount[(token, prevToken)] += 1
                else: bigramCount.update({(token, prevToken): 1})
            
            # Set previous token
            prevToken = token
            
    # Calculate unigram probabilities
    totalWordCount = sum(unigramCount.values())
    for token in unigramCount:
        unigramProb.update({token: unigramCount[token] / totalWordCount})
    
    # No smoothing
    if smoothing_type == "none":
        
        # Determine bigram probabilities
        for (token, prevToken) in bigramCount:
            bCount = bigramCount[(token, prevToken)]
            uCount = unigramCount[prevToken]
            bigramProb.update({(token, prevToken): bCount/uCount})
    
        # Add blank entry with probability 0 for unknown/unseen instances
        bigramProb.update({(None, None): 0})


    # Add-one smoothing
    elif smoothing_type == "add-one" or smoothing_type == "add-one-fast":
        
        # Get the size of vocabulary
        V = len(unigramCount.keys())
    
        # Determine bigram probabilities
        for (token, prevToken) in bigramCount:
            bCount = bigramCount[(token, prevToken)]
            uCount = unigramCount[prevToken]
            
            bigramProb.update({(token, prevToken): (bCount + 1)/(uCount + V)})
            
        # Add probabilities for all unseen bigrams
        if smoothing_type == "add-one":
            for prevToken in unigramCount:
                if prevToken != END_OF_SENTENCE:
                    for token in unigramCount:
                        if (token, prevToken) not in bigramProb and token != START_OF_SENTENCE:
                            bigramProb.update({(token, prevToken): 1/(unigramCount[prevToken] + V)})
        
        
    # Good-turing discounting based smoothing
    else:
        
        # Get total number of bigram occurrences 
        N = sum(bigramCount.values())
        
        # Create a dict for N_c values
        Nc = dict()
        
        # Calculate N_1
        n_i = sum(c == 1 for c in bigramCount.values())
        Nc.update({1: n_i})
        
        # Determine bigram probabilities
        for (token, prevToken) in bigramCount:
            bCount = bigramCount[(token, prevToken)]
            
            # If we have yet to calculate N_bCount, calculate it
            if bCount not in Nc.keys():
                n_i = sum(c == bCount for c in bigramCount.values())
                Nc.update({bCount: n_i})
            
            # If we have yet to calculate N_bCount+1, calculate it
            if bCount+1 not in Nc.keys():
                n_i = sum(c == bCount+1 for c in bigramCount.values())
                Nc.update({bCount+1: n_i})
            
            # Calculate c*
            newCount = 0
            if Nc[bCount] != 0: newCount = (bCount+1)*Nc[bCount+1] / Nc[bCount]
            
            # Add the probability
            bigramProb.update({(token, prevToken): newCount/N})
        
        # Add blank entry with probability N_1/N for unknown/unseen instances       
        bigramProb.update({(None, None): Nc[1]/N})
        
        
    # If a test file is provided, perform testing
    if test_file != None:
        
        # Get test set file lines
        testSet = open(test_file)
        lines = testSet.readlines()
        testSet.close()
        
        # Get sentence from test set file -> should only be one line
        sentence = lines[0]
        
        # Split line by whitespace
        tokens = sentence.lower().split()
        
        # Initialize values
        prevToken = None
        prob = 1
        
        # Loop through word_pos patterns (with start and end of sentence added)
        for t in [START_OF_SENTENCE] + tokens + [END_OF_SENTENCE]:
            
            # Get the word portion from the word_pos pattern
            token = t
            if token != START_OF_SENTENCE and token != END_OF_SENTENCE:
                token = t.split("_")[0]
            
            # Calculate and multiply P(token|prevToken)
            if prevToken != None:
                
                # token|prevToken exists in the bigram
                if (token, prevToken) in bigramProb: prob *= bigramProb[(token, prevToken)]
                
                # token|prevToken does not exist in the bigram
                elif smoothing_type == "add-one-fast": prob *= 1/(unigramCount[prevToken] + V)
                else: prob *= bigramProb[(None, None)]
            
            # Set previous token
            prevToken = token
        
        # Output results
        print("Probability of test sentence = " + str(prob))
        
    # If no test file is provided, display unigram/bigram counts and probabilities
    else:
        print("UNIGRAM COUNTS//////////////////////////////////////////////////")
        for token in unigramCount:
            if token == START_OF_SENTENCE: print("C(<s>) = " + str(unigramCount[token]))
            elif token == END_OF_SENTENCE: print("C(</s>) = " + str(unigramCount[token]))
            else: print("C(" + str(token) + ") = " + str(unigramCount[token]))
            
        print("\n\nBIGRAM COUNTS//////////////////////////////////////////////////")
        for (token, prevToken) in bigramCount:
            if prevToken == START_OF_SENTENCE: print("C(" + str(token) + " | <s>) = " + str(bigramCount[(token, prevToken)]))
            elif token == END_OF_SENTENCE: print("C(</s> | " + str(prevToken) + ") = " + str(bigramCount[(token, prevToken)]))
            else: print("C(" + str(token) + " | " + str(prevToken) + ") = " + str(bigramCount[(token, prevToken)]))
            
        print("\n\nUNIGRAM PROBABILITIES//////////////////////////////////////////")
        for token in unigramProb:
            if token == START_OF_SENTENCE: print("P(<s>) = " + str(unigramProb[token]))
            elif token == END_OF_SENTENCE: print("P(</s>) = " + str(unigramProb[token]))
            else: print("P(" + str(token) + ") = " + str(unigramProb[token]))
            
        print("\n\nBIGRAM PROBABILITIES///////////////////////////////////////////")
        for (token, prevToken) in bigramProb:
            if prevToken == START_OF_SENTENCE: print("P(" + str(token) + " | <s>) = " + str(bigramProb[(token, prevToken)]))
            elif token == END_OF_SENTENCE: print("P(</s> | " + str(prevToken) + ") = " + str(bigramProb[(token, prevToken)]))
            elif token == None: print("For all unseen bigrams: Probability = " + str(bigramProb[(token, prevToken)]))
            else: print("P(" + str(token) + " | " + str(prevToken) + ") = " + str(bigramProb[(token, prevToken)]))