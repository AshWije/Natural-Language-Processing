"""
tagging.py
    Description:
        This program builds a part of speech (POS) tagging model from a given training corpus
        and displays all bigram counts and probabilities of the model. This built model can
        then be used for POS tagging a given test sentence.
    
    Instructions:
        For displaying the model, run:
            python tagging.py
        
        For testing the model, run:
            python tagging.py <input-test-file>
        
        <input-test-file> must be a file containing the test input sentence as a single line.
    
    Imports:
        sys - Used to get the arguments from command line
"""
import sys

if __name__ == "__main__":
    
    # Print an error for the incorrect number of arguments
    if len(sys.argv) > 2:
        sys.exit("ERROR: Incorrect number of arguments.\n\n\tPlease execute with either: \n\t(1) python tagging.py\n\t(2) python tagging.py <input-test-file>")
    
    # Get arguments:
    #   test_file = location of input test file (optional)
    test_file = None
    if len(sys.argv) == 2:
        test_file = sys.argv[1]
        
    # Get training set file lines
    trainingSet = open("TrainingSet.txt")
    lines = trainingSet.readlines()
    trainingSet.close()
    
    # Initialize dicts for unigram and bigrams
    unigramTagCount = dict()    # key = <tag>
    bigramCount = dict()        # key = (<word|tag>, <previousTag>, <W|T>)
    bigramProb = dict()         # key = (<word|tag>, <previousTag>, <W|T>)
    
    # Initialize keys for start of sentence and end of sentence
    START_OF_SENTENCE = '<s>'
    END_OF_SENTENCE = '</s>'
    
    # Initialize set of tags
    tags = dict()
    
    ###########################################################################
    # Loop through lines in training set file
    for sentence in lines:
        
        # Start each sentence
        prevTag = START_OF_SENTENCE
        
        # Split each line by whitespace
        tokens = sentence.split()
        
        # Loop through word_pos patterns
        for token in tokens:
            
            # Extract the token and tag from the word_pos pattern
            word_pos = token.split("_")
            word = word_pos[0]
            tag = word_pos[1]
            
            # Add tag to set of tags
            if word in tags: tags[word].add(tag)
            else: tags.update({word: set(tag)})
            
            # Increment unigram counts by 1 for each tag
            if tag in unigramTagCount: unigramTagCount[tag] += 1
            else: unigramTagCount.update({tag: 1})
            
            # Increment bigram counts by 1 for each word|tag
            if (word, tag, 'W') in bigramCount: bigramCount[(word, tag, 'W')] += 1
            else: bigramCount.update({(word, tag, 'W'): 1})
            
            # Increment bigram counts by 1 for each tag|previous tag
            if (tag, prevTag, 'T') in bigramCount: bigramCount[(tag, prevTag, 'T')] += 1
            else: bigramCount.update({(tag, prevTag, 'T'): 1})
            
            # Set previous token
            prevTag = tag
            
        # Increment unigram count by 1 for <s>
        if START_OF_SENTENCE in unigramTagCount: unigramTagCount[START_OF_SENTENCE] += 1
        else: unigramTagCount.update({START_OF_SENTENCE: 1})
        
        # Increment bigram count by 1 for </s>|previous tag
        if (END_OF_SENTENCE, prevTag, 'T') in bigramCount: bigramCount[(END_OF_SENTENCE, prevTag, 'T')] += 1
        else: bigramCount.update({(END_OF_SENTENCE, prevTag, 'T'): 1})
            
    ###########################################################################
    # With no smoothing:
    #   Determine bigram probabilities
    for (token, tag, WT) in bigramCount:
        bCount = bigramCount[(token, tag, WT)]
        uCount = unigramTagCount[tag]
        bigramProb.update({(token, tag, WT): bCount/uCount})

    #   Add blank entry with probability 0 for unknown/unseen instances
    bigramProb.update({(None, None, None): 0})
        
    ###########################################################################
    # If a test file is provided, perform testing
    if test_file != None:
        
        # Get test set file lines
        testSet = open(test_file)
        lines = testSet.readlines()
        testSet.close()
        
        # Get sentence from test set file -> should only be one line
        sentence = lines[0]
        
        # Split line by whitespace
        words = sentence.split()
        
        # Initialize list of dicts with key = (<tag>, <prev-tag>)
        prob = [dict({(START_OF_SENTENCE, None): 1})]
        
        # Determine most common tag (to be used if word is unseen -> this should never happen)
        most_common_tag = max(unigramTagCount, key=unigramTagCount.get)
        
        #######################################################################
        # Loop through words patterns
        for i in range(len(words)):
            
            # Get word
            word = words[i]
                
            # Initialize probability dict for current word
            cur_prob = dict()
            
            # Get list of tags for word
            if word in tags: word_tags = list(tags[word])
            
            # If word is unseen, select most common tag -> this should never happen
            else:
                
                # Loop through probability values from previous word
                for prevTag, prevPrevTag in prob[-1]:
                     
                    # Update probability value to be previous probability
                    cur_prob.update({(most_common_tag, prevTag): prob[(prevTag, prevPrevTag)]})
                
                # Add probability dict for current word and skip
                prob.append(cur_prob)
                continue
            
            # Loop through probability values from previous word
            for prevTag, prevPrevTag in prob[-1]:
                    
                # Get previous word probability value
                prev_prob = prob[-1][(prevTag, prevPrevTag)]
            
                # Loop through tags for current word
                for tag in word_tags:
    
                    # Initialize current word's prob with previous' prob
                    cur_prob_val = prev_prob
                    
                    # Calculate P(word|tag) and multiply
                    if (word, tag, 'W') in bigramProb:
                        cur_prob_val *= bigramProb[(word, tag, 'W')]
                        
                    # If unseen word|tag -> probability = 0 and skip
                    else: continue
                    
                    # Calculate P(tag|prevTag) and multiply
                    if (tag, prevTag, 'T') in bigramProb:
                        cur_prob_val *= bigramProb[(tag, prevTag, 'T')]
                        
                    # If unseen tag|prevTag -> probability = 0 and skip
                    else: continue
                    
                    # If word is the last word in the sentence:
                    if i == (len(words) - 1):
                        
                        # Calculate P(</s>|tag)
                        if (END_OF_SENTENCE, tag, 'T') in bigramProb:
                            cur_prob_val *= bigramProb[(END_OF_SENTENCE, tag, 'T')]
                            
                        # If unseen </s>|tag -> probability = 0 and skip
                        else: continue
                    
                    # Add probability to dict (will be nonzero)
                    cur_prob.update({(tag, prevTag): cur_prob_val})
                    
            # Add dict for current word
            prob.append(cur_prob)
                
        # Get <tag> and <prevTag> of highest probability for last word
        best_tag, prevTag = max(prob[-1], key=prob[-1].get)
        
        # Initialize output message with last word
        out = word + "_" + best_tag
        
        # Loop through previous prob and words
        for i in reversed(range(len(words) - 1)):
            
            # Get word
            word = words[i]
            
            # Get prob dict for word (i+1 to avoid <s>)
            word_prob = prob[i + 1]
            
            # Get list of keys corresponding to next best tag
            word_prob_list = [(t, pt) for t, pt in word_prob if t==prevTag]
            
            # Get <tag> and <prevTag> of highest probability
            if len(word_prob_list) > 0:
                best_tag, prevTag = max(word_prob_list, key=word_prob.get)
            
            # If all probabilities are zero, select most common tag -> this should never happen
            else: best_tag, prevTag = prevTag, most_common_tag
            
            # Update output message
            out = word + "_" + best_tag + " " + out
        
        # Output results
        print("RESULT:\n" + out)
        
    ###########################################################################
    # If no test file is provided, display bigram counts and probabilities
    else:
        print("\n\nBIGRAM COUNTS//////////////////////////////////////////////////")
        for (token, prevToken, WT) in bigramCount:
            if prevToken == START_OF_SENTENCE: print("C(" + str(token) + " | <s>) = " + str(bigramCount[(token, prevToken, WT)]))
            elif token == END_OF_SENTENCE: print("C(</s> | " + str(prevToken) + ") = " + str(bigramCount[(token, prevToken, WT)]))
            else: print("C(" + str(token) + " | " + str(prevToken) + ") = " + str(bigramCount[(token, prevToken, WT)]))
            
        print("\n\nBIGRAM PROBABILITIES///////////////////////////////////////////")
        for (token, prevToken, WT) in bigramProb:
            if prevToken == START_OF_SENTENCE: print("P(" + str(token) + " | <s>) = " + str(bigramProb[(token, prevToken, WT)]))
            elif token == END_OF_SENTENCE: print("P(</s> | " + str(prevToken) + ") = " + str(bigramProb[(token, prevToken, WT)]))
            elif token == None: print("For all unseen bigrams: Probability = " + str(bigramProb[(token, prevToken, WT)]))
            else: print("P(" + str(token) + " | " + str(prevToken) + ") = " + str(bigramProb[(token, prevToken, WT)]))