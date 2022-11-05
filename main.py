import nltk, string
from nltk.util import bigrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import sys

# helper function remove_items
def remove_items(test_list, item):
  '''
  https://www.geeksforgeeks.org/remove-all-the-occurrences-of-an-      element-from-a-list-in-python/
  '''
  res = [i for i in test_list if i != item]
  return res
# helper function to split text
def split_text_70(test_str):
  ''' split the text file into 70% of original text
  https://www.geeksforgeeks.org/python-split-given-string-into-equal-halves/
  '''
  res_first = test_str[:int(len(test_str)*0.7)]
  return res_first
# helper function to split text
def split_text_30(test_str):
  ''' split the text file into the last 30% of original text
  https://www.geeksforgeeks.org/python-split-given-string-into-equal-halves/
  '''
  res_second = test_str[int(len(test_str)*0.7):]
  return res_second



def test_flag(test_file):
  '''
  parameters: test_file, a file given by the user as a CLA
  return: none, should print what the language model classifies as correct
  '''
  author_list_file = open("authorlist.txt", "r")
  authorlist = author_list_file.readlines()
  lm_list = []
  print("training LMs... (this may take a while)")
  for author in authorlist:
    #open author file, ie austen.txt
    author.replace("\n", "")
    authorfile = open(author, "r")
    authordata = authorfile.readlines()
    # strip text of punctuation
    #remove newlines
    for sentence in authordata:
      sentence = sentence.translate(str.maketrans('', '', string.punctuation))
      remove_items(sentence, " ")
      remove_items(sentence, "\n")
    
    
    #create bigrams
    list(flatten(pad_both_ends(bigrams(authordata, n=2) 
    for sentence in authordata)))
    train, vocab = padded_everygram_pipeline(2, authordata)
    #create and train language model
    lm = MLE(2)
    lm.fit(train, vocab)
    lm_list.append(lm)
    #model is now trained


  #open testfile
  our_test_file = open(test_file, "r")
  #tokenize test files
  data = our_test_file.split(".")

#test perplexity for each sentence in the test file  
  for line in data:
    #iterate through the lines in the test file
    perplexity_list = []
    # create a list to store the perplexity of the model for each line
    for model in lm_list:
      #iterate through our list of language models
      #calculate perplexity and add it to the list
      perplexity = model.perplexity(line)
      perplexity_list.append(perplexity)
  # after each sentence, print the model with the lowest perplexity
  if((min(perplexity_list)) == (perplexity_list[0])):
      print("austen" + "\n")
  elif ((min(perplexity_list)) == (perplexity_list[1])):
      print("dickens"+"\n")
  elif ((min(perplexity_list)) == (perplexity_list[2])):
      print("tolstoy"+"\n")
  elif ((min(perplexity_list)) == (perplexity_list[3])):
      print("wilde"+"\n")
     
def no_test_flag():
  '''
  input: none
  return: none (should print out the percentages correct for each LM)
  '''
  author_list_file = open("authorlist.txt", "r")
  authorlist = author_list_file.readlines()
  lm_list = []
  dev_set = []
  print("training LMs... (this may take a while)")
  for author in authorlist:
    #open author file, ie austen.txt
    author = author.replace("\n", "")
    authorfile = open(author, "r")
    data = authorfile.read()
    # split the author's text into dev and training set
    training_item = split_text_70(data)
    dev_set_item = split_text_30(data)
    
    
    #split the 2 sets into sentences
    training_item = training_item.split(".")
    dev_set_item = dev_set_item.split(".")
    # strip punctuation and newlines for each sentence in each set
    
    for sentence in training_item:
      sentence = sentence.translate(str.maketrans('', '', string.punctuation))
      remove_items(sentence, " ")
      remove_items(sentence, "\n")
    for sentence in dev_set_item:
      sentence = sentence.translate(str.maketrans('', '', string.punctuation))
      remove_items(sentence, " ")
      remove_items(sentence, "\n")
    #add the dev set item to a list
    dev_set.append(dev_set_item)
    
    #create bigrams
    flatten(pad_both_ends(item, n=2) for item in training_item)
    
    #train the lm
    train, vocab = padded_everygram_pipeline(2, training_item)
    lm = MLE(2)
    lm.fit(train, vocab)
    lm_list.append(lm)
   
    
# test perplexity of each sentence in each set in the dev set
  dev_set_len_list = []
# find the number of sentences in each set in the dev set
  for item in dev_set:
    ourlen = len(item)
    dev_set_len_list.append(ourlen)

  austen_counter = 0
  dickens_counter = 0
  tolstoy_counter = 0
  wilde_counter = 0
  
  for list in dev_set:
    #iterate through all sets in dev set
    for sent in list:
      #iterate over the number of sentences in each list 
      perplexity_list = []
      for model in lm_list:
      #iterate over all 4 language models
      #check if item length > 0
        perplexity = model.perplexity(sent)
        perplexity_list.append(perplexity)
      
    if((min(perplexity_list)) == (perplexity_list[0])):
           austen_counter+=1
    elif ((min(perplexity_list)) == (perplexity_list[1])):
           dickens_counter+=1
    elif ((min(perplexity_list)) == (perplexity_list[2])):
           tolstoy_counter+=1
    elif ((min(perplexity_list)) == (perplexity_list[3])):
           wilde_counter+=1
      
  print("Results on dev set:")
  print("austen      " + (austen_counter/dev_set_len_list[0]) + " correct\n")
  print("dickens      " + (dickens_counter/dev_set_len_list[1]) + " correct\n")
  print("tolstoy      " + (tolstoy_counter/dev_set_len_list[2]) + " correct\n")
  print("wilde      " + (wilde_counter/dev_set_len_list[3]) + " correct\n")

               
#print(len(sys.argv))
if len(sys.argv) < 3:
   no_test_flag()
else:
  test_flag(sys.argv[3])






