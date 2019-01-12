# a3

Part 1: Part-of-speech tagging

Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at least the 1950’s. One of the most basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step towards extracting semantics from natural language text. For example, consider the following sentence: Her position covers a number of daily tasks common to any social director. Part-of-speech tagging here is not easy because many of these words can take on different parts of speech depending on context. For example, position can be a noun (as in the above sentence) or a verb (as in “They position themselves near the exit”). 

The correct labeling for the above sentence is: Her	position covers	a number of daily tasks common to any social director. DET NOUN	VERB DET NOUN ADP ADJ NOUN ADJ ADP DET ADJ NOUN where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an adverb. Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence, as well as the relationships between the words.

Data: The dataset is a large corpus of labeled training and testing data, consisting of nearly 1 million words and 50,000 sentences. The file format of the datasets is: each line consists of a word, followed by a space, followed by one of 12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). Sentence boundaries are indicated by blank lines. 

label.py is the main program, pos scorer.py, which has the scoring code, and pos solver.py, which contains the actual part-of-speech estimation code. The program takes as input two filenames: a training file and a testing file and displays accuracy using simple probability, Bayes net variable elimination method and Viterbi algorithm to find the maximum a posteriori (MAP). 

It also displays the logarithm of the posterior probability for each solution it finds, as well as a running evaluation showing the percentage of words and whole sentences that have been labeled correctly according to the ground truth. 

To run the code:
python label.py part2 training_file testing_file

Part 2: Optical Character Recognition (OCR)

Here we have utilized the versatility of Hidden Markov Models (HMMs) and applied them to the problem of Optical Character Recognition. Modern OCR is very good at recognizing documents, but rather poor when recognizing isolated characters. It turns out that the main reason for OCR’s success is that there’s a strong language model: the algorithm can resolve ambiguities in recognition by using statistical constraints of English (or whichever language is being processed). These constraints can be incorporated very naturally using an HMM. Our goal here is to recognize text in an image where the font and font size is known ahead of time.

Data: A text string image is divided into little sub-images corresponding to individual letters; a real OCR system has to do this letter segmentation automatically, but here we’ll assume a fixed-width font so that we know exactly where each letter begins and ends ahead of time. In particular, we’ll assume each letter fits in a box that’s 14 pixels wide and 25 pixels tall. We’ll also assume that our documents only have the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation symbols, (),.-!?’". Suppose we’re trying to recognize a text string with n characters, so we have n observed variables (the subimage corresponding to each letter) O1 , ..., On and n hidden variables, l1 ..., ln , which are the letters we want to recognize. We’re thus interested in P(l1 , ..., ln |O1 , ..., On ). We can rewrite this using Bayes’ Law, estimate P(Oi |li) and P(li |li−1 ) from training data, then use probabilistic inference to estimate the posterior, in order to recognize letters. 

The program loads the image file, which contains images of letters to use for training. It also loads the text training file, which is simply some text document that is representative of the language (English, in this case) that will be recognized. Then, it uses the classifier it has learned to detect the text in test-image-file.png, using (1) simple Bayes net, and (2) Hidden Markov Model with MAP inference (Viterbi). The output displays the text as recognized using the above approaches.

The program is called like this: ./ocr.py train-image-file.png train-text.txt test-image-file.png
